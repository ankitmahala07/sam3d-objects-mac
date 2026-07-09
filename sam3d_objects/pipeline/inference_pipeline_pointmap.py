# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Union, Optional
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
import torchvision
from loguru import logger
from PIL import Image

try:
    from pytorch3d.renderer import look_at_view_transform
    from pytorch3d.transforms import Transform3d
except ImportError:
    look_at_view_transform = None
    Transform3d = None

from sam3d_objects.model.backbone.dit.embedder.pointmap import PointPatchEmbed
from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import (
    get_mask,
)
from sam3d_objects.data.dataset.tdfy.transforms_3d import (
    DecomposedTransform,
)
from sam3d_objects.pipeline.utils.pointmap import infer_intrinsics_from_pointmap
from sam3d_objects.pipeline.inference_utils import o3d_plane_estimation, estimate_plane_area, layout_post_optimization, layout_post_optimization_method_GS


def camera_to_pytorch3d_camera(device="cpu") -> DecomposedTransform:
    """
    R3 camera space --> PyTorch3D camera space
    Also needed for pointmaps
    """
    r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
        device=device,
    )
    return DecomposedTransform(
        rotation=r3_to_p3d_R,
        translation=r3_to_p3d_T,
        scale=torch.tensor(1.0, dtype=r3_to_p3d_R.dtype, device=device),
    )


def recursive_fn_factory(fn):
    def recursive_fn(b):
        if isinstance(b, dict):
            return {k: recursive_fn(b[k]) for k in b}
        if isinstance(b, list):
            return [recursive_fn(t) for t in b]
        if isinstance(b, tuple):
            return tuple(recursive_fn(t) for t in b)
        if isinstance(b, torch.Tensor):
            return fn(b)
        # Yes, writing out an explicit white list of
        # trivial types is tedious, but so are bugs that
        # come from not applying fn, when expected to have
        # applied it.
        if b is None:
            return b
        trivial_types = [bool, int, float]
        for t in trivial_types:
            if isinstance(b, t):
                return b
        raise TypeError(f"Unexpected type {type(b)}")

    return recursive_fn


recursive_contiguous = recursive_fn_factory(lambda x: x.contiguous())
recursive_clone = recursive_fn_factory(torch.clone)


def compile_wrapper(
    fn, *, mode="max-autotune", fullgraph=True, dynamic=False, name=None
):
    compiled_fn = torch.compile(fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    def compiled_fn_wrapper(*args, **kwargs):
        with torch.autograd.profiler.record_function(
            f"compiled {fn}" if name is None else name
        ):
            cont_args = recursive_contiguous(args)
            cont_kwargs = recursive_contiguous(kwargs)
            result = compiled_fn(*cont_args, **cont_kwargs)
            cloned_result = recursive_clone(result)
            return cloned_result

    return compiled_fn_wrapper


class InferencePipelinePointMap(InferencePipeline):

    def __init__(
        self, *args, depth_model, layout_post_optimization_method=layout_post_optimization, layout_post_optimization_method_GS=layout_post_optimization_method_GS, clip_pointmap_beyond_scale=None, **kwargs
    ):
        self.depth_model = depth_model
        self.layout_post_optimization_method = layout_post_optimization_method
        self.layout_post_optimization_method_GS = layout_post_optimization_method_GS
        self.clip_pointmap_beyond_scale = clip_pointmap_beyond_scale
        super().__init__(*args, **kwargs)

    def _compile(self):
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        torch._dynamo.config.capture_scalar_outputs = True
        compile_mode = "max-autotune"

        for embedder, _ in self.condition_embedders[
            "ss_condition_embedder"
        ].embedder_list:
            if isinstance(embedder, PointPatchEmbed):
                logger.info("Found PointPatchEmbed")
                embedder.inner_forward = compile_wrapper(
                    embedder.inner_forward,
                    mode=compile_mode,
                    fullgraph=True,
                )
            else:
                embedder.forward = compile_wrapper(
                    embedder.forward,
                    mode=compile_mode,
                    fullgraph=True,
                )

        self.models["ss_generator"].reverse_fn.inner_forward = compile_wrapper(
            self.models["ss_generator"].reverse_fn.inner_forward,
            mode=compile_mode,
            fullgraph=True,
        )

        self.models["ss_decoder"].forward = compile_wrapper(
            self.models["ss_decoder"].forward,
            mode=compile_mode,
            fullgraph=True,
        )

        self._warmup()

    def _warmup(self, num_warmup_iters=3):
        test_image = np.ones((512, 512, 4), dtype=np.uint8) * 255
        test_image[:, :, :3] = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(test_image)
        mask = None
        image = self.merge_image_and_mask(image, mask)
        with torch.inference_mode(False):
            with torch.no_grad():
                for _ in tqdm(range(num_warmup_iters)):
                    pointmap_dict = recursive_clone(self.compute_pointmap(image))
                    pointmap = pointmap_dict["pointmap"]

                    ss_input_dict = self.preprocess_image(
                        image, self.ss_preprocessor, pointmap=pointmap
                    )
                    ss_return_dict = self.sample_sparse_structure(
                        ss_input_dict, inference_steps=None
                    )

                    _ = self.run_layout_model(
                        ss_input_dict,
                        ss_return_dict,
                        inference_steps=None,
                    )

    def _preprocess_image_and_mask_pointmap(
        self, rgb_image, mask_image, pointmap, img_mask_pointmap_joint_transform
    ):
        for trans in img_mask_pointmap_joint_transform:
            rgb_image, mask_image, pointmap = trans(
                rgb_image, mask_image, pointmap=pointmap
            )
        return rgb_image, mask_image, pointmap

    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray],
        preprocessor,
        pointmap=None,
    ) -> torch.Tensor:
        # canonical type is numpy
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        assert image.ndim == 3  # no batch dimension as of now
        assert image.shape[-1] == 4  # rgba format
        assert image.dtype == np.uint8  # [0,255] range

        rgba_image = torch.from_numpy(self.image_to_float(image))
        rgba_image = rgba_image.permute(2, 0, 1).contiguous()
        rgb_image = rgba_image[:3]
        rgb_image_mask = get_mask(rgba_image, None, "ALPHA_CHANNEL")

        preprocessor_return_dict = preprocessor._process_image_mask_pointmap_mess(
            rgb_image, rgb_image_mask, pointmap
        )
        
        # Put in a for loop?
        _item = preprocessor_return_dict
        item = {
            "mask": _item["mask"][None].to(self.device),
            "image": _item["image"][None].to(self.device),
            "rgb_image": _item["rgb_image"][None].to(self.device),
            "rgb_image_mask": _item["rgb_image_mask"][None].to(self.device),
        }

        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            item["pointmap"] = _item["pointmap"][None].to(self.device)
            item["rgb_pointmap"] = _item["rgb_pointmap"][None].to(self.device)
            item["pointmap_scale"] = _item["pointmap_scale"][None].to(self.device)
            item["pointmap_shift"] = _item["pointmap_shift"][None].to(self.device)
            item["rgb_pointmap_scale"] = _item["rgb_pointmap_scale"][None].to(self.device)
            item["rgb_pointmap_shift"] = _item["rgb_pointmap_shift"][None].to(self.device)

        # Add unnormed pointmap for post-optimization
        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            full_pointmap = self._apply_transform(
                pointmap, preprocessor.pointmap_transform
            )
            item["rgb_pointmap_unnorm"] = full_pointmap[None].to(self.device)            

        return item

    def _clip_pointmap(self, pointmap: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.clip_pointmap_beyond_scale is None:
            return pointmap

        pointmap_size = (pointmap.shape[1], pointmap.shape[2])
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask_resized = torchvision.transforms.functional.resize(
            mask, pointmap_size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        ).squeeze(0)

        pointmap_flat = pointmap.reshape(3, -1)
        # Get valid points from the mask
        mask_bool = mask_resized.reshape(-1) > 0.5
        mask_points = pointmap_flat[:, mask_bool]
        mask_distance = mask_points.nanmedian(dim=-1).values[-1]
        logger.info(f"mask_distance: {mask_distance}")
        pointmap_clipped_flat = torch.where(
            pointmap_flat[2, ...].abs() > self.clip_pointmap_beyond_scale * mask_distance,
            torch.full_like(pointmap_flat, float('nan')),
            pointmap_flat
        )
        pointmap_clipped = pointmap_clipped_flat.reshape(pointmap.shape)
        return pointmap_clipped

    def refine_scale(self, revised_scale):
        # Check if all three channels of revised_scale are close to each other
        if not torch.allclose(revised_scale[0, 0:1], revised_scale[0, 1:2], atol=1e-3) or \
           not torch.allclose(revised_scale[0, 0:1], revised_scale[0, 2:3], atol=1e-3):
            logger.warning(
                f"revised_scale values are not close (tolerance=1e-3): "
            )
        # Use 3-channel mean value
        revised_scale = revised_scale.clone()
        mean_val = revised_scale.mean(dim=1, keepdim=True)
        revised_scale[:] = mean_val

        return revised_scale

    def compute_pointmap(self, image, pointmap=None):
        loaded_image = self.image_to_float(image)
        loaded_image = torch.from_numpy(loaded_image)
        loaded_mask = loaded_image[..., -1]
        loaded_image = loaded_image.permute(2, 0, 1).contiguous()[:3]

        if pointmap is None:
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    output = self.depth_model(loaded_image)
            pointmaps = output["pointmaps"]
            camera_convention_transform = (
                Transform3d()
                .rotate(camera_to_pytorch3d_camera(device=self.device).rotation)
                .to(self.device)
            )
            points_tensor = camera_convention_transform.transform_points(pointmaps)
            intrinsics = output.get("intrinsics", None)
        else:
            output = {}
            points_tensor = pointmap.to(self.device)
            if loaded_image.shape != points_tensor.shape:
                # Interpolate points_tensor to match loaded_image size
                # loaded_image has shape [3, H, W], we need H and W
                points_tensor = torch.nn.functional.interpolate(
                    points_tensor.permute(2, 0, 1).unsqueeze(0),
                    size=(loaded_image.shape[1], loaded_image.shape[2]),
                    mode="nearest",
                ).squeeze(0).permute(1, 2, 0)
            intrinsics = None
        
        # Prepare the point map tensor
        point_map_tensor = {
            "pts_color": loaded_image,
        }

        # If depth model doesn't provide intrinsics, infer them
        if intrinsics is None:
            camera_convention_transform = (
                Transform3d()
                .rotate(camera_to_pytorch3d_camera(device=self.device).rotation)
                .to(self.device)
            )
            points_tensor_moge = camera_convention_transform.inverse().transform_points(points_tensor)
            intrinsics_result = infer_intrinsics_from_pointmap(
                points_tensor_moge, device=self.device
            )
            point_map_tensor["intrinsics"] = intrinsics_result["intrinsics"]
        else:
            point_map_tensor["intrinsics"] = intrinsics

        points_tensor = points_tensor.permute(2, 0, 1)
        points_tensor = self._clip_pointmap(points_tensor, loaded_mask)
        point_map_tensor["pointmap"] = points_tensor

        return point_map_tensor

    @torch.autograd.grad_mode.inference_mode(mode=False)
    def run_post_optimization(self, mesh_glb, intrinsics, pose_dict, layout_input_dict):
        intrinsics = intrinsics.clone()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        re_focal = min(fx, fy)
        intrinsics[0, 0], intrinsics[1, 1] = re_focal, re_focal
        revised_quat, revised_t, revised_scale, final_iou, _, _ = (
            self.layout_post_optimization_method(
                mesh_glb,
                pose_dict["rotation"],
                pose_dict["translation"],
                pose_dict["scale"],
                layout_input_dict["rgb_image_mask"][0, 0],
                layout_input_dict["rgb_pointmap_unnorm"][0].permute(1, 2, 0),
                intrinsics,
                Enable_shape_ICP=False,
                min_size=518,
                device=self.device,
            )
        )

        revised_scale = self.refine_scale(revised_scale)
        return {
            "rotation": revised_quat,
            "translation": revised_t,
            "scale": revised_scale,
            "iou": final_iou,
        }

    @torch.autograd.grad_mode.inference_mode(mode=False)
    def run_post_optimization_GS(self, gs_input, intrinsics, pose_dict, layout_input_dict, backend="gsplat"):
        intrinsics = intrinsics.clone()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        re_focal = min(fx, fy)
        intrinsics[0, 0], intrinsics[1, 1] = re_focal, re_focal

        revised_quat, revised_t, revised_scale, final_iou, initial_iou, _, flag_optim = (
            self.layout_post_optimization_method_GS(
                gs_input,
                pose_dict["rotation"],
                pose_dict["translation"],
                pose_dict["scale"],
                layout_input_dict["rgb_image_mask"][0, 0],
                layout_input_dict["rgb_image"][0],
                layout_input_dict["rgb_pointmap_unnorm"][0].permute(1, 2, 0),
                intrinsics,
                Enable_occlusion_check=False,
                Enable_manual_alignment=False,
                Enable_shape_ICP=False,
                Enable_rendering_optimization=True,
                min_size=518,
                device=self.device,
                backend=backend,
            )
        )

        revised_scale = self.refine_scale(revised_scale)
        return {
            "rotation": revised_quat,
            "translation": revised_t,
            "scale": revised_scale,
            "iou": final_iou,
            "iou_before_optim": initial_iou,
            "optim_accepted": flag_optim,
        }

    def _free_modules(self, model_keys=(), embedder_keys=(), drop_depth=False):
        """Permanently free models we won't use again this run, to lower peak
        memory. Only safe for a single-shot run (the pipeline can't run again
        afterwards) — used by the CLI, which processes one object then exits.
        On unified-memory Macs this physically returns RAM before the next
        stage's activations allocate."""
        import gc
        for k in model_keys:
            if k in self.models:
                del self.models[k]
        for k in embedder_keys:
            if self.condition_embedders.get(k) is not None:
                self.condition_embedders[k] = None
        if drop_depth:
            self.depth_model = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _assert_finite_tensor(self, tensor, label):
        if tensor is None or not torch.is_floating_point(tensor):
            return
        if not bool(torch.isfinite(tensor).all().detach().cpu().item()):
            raise FloatingPointError(f"{label} contains NaN/Inf; aborting before PLY generation")

    def _assert_finite_gaussian(self, gs, label="gaussian"):
        for name in ("_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"):
            self._assert_finite_tensor(getattr(gs, name, None), f"{label}.{name}")

    def _sparse_to_cpu(self, slat):
        return type(slat)(
            feats=slat.feats.detach().cpu(),
            coords=slat.coords.detach().cpu(),
            shape=slat.shape,
            layout=slat.layout,
        )

    def run(
        self,
        image: Union[None, Image.Image, np.ndarray],
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed: Optional[int] = None,
        stage1_only=False,
        free_stage_models=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        with_layout_postprocess=False,
        use_vertex_color=False,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
        use_stage1_distillation=False,
        use_stage2_distillation=False,
        pointmap=None,
        decode_formats=None,
        estimate_plane=False,
        fail_on_nan=False,
        return_pointmap=True,
        return_latents=True,
        condition_images=None,
        progress_callback=None,
    ) -> dict:
        def emit_progress(event, **payload):
            if progress_callback is not None:
                progress_callback(event, **payload)

        def stage_progress(label):
            if progress_callback is None:
                return None

            def _callback(current, total):
                emit_progress(
                    "diffusion",
                    label=label,
                    current=current,
                    total=total,
                )

            return _callback

        image = self.merge_image_and_mask(image, mask)
        condition_images = [
            self.merge_image_and_mask(condition_image, None)
            for condition_image in (condition_images or [])
        ]
        with self.device: 
            emit_progress("phase", label="Depth pointmap")
            pointmap_dict = self.compute_pointmap(image, pointmap)
            pointmap = pointmap_dict["pointmap"]

            if estimate_plane:
                return self.estimate_plane(pointmap_dict, image)

            pts = pts_colors = None
            if return_pointmap:
                pts = type(self)._down_sample_img(pointmap).cpu().permute((1, 2, 0))
                pts_colors = (
                    type(self)
                    ._down_sample_img(pointmap_dict["pts_color"])
                    .cpu()
                    .permute((1, 2, 0))
                )

            ss_input_dict = self.preprocess_image(
                image, self.ss_preprocessor, pointmap=pointmap
            )
            if condition_images:
                view_count = len(condition_images) + 1
                emit_progress("phase", label=f"Multi-view condition ({view_count} views)")

                def ss_condition_inputs():
                    yield ss_input_dict
                    for condition_image in condition_images:
                        condition_pointmap_dict = self.compute_pointmap(condition_image)
                        condition_pointmap = condition_pointmap_dict["pointmap"]
                        condition_input = self.preprocess_image(
                            condition_image,
                            self.ss_preprocessor,
                            pointmap=condition_pointmap,
                        )
                        del condition_pointmap_dict, condition_pointmap
                        yield condition_input
                        del condition_input

                self._ensure_condition_embedder("ss_condition_embedder")
                ss_condition = self.average_condition_embedding(
                    self.condition_embedders["ss_condition_embedder"],
                    ss_condition_inputs(),
                    self.ss_condition_input_mapping,
                    label="Stage 1 multi-view",
                    view_count=view_count,
                    autocast_dtype=self.shape_model_dtype,
                )
                if ss_condition is not None:
                    ss_input_dict["_condition_embedding"] = ss_condition
            emit_progress("advance", label="Depth + image prep", amount=1)
            # MoGe depth model is only used by compute_pointmap above.
            if free_stage_models and not estimate_plane:
                self._free_modules(drop_depth=True)
            if seed is not None:
                torch.manual_seed(seed)
            emit_progress("phase", label="Stage 1 sparse diffusion")
            ss_return_dict = self.sample_sparse_structure(
                ss_input_dict,
                inference_steps=stage1_inference_steps,
                use_distillation=use_stage1_distillation,
                progress_callback=stage_progress("Stage 1 sparse"),
            )

            # We could probably use the decoder from the models themselves
            pointmap_scale = ss_input_dict.get("pointmap_scale", None)
            pointmap_shift = ss_input_dict.get("pointmap_shift", None)
            ss_return_dict.update(
                self.pose_decoder(
                    ss_return_dict,
                    scene_scale=pointmap_scale,
                    scene_shift=pointmap_shift,
                )
            )

            logger.info(f"Rescaling scale by {ss_return_dict['downsample_factor']} after downsampling")
            ss_return_dict["scale"] = ss_return_dict["scale"] * ss_return_dict["downsample_factor"]
            emit_progress("advance", label="Pose + downsample", amount=1)

            if stage1_only:
                logger.info("Finished!")
                ss_return_dict["voxel"] = ss_return_dict["coords"][:, 1:] / 64 - 0.5
                ret = {**ss_return_dict}
                if return_pointmap:
                    ret.update({"pointmap": pts, "pointmap_colors": pts_colors})
                return ret
                # return ss_return_dict

            if not return_latents:
                ss_return_dict.pop("shape", None)
                ss_return_dict.pop("coords_original", None)

            coords = ss_return_dict["coords"]
            if not with_layout_postprocess:
                del ss_input_dict
                pointmap_dict = None
                pointmap = None
            # Stage 1 is done (sparse structure + pose). Free its models before
            # SLAT allocates — the biggest single memory win on 24 GB Macs.
            if free_stage_models:
                self._free_modules(
                    model_keys=("ss_generator", "ss_decoder", "ss_encoder"),
                    embedder_keys=("ss_condition_embedder",),
                )
            emit_progress("phase", label="Stage 2 setup")
            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            if condition_images:
                view_count = len(condition_images) + 1

                def slat_condition_inputs():
                    yield slat_input_dict
                    for condition_image in condition_images:
                        condition_input = self.preprocess_image(
                            condition_image, self.slat_preprocessor
                        )
                        yield condition_input
                        del condition_input

                self._ensure_condition_embedder("slat_condition_embedder")
                slat_condition = self.average_condition_embedding(
                    self.condition_embedders["slat_condition_embedder"],
                    slat_condition_inputs(),
                    self.slat_condition_input_mapping,
                    label="Stage 2 multi-view",
                    view_count=view_count,
                    autocast_dtype=self.dtype,
                )
                if slat_condition is not None:
                    slat_input_dict["_condition_embedding"] = slat_condition
            emit_progress("advance", label="Stage 2 setup", amount=1)
            emit_progress("phase", label="Stage 2 latent diffusion")
            slat = self.sample_slat(
                slat_input_dict,
                coords,
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
                progress_callback=stage_progress("Stage 2 latent"),
            )
            if fail_on_nan:
                self._assert_finite_tensor(slat.feats, "structured latent")
            # SLAT sampling is done; only the decoder is needed to decode it.
            if free_stage_models:
                self._free_modules(
                    model_keys=("slat_generator",),
                    embedder_keys=("slat_condition_embedder",),
                )
            emit_progress("phase", label="Decode gaussian")
            outputs = self.decode_slat(
                slat, self.decode_formats if decode_formats is None else decode_formats
            )
            if fail_on_nan:
                for idx, gs in enumerate(outputs.get("gaussian", ())):
                    self._assert_finite_gaussian(gs, f"gaussian[{idx}]")
            outputs = self.postprocess_slat_output(
                outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
            )
            if free_stage_models and not with_layout_postprocess:
                del slat_input_dict
                del coords
                slat = self._sparse_to_cpu(slat)
                for key in ("gaussian", "gaussian_4"):
                    for gs in outputs.get(key, ()) or ():
                        gs.to("cpu")
                for key, value in list(ss_return_dict.items()):
                    if isinstance(value, torch.Tensor):
                        ss_return_dict[key] = value.detach().cpu()
                self._free_modules(
                    model_keys=("slat_decoder_gs", "slat_decoder_gs_4", "slat_decoder_mesh")
                )
            emit_progress("advance", label="Decode gaussian", amount=1)
            glb = outputs.get("glb", None)
            gs_input = outputs.get("gaussian", None)

            try:
                if with_layout_postprocess:
                    if (
                        gs_input is not None
                        and self.layout_post_optimization_method_GS is not None
                    ):
                        logger.info("Running GS layout post optimization method...")
                        postprocessed_pose = self.run_post_optimization_GS(
                            deepcopy(gs_input[0]),
                            pointmap_dict["intrinsics"],
                            ss_return_dict,
                            ss_input_dict,
                            backend="gsplat",
                        )
                        ss_return_dict.update(postprocessed_pose)
                        logger.info(f"Finished GS post-optimization!")
                    elif (
                        glb is not None
                        and self.layout_post_optimization_method is not None
                    ):
                        logger.info("Running mesh layout post optimization method...")
                        postprocessed_pose = self.run_post_optimization(
                            deepcopy(glb),
                            pointmap_dict["intrinsics"],
                            ss_return_dict,
                            ss_input_dict,
                        )
                        ss_return_dict.update(postprocessed_pose)
                        logger.info("Finished mesh post-optimization!")
                    else:
                        logger.info("No post-optimization method available (no GS or mesh found)")
            except Exception as e:
                logger.error(
                    f"Error during layout post optimization: {e}", exc_info=True
                )

            logger.info("Finished!")

            ret = {
                **ss_return_dict,
                **outputs,
                "slat": slat,
            }
            if return_pointmap:
                ret.update({"pointmap": pts, "pointmap_colors": pts_colors})
            return ret

    @staticmethod
    def _down_sample_img(img_3chw: torch.Tensor):
        # img_3chw: (3, H, W)
        x = img_3chw.unsqueeze(0)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        max_side = max(x.shape[2], x.shape[3])
        scale_factor = 1.0

        # heuristics
        if max_side > 3800:
            scale_factor = 0.125
        if max_side > 1900:
            scale_factor = 0.25
        elif max_side > 1200:
            scale_factor = 0.5

        x = torch.nn.functional.interpolate(
            x,
            scale_factor=(scale_factor, scale_factor),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )  # -> (1, 3, H/4, W/4)
        return x.squeeze(0)

    def estimate_plane(self, pointmap_dict, image, ground_area_threshold=0.25, min_points=100):
        assert image.shape[-1] == 4  # rgba format
        # Extract mask from alpha channel
        floor_mask = type(self)._down_sample_img(torch.from_numpy(image[..., -1]).float().unsqueeze(0))[0] > 0.5
        pts = type(self)._down_sample_img(pointmap_dict["pointmap"])

        # Get all points in 3D space (H, W, 3)
        pts_hwc = pts.cpu().permute((1, 2, 0))

        valid_mask_points = floor_mask.cpu().numpy()
        # Extract points that fall within the mask
        if valid_mask_points.any():
            # Get points within mask
            masked_points = pts_hwc[valid_mask_points]
            # Filter out invalid points (zero points from depth estimation failures)
            valid_points_mask = torch.norm(masked_points, dim=-1) > 1e-6
            valid_points = masked_points[valid_points_mask]
            points = valid_points.numpy()
        else:
            points = np.array([]).reshape(0, 3)
     
        # Calculate area coverage and check num of points
        overlap_area = estimate_plane_area(floor_mask)
        has_enough_points = len(points) >= min_points

        logger.info(f"Plane estimation: {len(points)} points, {overlap_area:.3f} area coverage")
        if overlap_area > ground_area_threshold and has_enough_points:
            try:
                mesh = o3d_plane_estimation(points)
                logger.info("Successfully estimated plane mesh")
            except Exception as e:
                logger.error(f"Failed to estimate plane: {e}")
                mesh = None
        else:
            logger.info(f"Skipping plane estimation: area={overlap_area:.3f}, points={len(points)}")
            mesh = None

        return {
            "glb": mesh,
            "translation": torch.tensor([[0.0, 0.0, 0.0]]),
            "scale": torch.tensor([[1.0, 1.0, 1.0]]),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        }
