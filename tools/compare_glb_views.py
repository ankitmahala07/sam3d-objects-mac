#!/usr/bin/env python3
"""Compare mesh.glb and mesh_game.glb from 12 fixed directions.

This is a CPU-only geometry comparator. It writes silhouette overlays and depth
error images so the game mesh can be checked without an OpenGL renderer.
"""

import argparse
import json
import math
import os

import numpy as np
from PIL import Image
import trimesh


def load_mesh(path):
    asset = trimesh.load(path, force="scene")
    if isinstance(asset, trimesh.Trimesh):
        return asset
    if hasattr(asset, "to_geometry"):
        try:
            mesh = asset.to_geometry()
            if isinstance(mesh, trimesh.Trimesh):
                return mesh
        except Exception:
            pass
    try:
        mesh = asset.dump(concatenate=True)
        if isinstance(mesh, trimesh.Trimesh):
            return mesh
    except Exception:
        pass
    meshes = list(asset.geometry.values())
    if not meshes:
        raise RuntimeError(f"No mesh geometry found: {path}")
    return trimesh.util.concatenate(meshes)


def view_basis(direction):
    forward = np.asarray(direction, dtype=np.float64)
    forward /= max(np.linalg.norm(forward), 1e-12)
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(forward, up))) > 0.95:
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(up, forward)
    right /= max(np.linalg.norm(right), 1e-12)
    true_up = np.cross(forward, right)
    true_up /= max(np.linalg.norm(true_up), 1e-12)
    return right, true_up, forward


def project(vertices, center, scale, direction, resolution):
    right, up, forward = view_basis(direction)
    local = (vertices - center) / scale
    x = local @ right
    y = local @ up
    z = local @ forward
    padding = 0.58
    px = (x / padding * 0.5 + 0.5) * (resolution - 1)
    py = (0.5 - y / padding * 0.5) * (resolution - 1)
    return np.column_stack([px, py, z])


def rasterize(mesh, center, scale, direction, resolution):
    points = project(np.asarray(mesh.vertices), center, scale, direction, resolution)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    depth = np.full((resolution, resolution), -np.inf, dtype=np.float32)
    mask = np.zeros((resolution, resolution), dtype=bool)

    for face in faces:
        tri = points[face]
        min_xy = np.floor(np.min(tri[:, :2], axis=0)).astype(int)
        max_xy = np.ceil(np.max(tri[:, :2], axis=0)).astype(int)
        min_x = max(0, min_xy[0])
        min_y = max(0, min_xy[1])
        max_x = min(resolution - 1, max_xy[0])
        max_y = min(resolution - 1, max_xy[1])
        if min_x > max_x or min_y > max_y:
            continue

        x0, y0 = tri[0, 0], tri[0, 1]
        x1, y1 = tri[1, 0], tri[1, 1]
        x2, y2 = tri[2, 0], tri[2, 1]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denom)) < 1e-10:
            continue

        xs = np.arange(min_x, max_x + 1)
        ys = np.arange(min_y, max_y + 1)
        grid_x, grid_y = np.meshgrid(xs, ys)
        sample_x = grid_x + 0.5
        sample_y = grid_y + 0.5
        w0 = ((y1 - y2) * (sample_x - x2) + (x2 - x1) * (sample_y - y2)) / denom
        w1 = ((y2 - y0) * (sample_x - x2) + (x0 - x2) * (sample_y - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not bool(inside.any()):
            continue
        tri_depth = w0 * tri[0, 2] + w1 * tri[1, 2] + w2 * tri[2, 2]
        target_depth = depth[min_y:max_y + 1, min_x:max_x + 1]
        update = inside & (tri_depth > target_depth)
        target_depth[update] = tri_depth[update]
        mask[min_y:max_y + 1, min_x:max_x + 1][update] = True

    return mask, depth


def save_overlay(source_mask, game_mask, depth_error, out_path):
    image = np.zeros((*source_mask.shape, 3), dtype=np.uint8)
    image[source_mask, 0] = 220
    image[game_mask, 1] = 220
    image[game_mask, 2] = 220
    both = source_mask & game_mask
    image[both] = np.asarray([220, 220, 220], dtype=np.uint8)
    edge = depth_error > 0.02
    image[edge] = np.asarray([255, 180, 0], dtype=np.uint8)
    Image.fromarray(image).save(out_path)


def compare(source_path, game_path, out_dir, resolution):
    os.makedirs(out_dir, exist_ok=True)
    source = load_mesh(source_path)
    game = load_mesh(game_path)
    bounds = np.vstack([source.bounds, game.bounds])
    center = bounds.reshape(-1, 3).mean(axis=0)
    scale = max(float(np.linalg.norm(bounds.max(axis=0) - bounds.min(axis=0))), 1e-8)

    results = []
    for idx in range(12):
        angle = math.tau * idx / 12.0
        direction = np.asarray([math.cos(angle), 0.22, math.sin(angle)], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        source_mask, source_depth = rasterize(source, center, scale, direction, resolution)
        game_mask, game_depth = rasterize(game, center, scale, direction, resolution)
        union = source_mask | game_mask
        intersection = source_mask & game_mask
        silhouette_iou = float(intersection.sum() / max(1, union.sum()))
        depth_error = np.zeros_like(source_depth, dtype=np.float32)
        if bool(intersection.any()):
            depth_error[intersection] = np.abs(source_depth[intersection] - game_depth[intersection])
            depth_mae = float(depth_error[intersection].mean())
            depth_p95 = float(np.percentile(depth_error[intersection], 95))
        else:
            depth_mae = 1.0
            depth_p95 = 1.0
        overlay_path = os.path.join(out_dir, f"view_{idx + 1:02d}_overlay.png")
        save_overlay(source_mask, game_mask, depth_error, overlay_path)
        results.append(
            {
                "view": idx + 1,
                "azimuth_degrees": idx * 30,
                "silhouette_iou": silhouette_iou,
                "depth_mae": depth_mae,
                "depth_p95": depth_p95,
                "overlay": os.path.basename(overlay_path),
            }
        )

    summary = {
        "source": source_path,
        "game": game_path,
        "resolution": resolution,
        "views": results,
        "min_silhouette_iou": min(item["silhouette_iou"] for item in results),
        "max_depth_p95": max(item["depth_p95"] for item in results),
        "mean_depth_mae": float(np.mean([item["depth_mae"] for item in results])),
    }
    with open(os.path.join(out_dir, "comparison_report.json"), "w", encoding="ascii") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Output folder containing mesh.glb and mesh_game.glb")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--fail", action="store_true", help="Exit non-zero when the 12-view match is not tight")
    args = parser.parse_args()

    source = os.path.join(args.folder, "mesh.glb")
    game = os.path.join(args.folder, "mesh_game.glb")
    out_dir = os.path.join(args.folder, "comparison_12views")
    summary = compare(source, game, out_dir, args.resolution)
    print(
        "12-view comparison: "
        f"min silhouette IoU={summary['min_silhouette_iou']:.4f}, "
        f"max depth p95={summary['max_depth_p95']:.5f}, "
        f"mean depth MAE={summary['mean_depth_mae']:.5f}"
    )
    if args.fail and (
        summary["min_silhouette_iou"] < 0.985 or summary["max_depth_p95"] > 0.02
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
