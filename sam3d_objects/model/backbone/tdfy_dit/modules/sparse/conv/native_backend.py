# Copyright (c) Meta Platforms, Inc. and affiliates.
# Pure-PyTorch sparse tensor container (no CUDA / spconv required).
import torch


class NativeSparseData:
    """
    Drop-in replacement for spconv.SparseConvTensor for the "native" backend.

    Coordinate layout: [N, 4] int32  — (batch, z, y, x), matching spconv's
    convention so that the weight-indexing code in conv_native.py stays aligned
    with the stored checkpoint weights.
    """

    def __init__(self, features, indices, spatial_shape, batch_size):
        # features : [N, C]  float
        # indices  : [N, 4]  int32 — (batch, z, y, x)
        self.features = features
        self._features = features        # spconv compatibility alias
        self.indices = indices
        self.spatial_shape = list(spatial_shape)
        self.batch_size = int(batch_size)
        self.indice_dict: dict = {}      # inverse-conv mapping cache

    def replace_feature(self, new_features: torch.Tensor) -> "NativeSparseData":
        d = NativeSparseData(new_features, self.indices, self.spatial_shape, self.batch_size)
        d.indice_dict = self.indice_dict
        return d

    def dense(self) -> torch.Tensor:
        B = self.batch_size
        C = self.features.shape[1]
        D, H, W = self.spatial_shape
        out = torch.zeros(B, C, D, H, W,
                          device=self.features.device,
                          dtype=self.features.dtype)
        idx = self.indices.long()
        b, z, y, x = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        out[b, :, z, y, x] = self.features
        return out
