#!/usr/bin/env python3
"""Regression test for the default-device bug: the pipeline calls convs under
`with torch.device("mps")`, which changes the default device for unpinned
tensor creation. Vec pair builders must pin every created tensor to the CPU
coords' device. Runs all three vec forwards inside that context on MPS feats
and compares against orig (outside the context). fp32, tol 1e-3.
"""
import os
import sys

os.environ.setdefault("SPARSE_BACKEND", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import torch

import sam3d_objects.model.backbone.tdfy_dit.modules.sparse.conv.conv_native as cn

torch.manual_seed(0)
DEV = torch.device("mps")

# synthetic coords: two small clusters
pts = []
for b in range(1):
    for z in range(4):
        for y in range(5):
            for x in range(6):
                if (x + y + z) % 3:
                    pts.append([b, z, y, x])
coords = torch.tensor(pts, dtype=torch.int64)
N = coords.shape[0]
Cin, Cout = 8, 12

feats = torch.randn(N, Cin)
weight = torch.randn(3, 3, 3, Cin, Cout) * 0.2
bias = torch.randn(Cout)

# orig reference (CPU, no device context)
ref = cn._subm_forward_orig(feats, coords, weight, bias)

# vec inside an MPS default-device context, feats on MPS (pipeline conditions;
# weight/bias on the feats device, as the module forward guarantees)
with torch.device("mps"):
    out = cn._subm_forward_vec(feats.to(DEV), coords, weight.to(DEV), bias.to(DEV))
d1 = (out.cpu() - ref).abs().max().item()
print(f"subm vec under default-device=mps: max diff {d1:.3e}")
assert d1 < 1e-3

# strided vec under the context (returns feats, coords, spatial)
ref_s, out_idx_ref, _ = cn._sparse_forward_orig(feats, coords, weight, bias, (2, 2, 2), 0, None, 1)
with torch.device("mps"):
    out_s, out_idx, _ = cn._sparse_forward_vec(feats.to(DEV), coords, weight.to(DEV), bias.to(DEV), (2, 2, 2), 0, None, 1)
same_coords = set(map(tuple, out_idx.cpu().tolist())) == set(map(tuple, out_idx_ref.tolist()))
# align rows by coord before comparing feats
def sort_by_coord(idx, f):
    k = idx[:, 0] * 1000000 + idx[:, 1] * 10000 + idx[:, 2] * 100 + idx[:, 3]
    o = torch.argsort(k)
    return f[o]
d2 = (sort_by_coord(out_idx.cpu(), out_s.cpu()) - sort_by_coord(out_idx_ref, ref_s)).abs().max().item()
print(f"strided vec under default-device=mps: coords equal {same_coords}, max diff {d2:.3e}")
assert same_coords and d2 < 1e-3

# inverse vec under the context (weight layout is [Kz,Ky,Kx,Cout,Cin])
weight_t = torch.randn(3, 3, 3, Cout, Cin) * 0.2
ref_i, fine_ref, _ = cn._inverse_scatter_orig(feats, coords, weight_t, bias, (2, 2, 2))
with torch.device("mps"):
    out_i, fine_idx, _ = cn._inverse_scatter_vec(feats.to(DEV), coords, weight_t.to(DEV), bias.to(DEV), (2, 2, 2))
same_fine = set(map(tuple, fine_idx.cpu().tolist())) == set(map(tuple, fine_ref.tolist()))
d3 = (sort_by_coord(fine_idx.cpu(), out_i.cpu()) - sort_by_coord(fine_ref, ref_i)).abs().max().item()
print(f"inverse vec under default-device=mps: coords equal {same_fine}, max diff {d3:.3e}")
assert same_fine and d3 < 1e-3

print("DEFAULT-DEVICE REGRESSION: PASS")
