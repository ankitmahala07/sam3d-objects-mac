#!/usr/bin/env python3
"""A/B verification: vectorized sparse conv ("vec") vs original ("orig").

Run with the s3d_env python, from the repo root:
    SAM3D_CONV_IMPL=vec ~/alt3d/s3d_env/bin/python verify_conv_vec.py

Exit code 0 iff every comparison passes the tolerance gate.
"""
import os, sys, time, json

os.environ.setdefault("SPARSE_BACKEND", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import torch

torch.manual_seed(1234)

import sam3d_objects.model.backbone.tdfy_dit.modules.sparse.conv.conv_native as cn

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
TOL = 1e-3
results = []


def report(name, a, b, extra=""):
    diff = (a - b).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    scale = float(b.abs().max()) if b.numel() else 0.0
    rel = max_abs / max(scale, 1e-9)
    ok = max_abs <= TOL
    results.append((name, max_abs, rel, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max_abs_diff={max_abs:.3e} rel={rel:.3e} {extra}")
    return ok


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def rand_coords(n, grid, batches=1, seed=7):
    g = torch.Generator().manual_seed(seed)
    total = grid ** 3
    lin = torch.randperm(total, generator=g)[:n]
    z = lin // (grid * grid)
    y = (lin // grid) % grid
    x = lin % grid
    b = torch.randint(0, batches, (n,), generator=g)
    return torch.stack([b, z, y, x], dim=1).long()


# ---------------------------------------------------------------- subm ------
def test_subm(dev):
    for K, cin, cout, n, grid, batches in [(3, 5, 7, 300, 10, 2), (1, 6, 6, 200, 8, 1), (3, 16, 16, 2000, 24, 1)]:
        coords = rand_coords(n, grid, batches)
        feats = torch.randn(n, cin)
        w = torch.randn(K, K, K, cin, cout) * 0.1
        bias = torch.randn(cout)
        fd, cd, wd, bd = feats.to(dev), coords.to(dev), w.to(dev), bias.to(dev)
        a = cn._subm_forward_orig(fd, cd, wd, bd)
        b = cn._subm_forward_vec(fd, cd, wd, bd)
        report(f"subm K={K} cin={cin} N={n} b={batches} [{dev}]", a.cpu(), b.cpu())


def test_subm_real(dev):
    d = torch.load("outputs/craftsman/slat.pt", map_location="cpu")
    coords = d["coords"].long()
    n = coords.shape[0]
    for cin, cout in [(128, 128), (8, 128), (2048, 128)]:
        feats = torch.randn(n, cin)
        w = torch.randn(3, 3, 3, cin, cout) * (1.0 / (27 * cin) ** 0.5)
        bias = torch.randn(cout) * 0.1
        fd, cd, wd, bd = feats.to(dev), coords.to(dev), w.to(dev), bias.to(dev)
        t0 = time.perf_counter(); a = cn._subm_forward_orig(fd, cd, wd, bd); sync(dev); t_orig = time.perf_counter() - t0
        t0 = time.perf_counter(); b = cn._subm_forward_vec(fd, cd, wd, bd); sync(dev); t_vec1 = time.perf_counter() - t0
        t0 = time.perf_counter(); b2 = cn._subm_forward_vec(fd, cd, wd, bd); sync(dev); t_vec2 = time.perf_counter() - t0
        report(f"subm REAL slat coords {cin}->{cout} N={n} [{dev}]", a.cpu(), b.cpu(),
               extra=f"| orig {t_orig:.2f}s vec(cold) {t_vec1:.3f}s vec(cached) {t_vec2:.3f}s")
        assert torch.equal(b.cpu(), b2.cpu()), "vec not deterministic across cached calls"


# -------------------------------------------------------------- strided -----
def test_strided(dev):
    cases = [
        ((2, 2, 2), 1, [16, 16, 16]),
        ((1, 1, 1), 1, [16, 16, 16]),
        ((2, 2, 2), 0, [16, 16, 16]),
        ((1, 1, 1), 0, [12, 12, 12]),
    ]
    for stride, pad, shape in cases:
        coords = rand_coords(800, 16, 1)
        n = coords.shape[0]
        feats = torch.randn(n, 6)
        w = torch.randn(3, 3, 3, 6, 10) * 0.1
        bias = torch.randn(10)
        fd, cd, wd, bd = feats.to(dev), coords.to(dev), w.to(dev), bias.to(dev)
        fa, ia, sa = cn._sparse_forward_orig(fd, cd, wd, bd, stride, pad, shape, 1)
        fb, ib, sb = cn._sparse_forward_vec(fd, cd, wd, bd, stride, pad, shape, 1)
        same_coords = torch.equal(ia.cpu(), ib.cpu())
        print(f"  strided stride={stride} pad={pad}: coords identical: {same_coords}, spatial {sa} vs {sb}")
        assert same_coords and sa == sb, "strided coord/spatial mismatch"
        report(f"strided stride={stride} pad={pad} [{dev}]", fa.cpu(), fb.cpu())


# -------------------------------------------------------------- inverse -----
def test_inverse(dev):
    for stride in [(2, 2, 2), (1, 1, 1)]:
        coords = rand_coords(500, 10, 1)
        n = coords.shape[0]
        cin, cout = 6, 9
        feats = torch.randn(n, cin)
        w = torch.randn(3, 3, 3, cout, cin) * 0.1  # [K,K,K,Cout,Cin]
        bias = torch.randn(cout)
        fd, cd, wd, bd = feats.to(dev), coords.to(dev), w.to(dev), bias.to(dev)
        fa, ia, sa = cn._inverse_scatter_orig(fd, cd, wd, bd, stride)
        fb, ib, sb = cn._inverse_scatter_vec(fd, cd, wd, bd, stride)
        same_coords = torch.equal(ia.cpu(), ib.cpu())
        print(f"  inverse stride={stride}: fine coords identical: {same_coords}, spatial {sa} vs {sb}")
        assert same_coords and sa == sb, "inverse coord/spatial mismatch"
        report(f"inverse stride={stride} [{dev}]", fa.cpu(), fb.cpu())


# --------------------------------------------------------------- module -----
def test_module(dev):
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    d = torch.load("outputs/craftsman/slat.pt", map_location="cpu")
    coords32 = d["coords"]  # int32 [N,4]
    n = coords32.shape[0]

    def make_x(cin):
        feats = torch.randn(n, cin).to(dev)
        coords = coords32.to(dev)
        return sp.SparseTensor(feats, coords)

    # SubM module (the exercised path), env-flip A/B
    mod = cn.SparseConv3d(8, 32, 3).to(dev)
    with torch.no_grad():
        os.environ["SAM3D_CONV_IMPL"] = "orig"
        a = mod(make_x(8))
        os.environ["SAM3D_CONV_IMPL"] = "vec"
        b = mod(make_x(8))
    # NOTE: make_x draws new random feats each call -> rebuild with same feats
    feats = torch.randn(n, 8)
    def make_x_fixed():
        return sp.SparseTensor(feats.to(dev), coords32.to(dev))
    with torch.no_grad():
        os.environ["SAM3D_CONV_IMPL"] = "orig"
        a = mod(make_x_fixed())
        os.environ["SAM3D_CONV_IMPL"] = "vec"
        b = mod(make_x_fixed())
    assert torch.equal(a.coords.cpu(), b.coords.cpu())
    report(f"module SparseConv3d subm 8->32 real coords [{dev}]", a.feats.cpu(), b.feats.cpu())

    # Strided forward + inverse pair (module level)
    fwd = cn.SparseConv3d(8, 16, 3, stride=2, padding=1).to(dev)
    inv = cn.SparseInverseConv3d(16, 8, 3, stride=2).to(dev)
    with torch.no_grad():
        os.environ["SAM3D_CONV_IMPL"] = "orig"
        xa = fwd(make_x_fixed())
        ya = inv(xa)
        os.environ["SAM3D_CONV_IMPL"] = "vec"
        xb = fwd(make_x_fixed())
        yb = inv(xb)
    assert torch.equal(xa.coords.cpu(), xb.coords.cpu()), "module strided coords differ"
    report("module strided fwd 8->16 s2", xa.feats.cpu(), xb.feats.cpu())
    assert torch.equal(ya.coords.cpu(), yb.coords.cpu()), "module inverse coords differ"
    report("module inverse pair 16->8 s2", ya.feats.cpu(), yb.feats.cpu())
    os.environ["SAM3D_CONV_IMPL"] = "vec"


def main():
    for dev in DEVICES:
        print(f"===== device: {dev} =====")
        test_subm(dev)
        test_strided(dev)
        test_inverse(dev)
    # real-coords + module tests on MPS if available, else CPU
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    test_subm_real(dev)
    test_module(dev)
    bad = [r for r in results if not r[3]]
    print(f"\n==== {len(results) - len(bad)}/{len(results)} passed, tol={TOL} ====")
    if bad:
        print("FAILURES:", [r[0] for r in bad])
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
