"""Physics test for the learned pairwise C6 dispersion term."""

from __future__ import annotations

import torch

from mace_ictd.models.dispersion import PairwiseDispersion
from mace_ictd.test.test_multipole_long_range import _build_model, _neighbor_list, _random_rotation


def test_dispersion_physics():
    dtype = torch.float64
    torch.manual_seed(0)
    c = 8
    disp = PairwiseDispersion(feature_dim=c).to(dtype)
    feats = torch.randn(2, c, dtype=dtype)
    src = torch.tensor([0, 1])
    dst = torch.tensor([1, 0])

    def energy(r):
        lengths = torch.tensor([r, r], dtype=dtype)
        return disp(feats, src, dst, lengths).sum()

    e_close, e_mid, e_far = energy(2.0), energy(4.0), energy(8.0)
    # attractive and monotonically decaying toward 0 with separation
    assert e_close < 0 and e_mid < 0, "dispersion must be attractive (negative)"
    assert e_close < e_mid < e_far <= 1e-12, "dispersion must decay toward 0 with distance"
    # Becke-Johnson damping keeps it finite as r -> 0 (short-range network owns contact)
    assert torch.isfinite(energy(0.05)), "dispersion not finite at small r"

    # forces flow and are attractive (atoms pulled together)
    pos = torch.tensor([[0.0, 0, 0], [3.0, 0, 0]], dtype=dtype, requires_grad=True)
    lengths = (pos[dst] - pos[src]).norm(dim=-1)
    e = disp(feats, src, dst, lengths).sum()
    (grad,) = torch.autograd.grad(e, pos)
    force = -grad
    assert torch.isfinite(force).all() and force.abs().sum() > 0, "bad force"
    assert force[0, 0] > 0 and force[1, 0] < 0, "dispersion force is not attractive"

    # rotation/translation invariance is by construction (depends only on |r_ij|): spot-check
    R = torch.linalg.qr(torch.randn(3, 3, dtype=dtype))[0]
    pos_r = pos.detach() @ R.T + torch.tensor([1.3, -0.7, 0.2], dtype=dtype)
    lengths_r = (pos_r[dst] - pos_r[src]).norm(dim=-1)
    e_r = disp(feats, src, dst, lengths_r).sum()
    assert torch.allclose(e.detach(), e_r, atol=1e-10), "not rotation/translation invariant"


def test_model_complete_long_range_smoke():
    """Model with BOTH multipole electrostatics AND C6 dispersion (the complete long-range):
    runs, finite energy + forces, rotation-invariant total energy, and both new heads train."""
    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=2, dispersion=True).double().train()
    for m in model.modules():
        if getattr(m, "energy_scale", None) is not None:
            with torch.no_grad():
                m.energy_scale.fill_(0.1)  # activate the reciprocal term (inits to 0)

    torch.manual_seed(1)
    L = 8.0
    cell = (torch.eye(3, dtype=torch.float64) * L).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7, 8, 1, 6, 7, 8])
    n = A.numel()
    pos0 = torch.rand(n, 3, dtype=torch.float64) * L
    batch = torch.zeros(n, dtype=torch.long)
    es, ed, sh = _neighbor_list(pos0, cell[0], r_max=4.5)

    pos = pos0.clone().requires_grad_(True)
    e = model(pos, A, batch, es, ed, sh, cell).sum()
    assert torch.isfinite(e), "energy not finite"
    (force,) = torch.autograd.grad(e, pos, create_graph=True)
    assert torch.isfinite(force).all() and force.abs().sum() > 0, "bad force"

    R = _random_rotation(torch.float64)
    e_r = model(pos0 @ R.T, A, batch, es, ed, sh, (cell[0] @ R.T).reshape(1, 3, 3)).sum()
    assert torch.allclose(e.detach(), e_r.detach(), atol=1e-6), (
        f"complete long-range not rotation-invariant: {(e - e_r).abs().item():.2e}"
    )

    (force ** 2).mean().backward()
    mp = [p.grad for p in model.multipole_readout.parameters() if p.grad is not None]
    dp = [p.grad for p in model.dispersion.parameters() if p.grad is not None]
    assert mp and any(g.abs().sum() > 0 for g in mp), "multipole readout got no gradient"
    assert dp and any(g.abs().sum() > 0 for g in dp), "dispersion got no gradient"


def test_dispersion_neighbor_list_matches_bruteforce():
    """The longer-cutoff dispersion neighbor list matches a brute-force periodic search
    (cutoff < box so a single image shell is complete), removing the short-range truncation."""
    from mace_ictd.models.dispersion import dispersion_neighbor_list
    from mace_ictd.test.test_multipole_long_range import _neighbor_list

    dtype = torch.float64
    torch.manual_seed(2)
    box, cutoff, n = 12.0, 5.0, 6
    cell = torch.eye(3, dtype=dtype) * box
    pos = torch.rand(n, 3, dtype=dtype) * box
    batch = torch.zeros(n, dtype=torch.long)

    src, dst, shifts = dispersion_neighbor_list(pos, batch, cell.reshape(1, 3, 3), cutoff, pbc=True)
    bsrc, bdst, bsh = _neighbor_list(pos, cell, cutoff)

    def keyset(s, d, sh):
        return {(int(a), int(b), tuple(int(x) for x in c)) for a, b, c in zip(s, d, sh)}

    assert keyset(src, dst, shifts) == keyset(bsrc, bdst, bsh), "dispersion list != brute force"
    dlen = (pos[dst] - pos[src] + shifts.to(dtype) @ cell).norm(dim=1)
    assert (dlen > 1e-8).all() and (dlen <= cutoff + 1e-9).all(), "pairs outside cutoff"


if __name__ == "__main__":
    test_dispersion_physics()
    print("OK: dispersion physics (attractive, decaying, BJ-finite, attractive forces, invariant)")
    test_dispersion_neighbor_list_matches_bruteforce()
    print("OK: dispersion neighbor list matches brute-force periodic search")
    test_model_complete_long_range_smoke()
    print("OK: complete long-range smoke (multipole electrostatics + dispersion, both train)")
