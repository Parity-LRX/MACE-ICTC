"""Integration test for the multipole long-range path.

Test A: the reciprocal multipole energy (LatentReciprocalLongRange.forward_multipole
-> MeshLongRangeKernel3D.multipole_energy) is rotation-invariant under a joint
rotation of positions, lattice, dipole (R mu) and quadrupole (R Q R^T), and is
differentiable w.r.t. positions.
"""

from __future__ import annotations

import torch

from mace_ictd.models.long_range import build_long_range_module


def _random_rotation(dtype) -> torch.Tensor:
    g = torch.Generator().manual_seed(11)
    a = torch.randn(3, 3, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(dtype)


def _make_lr(dtype):
    lr = build_long_range_module(
        mode="reciprocal-spectral-v1",
        feature_dim=8,
        reciprocal_backend="mesh_fft",
        boundary="periodic",
        mesh_size=32,
        source_channels=1,
        green_mode="poisson",
        assignment="pcs",  # higher-order (vs cic) for accurate, translation-stable PME
        neutralize=True,
        mesh_fft_full_ewald=True,  # Ewald Gaussian screening -> band-limited reciprocal sum
    ).to(dtype)
    if getattr(lr, "energy_scale", None) is not None:
        with torch.no_grad():
            lr.energy_scale.fill_(1.0)  # init may be 0 -> make the test non-trivial
    return lr


def test_forward_multipole_rotation_invariance_and_forces():
    dtype = torch.float64
    lr = _make_lr(dtype)
    n = 5
    g = torch.Generator().manual_seed(1)
    L = 6.0
    pos = torch.rand(n, 3, generator=g, dtype=dtype) * L
    batch = torch.zeros(n, dtype=torch.long)
    cell = (torch.eye(3, dtype=dtype) * L).unsqueeze(0)  # [1,3,3], rows = lattice vectors
    q = torch.randn(n, 1, generator=g, dtype=dtype)
    mu = torch.randn(n, 1, 3, generator=g, dtype=dtype)
    Q = torch.randn(n, 1, 3, 3, generator=g, dtype=dtype)
    Q = 0.5 * (Q + Q.transpose(-1, -2))

    e1 = lr.forward_multipole(pos, batch, cell, q, mu, Q).sum()

    R = _random_rotation(dtype)
    pos_r = pos @ R.T
    cell_r = (cell[0] @ R.T).unsqueeze(0)
    mu_r = mu @ R.T
    Q_r = torch.einsum("ij,ncjk,lk->ncil", R, Q, R)
    e2 = lr.forward_multipole(pos_r, batch, cell_r, q, mu_r, Q_r).sum()

    assert torch.allclose(e1, e2, atol=1e-8), f"not rotation-invariant: {(e1 - e2).abs().item()}"

    # forces flow and are translation-invariant (uniform shift -> same energy)
    pos2 = pos.clone().requires_grad_(True)
    e = lr.forward_multipole(pos2, batch, cell, q, mu, Q).sum()
    (grad,) = torch.autograd.grad(e, pos2)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0, "no/invalid force"

    # exact invariance under a full lattice-vector translation: confirms periodic
    # wrapping/spreading is correct. (In-cell sub-grid translation is only as accurate
    # as the underlying mesh PME -- CIC + mesh resolution -- which is a property/knob of
    # the long_range module, not of this multipole wiring; rotation-invariance above is
    # the equivariance-correctness check.)
    e_cell = lr.forward_multipole(pos + cell[0, 0], batch, cell, q, mu, Q).sum()
    assert torch.allclose(e1, e_cell, atol=1e-9), (
        f"not invariant under a lattice-vector shift: {(e1 - e_cell).abs().item():.2e}"
    )

    # sub-grid (in-cell) translation accuracy: with Ewald screening (full_ewald) +
    # higher-order assignment (pcs) this is now ~1e-3 (was tens of % with bare
    # poisson + CIC -- the #2 fix).
    e_sub = lr.forward_multipole(pos + torch.tensor([0.05, 0.05, 0.05], dtype=dtype), batch, cell, q, mu, Q).sum()
    rel_sub = (e1 - e_sub).abs() / e1.abs().clamp_min(1e-12)
    assert rel_sub < 5e-3, f"sub-grid translation error too large: {rel_sub.item():.2e}"


def _build_model(*, max_multipole_l: int, dispersion: bool = False):
    from mace_ictd.models.pure_cartesian_ictd_fix import PureCartesianICTDFix

    torch.set_default_dtype(torch.float64)
    c = 8
    return PureCartesianICTDFix(
        max_embed_radius=4.5,
        main_max_radius=4.5,
        main_number_of_basis=8,
        hidden_dim_conv=c,
        hidden_dim_sh=c,
        hidden_dim=c,
        channel_in2=c,
        embedding_dim=c,
        max_atomvalue=10,
        atomic_numbers=[1, 6, 7, 8],
        num_interaction=2,
        function_type_main="bessel",
        lmax=2,
        ictd_fix_edge_lmax=2,
        ictd_fix_route="baseline",
        ictd_fix_product_backend="ictd-bridge-u",
        ictd_fix_use_reduced_cg=False,
        save_contraction_order=3,
        avg_num_neighbors=8.0,
        angular_basis="ictd",
        internal_compute_dtype=torch.float64,
        device="cpu",
        long_range_mode="reciprocal-spectral-v1",
        long_range_reciprocal_backend="mesh_fft",
        long_range_boundary="periodic",
        long_range_mesh_size=16,
        long_range_assignment="pcs",
        long_range_mesh_fft_full_ewald=True,
        long_range_max_multipole_l=int(max_multipole_l),
        long_range_dispersion=bool(dispersion),
    )


def test_model_multipole_gating():
    # OFF (default lmax 0): no multipole readout -> the scalar latent-source path is used,
    # so the long-range-off / bridge-U numerics are byte-identical.
    off = _build_model(max_multipole_l=0)
    assert off.multipole_readout is None
    assert off.long_range_max_multipole_l == 0
    # ON: the equivariant multipole readout is wired in.
    on = _build_model(max_multipole_l=2)
    assert on.multipole_readout is not None
    assert on.long_range_max_multipole_l == 2


def _neighbor_list(pos, cell, r_max):
    """Minimal-image periodic neighbor list. Returns edge_src(j), edge_dst(i), unit_shifts;
    model edge vec = pos[i] - pos[j] + unit_shifts @ cell."""
    import itertools

    n = pos.shape[0]
    src, dst, shifts = [], [], []
    for i in range(n):
        for j in range(n):
            for s in itertools.product([-1, 0, 1], repeat=3):
                s_t = torch.tensor(s, dtype=pos.dtype)
                d = pos[i] - pos[j] + s_t @ cell
                r = float(d.norm())
                if 0.0 < r <= r_max:
                    src.append(j)
                    dst.append(i)
                    shifts.append(list(s))
    return (
        torch.tensor(src, dtype=torch.long),
        torch.tensor(dst, dtype=torch.long),
        torch.tensor(shifts, dtype=pos.dtype),
    )


def test_model_multipole_forward_smoke():
    """End-to-end: a wired model with multipole long-range ON runs forward, gives a finite
    energy + finite forces, and a rotation-invariant total energy."""
    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=2).double().eval()
    torch.manual_seed(0)
    L = 8.0
    cell = torch.eye(3, dtype=torch.float64) * L
    A = torch.tensor([1, 6, 7, 8, 1, 6, 7, 8], dtype=torch.long)
    n = A.numel()
    pos0 = torch.rand(n, 3, dtype=torch.float64) * L
    batch = torch.zeros(n, dtype=torch.long)
    edge_src, edge_dst, unit_shifts = _neighbor_list(pos0, cell, r_max=4.5)
    assert edge_src.numel() > 0, "empty neighbor list"

    pos = pos0.clone().requires_grad_(True)
    e = model(pos, A, batch, edge_src, edge_dst, unit_shifts, cell.reshape(1, 3, 3)).sum()
    assert torch.isfinite(e), "energy not finite"
    (forces,) = torch.autograd.grad(-e, pos)
    assert torch.isfinite(forces).all() and forces.abs().sum() > 0, "bad forces"

    # rotation invariance: rotate positions + lattice consistently (integer shifts unchanged)
    R = _random_rotation(torch.float64)
    pos_r = (pos0 @ R.T).requires_grad_(True)
    e_r = model(pos_r, A, batch, edge_src, edge_dst, unit_shifts, (cell @ R.T).reshape(1, 3, 3)).sum()
    assert torch.allclose(e.detach(), e_r.detach(), atol=1e-6), (
        f"total energy not rotation-invariant: {(e - e_r).abs().item():.2e}"
    )


def test_model_multipole_training_smoke():
    """A few optimizer steps with multipole long-range ON: the loss decreases and the
    multipole readout receives gradients (the long-range path is trainable end-to-end)."""
    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=2).double().train()
    # the long-range energy_scale inits to 0 (no-op); activate it so the multipole path
    # is exercised from step 1.
    for m in model.modules():
        if getattr(m, "energy_scale", None) is not None:
            with torch.no_grad():
                m.energy_scale.fill_(0.1)

    torch.manual_seed(0)
    L = 8.0
    cell = (torch.eye(3, dtype=torch.float64) * L).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7, 8, 1, 6, 7, 8], dtype=torch.long)
    n = A.numel()
    pos0 = torch.rand(n, 3, dtype=torch.float64) * L
    batch = torch.zeros(n, dtype=torch.long)
    edge_src, edge_dst, shifts = _neighbor_list(pos0, cell[0], r_max=4.5)
    target_f = torch.randn(n, 3, dtype=torch.float64) * 0.1

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        pos = pos0.clone().requires_grad_(True)
        e = model(pos, A, batch, edge_src, edge_dst, shifts, cell).sum()
        (force,) = torch.autograd.grad(-e, pos, create_graph=True)
        loss = ((force - target_f) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.3e} -> {losses[-1]:.3e}"
    mp_grads = [p.grad for p in model.multipole_readout.parameters() if p.grad is not None]
    assert mp_grads and any(g.abs().sum() > 0 for g in mp_grads), "multipole readout received no gradient"
    return losses[0], losses[-1]


def test_export_reciprocal_source_equivariant_layout():
    """Export mode (return_reciprocal_source=True): the model emits a packed [q|mu|Q] source of
    width S*(1+3+9)=13S matching mff_reciprocal_solver's narrow/reshape decode, and the source
    transforms equivariantly (q invariant, mu->R mu, Q->R Q R^T) so the C++ reciprocal energy is
    rotation-invariant. Validates the Python-export <-> C++-solver contract."""
    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=2).double().eval()
    assert model.long_range_exports_reciprocal_source, "multipole model must export reciprocal_source"
    box = 8.0
    cell = (torch.eye(3, dtype=torch.float64) * box).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7, 8, 1, 6])
    n = A.numel()
    s = model.multipole_readout.source_channels
    torch.manual_seed(3)
    pos = torch.rand(n, 3, dtype=torch.float64) * box
    batch = torch.zeros(n, dtype=torch.long)
    es, ed, sh = _neighbor_list(pos, cell[0], r_max=4.5)

    def emit(p, c):
        _out, rs = model(p, A, batch, es, ed, sh, c, return_reciprocal_source=True)
        q = rs[:, :s]
        mu = rs[:, s:4 * s].reshape(n, s, 3)            # C++ decode: narrow(C,3C).reshape(C,3)
        quad = rs[:, 4 * s:13 * s].reshape(n, s, 3, 3)  # narrow(4C,9C).reshape(C,3,3)
        return rs, q, mu, quad

    rs, q, mu, quad = emit(pos, cell)
    assert rs.shape == (n, s * 13), rs.shape
    assert torch.isfinite(rs).all()

    R = _random_rotation(torch.float64)
    _, q_r, mu_r, quad_r = emit(pos @ R.T, (cell[0] @ R.T).reshape(1, 3, 3))
    assert torch.allclose(q, q_r, atol=1e-8), "monopole source not invariant"
    assert torch.allclose(mu @ R.T, mu_r, atol=1e-8), (mu @ R.T - mu_r).abs().max()
    assert torch.allclose(
        torch.einsum("ij,nsjk,lk->nsil", R, quad, R), quad_r, atol=1e-8
    ), "quadrupole source not equivariant"


if __name__ == "__main__":
    test_forward_multipole_rotation_invariance_and_forces()
    print("OK: forward_multipole rotation-invariance + forces + translation-invariance")
    test_model_multipole_gating()
    print("OK: model multipole gating (off -> None / on -> readout)")
    test_model_multipole_forward_smoke()
    print("OK: full-model forward smoke (energy + forces + rotation-invariance, multipole ON)")
    l0, l1 = test_model_multipole_training_smoke()
    print(f"OK: training smoke (loss {l0:.3e} -> {l1:.3e}, multipole readout gets gradient)")
    test_export_reciprocal_source_equivariant_layout()
    print("OK: export reciprocal_source packed [q|mu|Q] layout + equivariance (C++ contract)")
