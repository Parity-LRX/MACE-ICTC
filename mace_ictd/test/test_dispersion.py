"""Physics test for the learned pairwise C6 dispersion term."""

from __future__ import annotations

import torch

from mace_ictd.models.dispersion import (
    LongRangeDispersion,
    ManyBodyDispersion,
    ManyBodyDispersionSLQ,
    PairwiseDispersion,
)
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


def test_long_range_dispersion_wrapper_matches_pairwise_edge_mode():
    dtype = torch.float64
    torch.manual_seed(4)
    c = 8
    pair = PairwiseDispersion(feature_dim=c).to(dtype)
    wrapped = LongRangeDispersion(feature_dim=c, mode="pairwise-c6", cutoff=0.0, pbc=False).to(dtype)
    wrapped.term.load_state_dict(pair.state_dict())

    feats = torch.randn(4, c, dtype=dtype)
    pos = torch.randn(4, 3, dtype=dtype)
    batch = torch.zeros(4, dtype=torch.long)
    cell = torch.eye(3, dtype=dtype).reshape(1, 3, 3) * 20.0
    src = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    dst = torch.tensor([1, 0, 3, 2, 2, 0], dtype=torch.long)
    lengths = (pos[dst] - pos[src]).norm(dim=-1)

    e_pair = pair(feats, src, dst, lengths)
    e_wrapped = wrapped(
        feats,
        pos,
        batch,
        cell,
        edge_src=src,
        edge_dst=dst,
        edge_lengths=lengths,
        cutoff=0.0,
        pbc=False,
    )
    assert torch.allclose(e_pair, e_wrapped, atol=0.0, rtol=0.0), "wrapper changed pairwise C6 numerics"


def test_many_body_dispersion_is_finite_invariant_and_nonadditive():
    dtype = torch.float64
    c = 4
    mbd = ManyBodyDispersion(feature_dim=c).to(dtype)

    def inv_softplus(x):
        return torch.log(torch.expm1(torch.as_tensor(x, dtype=dtype)))

    with torch.no_grad():
        for p in mbd.parameters():
            p.zero_()
        mbd.alpha_head[-1].bias.fill_(inv_softplus(1.2))
        mbd.omega_head[-1].bias.fill_(inv_softplus(0.7))
        mbd.coupling_scale.fill_(0.2)
        mbd.beta_raw.fill_(inv_softplus(1.1))

    def complete_directed_edges(pos):
        n = pos.shape[0]
        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(j)
                    dst.append(i)
        src = torch.tensor(src, dtype=torch.long)
        dst = torch.tensor(dst, dtype=torch.long)
        return src, dst, pos[dst] - pos[src]

    pos = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.7, 2.6, 0.0]], dtype=dtype, requires_grad=True)
    feats = torch.zeros(3, c, dtype=dtype)
    batch = torch.zeros(3, dtype=torch.long)
    src, dst, edge_vec = complete_directed_edges(pos)
    e3 = mbd(feats, batch, src, dst, edge_vec).sum()
    assert torch.isfinite(e3), "MBD energy is not finite"
    (grad,) = torch.autograd.grad(e3, pos, create_graph=True)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0, "MBD force gradient did not flow"

    R = _random_rotation(dtype)
    pos_r = (pos.detach() @ R.T).requires_grad_(True)
    src_r, dst_r, edge_vec_r = complete_directed_edges(pos_r)
    e3_r = mbd(feats, batch, src_r, dst_r, edge_vec_r).sum()
    assert torch.allclose(e3.detach(), e3_r.detach(), atol=1e-10), "MBD energy is not rotation invariant"

    pair_sum = pos.new_tensor(0.0)
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        pp = pos.detach()[torch.tensor([a, b])].clone().requires_grad_(True)
        pf = feats[:2]
        pb = torch.zeros(2, dtype=torch.long)
        ps, pd, pev = complete_directed_edges(pp)
        pair_sum = pair_sum + mbd(pf, pb, ps, pd, pev).sum()
    assert (e3.detach() - pair_sum.detach()).abs() > 1e-10, "MBD collapsed to pairwise-additive energy"


def test_many_body_dispersion_slq_basis_matches_dense_oracle():
    dtype = torch.float64
    c = 4
    torch.manual_seed(17)
    dense = ManyBodyDispersion(feature_dim=c).to(dtype)
    slq = ManyBodyDispersionSLQ(feature_dim=c, probe_mode="basis", lanczos_steps=32).to(dtype)
    slq.load_state_dict(dense.state_dict(), strict=False)

    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [2.7, 0.2, 0.0], [0.4, 2.4, 0.3], [1.5, 1.2, 2.1]],
        dtype=dtype,
        requires_grad=True,
    )
    feats = torch.randn(4, c, dtype=dtype)
    batch = torch.zeros(4, dtype=torch.long)
    src, dst = [], []
    for i in range(pos.shape[0]):
        for j in range(pos.shape[0]):
            if i != j:
                src.append(j)
                dst.append(i)
    src = torch.tensor(src, dtype=torch.long)
    dst = torch.tensor(dst, dtype=torch.long)
    edge_vec = pos[dst] - pos[src]

    e_dense = dense(feats, batch, src, dst, edge_vec).sum()
    e_slq = slq(feats, batch, src, dst, edge_vec).sum()
    assert torch.allclose(e_dense, e_slq, atol=2e-8, rtol=2e-8), (
        f"basis SLQ does not match dense MBD: dense={e_dense.item():.8e}, slq={e_slq.item():.8e}"
    )

    (g_dense,) = torch.autograd.grad(e_dense, pos, retain_graph=True)
    (g_slq,) = torch.autograd.grad(e_slq, pos)
    assert torch.allclose(g_dense, g_slq, atol=2e-7, rtol=2e-7), "basis SLQ force gradient != dense MBD"


def test_model_accepts_explicit_dispersion_neighbor_list():
    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=0, dispersion=True).double().eval()
    model.dispersion_cutoff = 5.0
    if model.dispersion is not None:
        model.dispersion.cutoff = 5.0

    torch.manual_seed(5)
    L = 12.0
    cell = (torch.eye(3, dtype=torch.float64) * L).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7, 8, 1, 6], dtype=torch.long)
    pos = torch.rand(A.numel(), 3, dtype=torch.float64) * L
    batch = torch.zeros(A.numel(), dtype=torch.long)
    main_src, main_dst, main_shift = _neighbor_list(pos, cell[0], r_max=3.0)
    disp_src, disp_dst, disp_shift = _neighbor_list(pos, cell[0], r_max=5.0)

    with torch.no_grad():
        e_internal = model(pos, A, batch, main_src, main_dst, main_shift, cell)
        e_explicit = model(
            pos,
            A,
            batch,
            main_src,
            main_dst,
            main_shift,
            cell,
            dispersion_edge_src=disp_src,
            dispersion_edge_dst=disp_dst,
            dispersion_edge_shifts=disp_shift,
        )
    assert torch.allclose(e_internal, e_explicit, atol=1e-10, rtol=1e-10), (
        "explicit dispersion neighbor list changed the cutoff-based dispersion result"
    )


def test_model_mbd_dispersion_smoke():
    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=0, dispersion=True).double().train()
    model.long_range_dispersion_mode = "mbd"
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd",
        cutoff=0.0,
        pbc=True,
    ).to(dtype=torch.float64)

    torch.manual_seed(6)
    L = 10.0
    cell = (torch.eye(3, dtype=torch.float64) * L).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7, 8, 1], dtype=torch.long)
    pos = (torch.rand(A.numel(), 3, dtype=torch.float64) * L).requires_grad_(True)
    batch = torch.zeros(A.numel(), dtype=torch.long)
    main_src, main_dst, main_shift = _neighbor_list(pos.detach(), cell[0], r_max=4.0)
    disp_src, disp_dst, disp_shift = _neighbor_list(pos.detach(), cell[0], r_max=6.0)

    e = model(
        pos,
        A,
        batch,
        main_src,
        main_dst,
        main_shift,
        cell,
        dispersion_edge_src=disp_src,
        dispersion_edge_dst=disp_dst,
        dispersion_edge_shifts=disp_shift,
    ).sum()
    assert torch.isfinite(e), "model+MBD energy is not finite"
    (force,) = torch.autograd.grad(e, pos, create_graph=True)
    assert torch.isfinite(force).all() and force.abs().sum() > 0, "model+MBD force gradient did not flow"
    force.pow(2).mean().backward()
    grads = [p.grad for p in model.dispersion.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), "MBD dispersion parameters got no gradient"


def test_model_mbd_slq_dispersion_smoke():
    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=0, dispersion=True).double().train()
    model.long_range_dispersion_mode = "mbd-slq"
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd-slq",
        cutoff=0.0,
        pbc=True,
    ).to(dtype=torch.float64)
    model.dispersion.term.num_probes = 4
    model.dispersion.term.lanczos_steps = 8

    torch.manual_seed(18)
    L = 10.0
    cell = (torch.eye(3, dtype=torch.float64) * L).reshape(1, 3, 3)
    A = torch.tensor([1, 6, 7, 8, 1, 6], dtype=torch.long)
    pos = (torch.rand(A.numel(), 3, dtype=torch.float64) * L).requires_grad_(True)
    batch = torch.zeros(A.numel(), dtype=torch.long)
    main_src, main_dst, main_shift = _neighbor_list(pos.detach(), cell[0], r_max=4.0)
    disp_src, disp_dst, disp_shift = _neighbor_list(pos.detach(), cell[0], r_max=6.0)

    e = model(
        pos,
        A,
        batch,
        main_src,
        main_dst,
        main_shift,
        cell,
        dispersion_edge_src=disp_src,
        dispersion_edge_dst=disp_dst,
        dispersion_edge_shifts=disp_shift,
    ).sum()
    assert torch.isfinite(e), "model+SLQ-MBD energy is not finite"
    (force,) = torch.autograd.grad(e, pos, create_graph=True)
    assert torch.isfinite(force).all() and force.abs().sum() > 0, "model+SLQ-MBD force gradient did not flow"
    force.pow(2).mean().backward()
    grads = [p.grad for p in model.dispersion.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), "SLQ-MBD dispersion parameters got no gradient"


def test_model_mbd_slq_dispersion_batched_variable_n_smoke():
    torch.set_default_dtype(torch.float64)
    model = _build_model(max_multipole_l=0, dispersion=True).double().train()
    model.long_range_dispersion_mode = "mbd-slq"
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd-slq",
        cutoff=0.0,
        pbc=True,
    ).to(dtype=torch.float64)
    model.dispersion.term.num_probes = 3
    model.dispersion.term.lanczos_steps = 6

    torch.manual_seed(19)
    L = 10.0
    elems = torch.tensor([1, 6, 7, 8], dtype=torch.long)
    pos_parts, atom_parts, batch_parts = [], [], []
    main_src_parts, main_dst_parts, main_shift_parts = [], [], []
    disp_src_parts, disp_dst_parts, disp_shift_parts = [], [], []
    cells = []
    offset = 0
    for graph_idx, n_atoms in enumerate((4, 7)):
        cell = torch.eye(3, dtype=torch.float64) * L
        pos_g = torch.rand(n_atoms, 3, dtype=torch.float64) * L
        A_g = elems[torch.arange(n_atoms) % elems.numel()]
        main_src, main_dst, main_shift = _neighbor_list(pos_g, cell, r_max=4.0)
        disp_src, disp_dst, disp_shift = _neighbor_list(pos_g, cell, r_max=6.0)

        pos_parts.append(pos_g)
        atom_parts.append(A_g)
        batch_parts.append(torch.full((n_atoms,), graph_idx, dtype=torch.long))
        main_src_parts.append(main_src + offset)
        main_dst_parts.append(main_dst + offset)
        main_shift_parts.append(main_shift)
        disp_src_parts.append(disp_src + offset)
        disp_dst_parts.append(disp_dst + offset)
        disp_shift_parts.append(disp_shift)
        cells.append(cell)
        offset += n_atoms

    pos = torch.cat(pos_parts).requires_grad_(True)
    A = torch.cat(atom_parts)
    batch = torch.cat(batch_parts)
    cell = torch.stack(cells)
    main_src = torch.cat(main_src_parts)
    main_dst = torch.cat(main_dst_parts)
    main_shift = torch.cat(main_shift_parts)
    disp_src = torch.cat(disp_src_parts)
    disp_dst = torch.cat(disp_dst_parts)
    disp_shift = torch.cat(disp_shift_parts)

    e = model(
        pos,
        A,
        batch,
        main_src,
        main_dst,
        main_shift,
        cell,
        dispersion_edge_src=disp_src,
        dispersion_edge_dst=disp_dst,
        dispersion_edge_shifts=disp_shift,
    ).sum()
    assert torch.isfinite(e), "batched variable-N SLQ-MBD energy is not finite"
    (force,) = torch.autograd.grad(e, pos, create_graph=True)
    assert torch.isfinite(force).all() and force.abs().sum() > 0, "batched variable-N SLQ-MBD force did not flow"
    force.pow(2).mean().backward()
    grads = [p.grad for p in model.dispersion.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), "batched variable-N SLQ-MBD got no gradient"


def test_mbd_torchscript_core_accepts_variable_atom_and_edge_counts():
    """The LibTorch deployment core must not bake the traced MBD matrix size."""
    torch.set_default_dtype(torch.float64)
    from mace_ictd.interfaces.lammps_mliap import _TorchScriptEdgeVecCore

    model = _build_model(max_multipole_l=0, dispersion=True).double().eval()
    model.long_range_dispersion_mode = "mbd"
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd",
        cutoff=0.0,
        pbc=True,
    ).to(dtype=torch.float64)

    def make_inputs(n: int):
        torch.manual_seed(n)
        box = 14.0
        cell = (torch.eye(3, dtype=torch.float64) * box).reshape(1, 3, 3)
        pos = torch.rand(n, 3, dtype=torch.float64) * box
        elements = torch.tensor([1, 6, 7, 8], dtype=torch.long)
        A = elements[torch.arange(n) % elements.numel()]
        batch = torch.zeros(n, dtype=torch.long)
        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(j)
                    dst.append(i)
        src = torch.tensor(src, dtype=torch.long)
        dst = torch.tensor(dst, dtype=torch.long)
        shifts = torch.zeros(src.numel(), 3, dtype=torch.float64)
        edge_vec = pos[dst] - pos[src]
        external = torch.empty(0, dtype=torch.float64)
        return (pos, A, batch, src, dst, shifts, cell, edge_vec, src, dst, shifts, edge_vec, external)

    core = _TorchScriptEdgeVecCore(model).eval()
    traced = torch.jit.trace(core, make_inputs(5), check_trace=False, strict=False)
    for n in (4, 7):
        out = traced(*make_inputs(n))
        assert isinstance(out, tuple) and len(out) == 6
        assert out[0].shape == (n, 1)
        assert torch.isfinite(out[0]).all(), f"non-finite traced MBD energy at N={n}"


def test_mbd_slq_torchscript_core_accepts_variable_atom_and_edge_counts():
    """The matrix-free MBD core must keep the deployment ABI and avoid fixed edge counts."""
    torch.set_default_dtype(torch.float64)
    from mace_ictd.interfaces.lammps_mliap import _TorchScriptEdgeVecCore

    model = _build_model(max_multipole_l=0, dispersion=True).double().eval()
    model.long_range_dispersion_mode = "mbd-slq"
    model.dispersion = LongRangeDispersion(
        feature_dim=model.channels,
        mode="mbd-slq",
        cutoff=0.0,
        pbc=True,
    ).to(dtype=torch.float64)
    model.dispersion.term.num_probes = 3
    model.dispersion.term.lanczos_steps = 6

    def make_inputs(n: int):
        torch.manual_seed(n + 100)
        box = 14.0
        cell = (torch.eye(3, dtype=torch.float64) * box).reshape(1, 3, 3)
        pos = torch.rand(n, 3, dtype=torch.float64) * box
        elements = torch.tensor([1, 6, 7, 8], dtype=torch.long)
        A = elements[torch.arange(n) % elements.numel()]
        batch = torch.zeros(n, dtype=torch.long)
        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(j)
                    dst.append(i)
        src = torch.tensor(src, dtype=torch.long)
        dst = torch.tensor(dst, dtype=torch.long)
        shifts = torch.zeros(src.numel(), 3, dtype=torch.float64)
        edge_vec = pos[dst] - pos[src]
        external = torch.empty(0, dtype=torch.float64)
        return (pos, A, batch, src, dst, shifts, cell, edge_vec, src, dst, shifts, edge_vec, external)

    core = _TorchScriptEdgeVecCore(model).eval()
    traced = torch.jit.trace(core, make_inputs(6), check_trace=False, strict=False)
    for n in (4, 8):
        out = traced(*make_inputs(n))
        assert isinstance(out, tuple) and len(out) == 6
        assert out[0].shape == (n, 1)
        assert torch.isfinite(out[0]).all(), f"non-finite traced SLQ-MBD energy at N={n}"


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
    test_long_range_dispersion_wrapper_matches_pairwise_edge_mode()
    print("OK: unified long-range dispersion wrapper matches pairwise edge mode")
    test_many_body_dispersion_is_finite_invariant_and_nonadditive()
    print("OK: MBD finite, differentiable, invariant, and nonadditive")
    test_many_body_dispersion_slq_basis_matches_dense_oracle()
    print("OK: basis SLQ-MBD matches dense MBD oracle")
    test_model_accepts_explicit_dispersion_neighbor_list()
    print("OK: model accepts an explicit dispersion neighbor list")
    test_model_mbd_dispersion_smoke()
    print("OK: model-level MBD dispersion smoke")
    test_model_mbd_slq_dispersion_smoke()
    print("OK: model-level SLQ-MBD dispersion smoke")
    test_mbd_torchscript_core_accepts_variable_atom_and_edge_counts()
    print("OK: traced MBD deployment core accepts variable atom/edge counts")
    test_mbd_slq_torchscript_core_accepts_variable_atom_and_edge_counts()
    print("OK: traced SLQ-MBD deployment core accepts variable atom/edge counts")
    test_dispersion_neighbor_list_matches_bruteforce()
    print("OK: dispersion neighbor list matches brute-force periodic search")
    test_model_complete_long_range_smoke()
    print("OK: complete long-range smoke (multipole electrostatics + dispersion, both train)")
