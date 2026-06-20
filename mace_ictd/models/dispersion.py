"""Learned pairwise C6 dispersion (van der Waals) long-range term.

Completes the long-range physics alongside the multipole electrostatics: a degree-l
ICTD carrier gives the equivariant multipoles for electrostatics, while dispersion
needs only per-atom *invariant* coefficients (C6 is a scalar). The term is
E_disp = -1/2 sum_{i!=j} s6 * C6_ij / (r_ij^6 + R0_ij^6)  (Becke-Johnson-style damping,
smooth as r->0 so the short-range network owns the contact region), with the
geometric-mean combination C6_ij = sqrt(C6_i * C6_j) and R0_ij = R0_i + R0_j.

r^-6 is absolutely convergent in 3D -> a real-space pairwise sum suffices (no Ewald).
The energy depends only on edge lengths and per-atom scalars, so it is exactly
rotation- and translation-invariant by construction.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def dispersion_neighbor_list(pos, batch, cell, cutoff, pbc=True):
    """Periodic pair list within ``cutoff`` (per graph), repo convention
    edge_vec = pos[i] - pos[j] + shifts @ cell, returning (src=j, dst=i, shifts).

    Pure torch (no torch_cluster); O(N_g^2 * images) per graph -> intended for the
    small/medium validation systems (production gets its list from LAMMPS/ASE). Only
    indices+integer shifts are produced here; recompute lengths from ``pos`` for
    differentiable forces. ``cutoff`` should be <~ min lattice length for periodic cells.
    """
    dev, dt = pos.device, pos.dtype
    cutoff_t = torch.as_tensor(float(cutoff), device=dev, dtype=dt)
    src_all, dst_all, shift_all = [], [], []
    for g in range(cell.shape[0]):
        idx = (batch == g).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        p = pos.index_select(0, idx)  # [m, 3]
        c = cell[g]                   # [3, 3]
        if pbc:
            lengths = c.norm(dim=-1).clamp_min(1e-6)
            nmax = torch.ceil(cutoff_t / lengths).to(torch.long)
            axes = [torch.arange(-int(nmax[a]), int(nmax[a]) + 1, device=dev) for a in range(3)]
        else:
            axes = [torch.zeros(1, dtype=torch.long, device=dev) for _ in range(3)]
        shifts = torch.cartesian_prod(*axes).to(dt)  # [S, 3]
        shift_vecs = shifts @ c                       # [S, 3]
        disp = p[:, None, None, :] - p[None, :, None, :] + shift_vecs[None, None, :, :]  # [i, j, S, 3]
        dist = disp.norm(dim=-1)                      # [i, j, S]
        mask = (dist > 1e-8) & (dist <= cutoff_t)
        ii, jj, ss = mask.nonzero(as_tuple=True)
        if ii.numel() == 0:
            continue
        src_all.append(idx[jj])
        dst_all.append(idx[ii])
        shift_all.append(shifts[ss].to(torch.long))
    if not src_all:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return z, z, torch.zeros(0, 3, dtype=torch.long, device=dev)
    return torch.cat(src_all), torch.cat(dst_all), torch.cat(shift_all)


class PairwiseDispersion(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32, r0_floor: float = 0.5):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.r0_floor = float(r0_floor)
        self.c6_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.r0_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.s6 = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        node_feats: torch.Tensor,   # [N, feature_dim] per-atom INVARIANT descriptor
        edge_src: torch.Tensor,     # [E] sender j
        edge_dst: torch.Tensor,     # [E] receiver i
        edge_lengths: torch.Tensor, # [E] |r_ij|
    ) -> torch.Tensor:
        c6 = F.softplus(self.c6_head(node_feats)).squeeze(-1)               # [N] >0
        r0 = F.softplus(self.r0_head(node_feats)).squeeze(-1) + self.r0_floor  # [N] >0 (Angstrom)
        c6_ij = torch.sqrt((c6[edge_src] * c6[edge_dst]).clamp_min(0.0))    # geometric-mean rule
        r0_ij = r0[edge_src] + r0[edge_dst]
        r6 = edge_lengths.clamp_min(1e-6).pow(6)
        e_edge = -self.s6 * c6_ij / (r6 + r0_ij.pow(6))                     # BJ-damped, attractive
        # directed edge list double-counts each pair -> 0.5; partition onto the receiver atom.
        per_atom = node_feats.new_zeros(node_feats.shape[0])
        per_atom.index_add_(0, edge_dst, 0.5 * e_edge)
        return per_atom.unsqueeze(-1)  # [N, 1]


class ManyBodyDispersion(nn.Module):
    """Isotropic QHO many-body dispersion baseline.

    Each atom gets a learned static polarizability alpha_i and oscillator
    frequency omega_i from invariant node features. For each graph, build the
    finite-range coupled-oscillator matrix

        C_ii = omega_i^2 I_3
        C_ij = s_MBD omega_i omega_j sqrt(alpha_i alpha_j) f_damp(r_ij) T_ij

    where T_ij = 3 rr/r^5 - I/r^3. The per-graph MBD energy is the zero-point
    energy shift 0.5 sum_p sqrt(lambda_p) - 1.5 sum_i omega_i, partitioned
    uniformly over atoms. This is O(N^3) and intended as a correctness baseline
    before approximate/deployment kernels.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 32,
        alpha_floor: float = 1.0e-4,
        omega_floor: float = 1.0e-3,
        eig_floor: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.alpha_floor = float(alpha_floor)
        self.omega_floor = float(omega_floor)
        self.eig_floor = float(eig_floor)
        self.alpha_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.omega_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        # Small initial coupling keeps early random models positive definite.
        self.coupling_scale = nn.Parameter(torch.tensor(0.03))
        self.beta_raw = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        node_feats: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_vec: torch.Tensor,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        n_atoms = node_feats.shape[0]
        per_atom = node_feats.new_zeros(n_atoms)
        if n_atoms == 0:
            return per_atom.unsqueeze(-1)

        alpha = F.softplus(self.alpha_head(node_feats)).squeeze(-1) + self.alpha_floor
        omega = F.softplus(self.omega_head(node_feats)).squeeze(-1) + self.omega_floor
        beta = F.softplus(self.beta_raw) + 1.0e-6
        coupling_scale = self.coupling_scale
        eye3 = torch.eye(3, dtype=node_feats.dtype, device=node_feats.device)

        num_graphs = int(num_graphs) if num_graphs is not None else (int(batch.max().item()) + 1 if batch.numel() else 0)
        for g in range(num_graphs):
            if num_graphs == 1:
                idx = torch.arange(n_atoms, dtype=torch.long, device=node_feats.device)
                m = n_atoms
                local = None
                same_graph = None
            else:
                idx = (batch == g).nonzero(as_tuple=True)[0]
                m = idx.numel()
                if m <= 1:
                    continue
                local = torch.full((n_atoms,), -1, dtype=torch.long, device=node_feats.device)
                local[idx] = torch.arange(m, dtype=torch.long, device=node_feats.device)

                same_graph = (batch[edge_src] == g) & (batch[edge_dst] == g) & (edge_src != edge_dst)
            # Directed neighbor lists normally contain i<-j and j<-i. T(r)=T(-r), so keep
            # one canonical orientation to avoid double-strength coupling. For the single-graph
            # AOTI path, keep the edge tensor length fixed and zero non-canonical couplings instead
            # of boolean-filtering to a data-dependent length.
            if same_graph is None:
                es = edge_src
                ed = edge_dst
                ev = edge_vec
                edge_weight = ((edge_src != edge_dst) & (edge_src < edge_dst)).to(dtype=node_feats.dtype)
            else:
                same_graph = same_graph & (edge_src < edge_dst)
                es = edge_src[same_graph]
                ed = edge_dst[same_graph]
                ev = edge_vec[same_graph]
                edge_weight = None

            cmat = torch.diag_embed(omega[idx].repeat_interleave(3).pow(2))
            li = ed if local is None else local[ed]
            lj = es if local is None else local[es]
            r = ev.norm(dim=-1).clamp_min(1.0e-6)
            rhat = ev / r.unsqueeze(-1)
            tensor = (3.0 * rhat.unsqueeze(-1) * rhat.unsqueeze(-2) - eye3) / r.pow(3).view(-1, 1, 1)
            radius = alpha[es].pow(1.0 / 3.0) + alpha[ed].pow(1.0 / 3.0) + 1.0e-6
            damp = 1.0 - torch.exp(-((r / (beta * radius)).clamp_min(0.0)).pow(6))
            pref = coupling_scale * omega[es] * omega[ed] * torch.sqrt((alpha[es] * alpha[ed]).clamp_min(0.0)) * damp
            if edge_weight is not None:
                pref = pref * edge_weight
            blocks = pref.view(-1, 1, 1) * tensor
            rows = (3 * li).unsqueeze(1) + torch.arange(3, device=node_feats.device).view(1, 3)
            cols = (3 * lj).unsqueeze(1) + torch.arange(3, device=node_feats.device).view(1, 3)
            cmat.index_put_((rows.unsqueeze(2), cols.unsqueeze(1)), blocks, accumulate=True)
            cmat.index_put_((cols.unsqueeze(2), rows.unsqueeze(1)), blocks.transpose(-1, -2), accumulate=True)

            eigvals = torch.linalg.eigvalsh(cmat).clamp_min(self.eig_floor)
            e_graph = 0.5 * eigvals.sqrt().sum() - 1.5 * omega[idx].sum()
            per_atom[idx] = e_graph / m
        return per_atom.unsqueeze(-1)


class ManyBodyDispersionSLQ(nn.Module):
    """Matrix-free stochastic-Lanczos QHO many-body dispersion.

    This approximates the dense MBD zero-point energy

        0.5 Tr sqrt(C) - 1.5 sum_i omega_i

    without constructing ``C`` or diagonalizing the full ``3N x 3N`` matrix. The
    expensive operation is a matrix-vector product assembled from the cutoff
    dispersion edge list, so the cost is O(num_probes * lanczos_steps * E_disp)
    for fixed-size graphs/probe settings. The deterministic Rademacher probes
    keep training and force labels reproducible.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 32,
        alpha_floor: float = 1.0e-4,
        omega_floor: float = 1.0e-3,
        eig_floor: float = 1.0e-8,
        num_probes: int = 8,
        lanczos_steps: int = 16,
        probe_mode: str = "rademacher",
        quadrature: str = "eigh",
        sqrt_iterations: int = 8,
    ) -> None:
        super().__init__()
        if probe_mode not in {"rademacher", "atom-rademacher", "basis"}:
            raise ValueError(f"Unsupported SLQ probe mode: {probe_mode!r}")
        if quadrature not in {"eigh", "newton-schulz"}:
            raise ValueError(f"Unsupported SLQ quadrature: {quadrature!r}")
        self.feature_dim = int(feature_dim)
        self.alpha_floor = float(alpha_floor)
        self.omega_floor = float(omega_floor)
        self.eig_floor = float(eig_floor)
        self.num_probes = int(num_probes)
        self.lanczos_steps = int(lanczos_steps)
        self.probe_mode = str(probe_mode)
        self.quadrature = str(quadrature)
        self.sqrt_iterations = int(sqrt_iterations)
        self.alpha_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.omega_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.coupling_scale = nn.Parameter(torch.tensor(0.03))
        self.beta_raw = nn.Parameter(torch.tensor(1.0))

    def _make_probes(self, m: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        dim = 3 * m
        if dim <= 0:
            return torch.zeros(0, 0, 3, device=device, dtype=dtype)
        if self.probe_mode == "basis":
            return torch.eye(dim, device=device, dtype=dtype).reshape(dim, m, 3)
        n_probe = max(int(self.num_probes), 1)
        if self.probe_mode == "atom-rademacher":
            probe_idx = torch.arange(n_probe, device=device, dtype=torch.long).view(n_probe, 1)
            atom_idx = torch.arange(m, device=device, dtype=torch.long).view(1, m)
            h = (
                (probe_idx.to(dtype=dtype) + 1.0) * 12.9898
                + (atom_idx.to(dtype=dtype) + 1.0) * 78.233
                + (probe_idx.to(dtype=dtype) + 1.0) * (atom_idx.to(dtype=dtype) + 1.0) * 0.137
            )
            signs = torch.where(torch.sin(h) >= 0.0, 1.0, -1.0).to(dtype=dtype)
            eye = torch.eye(3, device=device, dtype=dtype)
            return (signs[:, None, :, None] * eye[None, :, None, :]).reshape(3 * n_probe, m, 3)
        probe_idx = torch.arange(n_probe, device=device, dtype=torch.long).view(n_probe, 1, 1)
        atom_idx = torch.arange(m, device=device, dtype=torch.long).view(1, m, 1)
        comp_idx = torch.arange(3, device=device, dtype=torch.long).view(1, 1, 3)
        # Deterministic sinusoidal hash -> Rademacher signs. Avoid integer-parity hashes here:
        # they can collapse to only a few unique probe rows for regular atom/component grids.
        h = (
            (probe_idx.to(dtype=dtype) + 1.0) * 12.9898
            + (atom_idx.to(dtype=dtype) + 1.0) * 78.233
            + (comp_idx.to(dtype=dtype) + 1.0) * 37.719
            + (probe_idx.to(dtype=dtype) + 1.0) * (atom_idx.to(dtype=dtype) + 1.0) * 0.137
            + (probe_idx.to(dtype=dtype) + 1.0) * (comp_idx.to(dtype=dtype) + 1.0) * 0.193
        )
        signs = torch.where(torch.sin(h) >= 0.0, 1.0, -1.0)
        return signs.to(dtype=dtype)

    def _sqrt_first_moment(self, tri: torch.Tensor) -> torch.Tensor:
        if self.quadrature == "newton-schulz":
            eye = torch.eye(tri.size(-1), dtype=tri.dtype, device=tri.device).unsqueeze(0).expand_as(tri)
            scale = tri.norm(dim=(-2, -1)).clamp_min(self.eig_floor)
            y = tri / scale.view(-1, 1, 1)
            z = eye
            for _ in range(max(int(self.sqrt_iterations), 1)):
                t = 0.5 * (3.0 * eye - torch.matmul(z, y))
                y = torch.matmul(y, t)
                z = torch.matmul(t, z)
            return (y * scale.sqrt().view(-1, 1, 1))[:, 0, 0].clamp_min(self.eig_floor ** 0.5)
        evals, evecs = torch.linalg.eigh(tri)
        sqrt_evals = evals.clamp_min(self.eig_floor).sqrt()
        weights = evecs[:, 0, :].square()
        return (weights * sqrt_evals).sum(dim=-1)

    def forward(
        self,
        node_feats: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_vec: torch.Tensor,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        n_atoms = node_feats.shape[0]
        per_atom = node_feats.new_zeros(n_atoms)
        if n_atoms == 0:
            return per_atom.unsqueeze(-1)

        alpha = F.softplus(self.alpha_head(node_feats)).squeeze(-1) + self.alpha_floor
        omega = F.softplus(self.omega_head(node_feats)).squeeze(-1) + self.omega_floor
        beta = F.softplus(self.beta_raw) + 1.0e-6
        coupling_scale = self.coupling_scale
        eye3 = torch.eye(3, dtype=node_feats.dtype, device=node_feats.device)

        num_graphs = int(num_graphs) if num_graphs is not None else (int(batch.max().item()) + 1 if batch.numel() else 0)
        for g in range(num_graphs):
            if num_graphs == 1:
                idx = torch.arange(n_atoms, dtype=torch.long, device=node_feats.device)
                m = n_atoms
                local = None
                same_graph = None
            else:
                idx = (batch == g).nonzero(as_tuple=True)[0]
                m = idx.size(0)
                if m <= 0:
                    continue
                local = torch.full((n_atoms,), -1, dtype=torch.long, device=node_feats.device)
                local[idx] = torch.arange(m, dtype=torch.long, device=node_feats.device)
                same_graph = (batch[edge_src] == g) & (batch[edge_dst] == g)
            steps = max(int(self.lanczos_steps), 1)
            if self.probe_mode == "basis":
                steps = min(steps, 3 * int(m))

            if same_graph is None:
                es = edge_src
                ed = edge_dst
                ev = edge_vec
                edge_weight = ((edge_src != edge_dst) & (edge_src < edge_dst)).to(dtype=node_feats.dtype)
            else:
                same_graph = same_graph & (edge_src < edge_dst)
                es = edge_src[same_graph]
                ed = edge_dst[same_graph]
                ev = edge_vec[same_graph]
                edge_weight = None
            li = ed if local is None else local[ed]
            lj = es if local is None else local[es]
            r = ev.norm(dim=-1).clamp_min(1.0e-6)
            rhat = ev / r.unsqueeze(-1)
            tensor = (3.0 * rhat.unsqueeze(-1) * rhat.unsqueeze(-2) - eye3) / r.pow(3).view(-1, 1, 1)
            radius = alpha[es].pow(1.0 / 3.0) + alpha[ed].pow(1.0 / 3.0) + 1.0e-6
            damp = 1.0 - torch.exp(-((r / (beta * radius)).clamp_min(0.0)).pow(6))
            pref = coupling_scale * omega[es] * omega[ed] * torch.sqrt((alpha[es] * alpha[ed]).clamp_min(0.0)) * damp
            if edge_weight is not None:
                pref = pref * edge_weight
            blocks = pref.view(-1, 1, 1) * tensor

            omega_local = omega[idx]

            def matvec(v: torch.Tensor) -> torch.Tensor:
                y = omega_local.square().view(1, -1, 1) * v
                v_j = v.index_select(1, lj)
                v_i = v.index_select(1, li)
                contrib_i = torch.matmul(blocks.unsqueeze(0), v_j.unsqueeze(-1)).squeeze(-1)
                contrib_j = torch.matmul(blocks.transpose(-1, -2).unsqueeze(0), v_i.unsqueeze(-1)).squeeze(-1)
                idx_i = li.view(1, -1, 1).expand(v.size(0), -1, 3)
                idx_j = lj.view(1, -1, 1).expand(v.size(0), -1, 3)
                y = y.scatter_add(1, idx_i, contrib_i)
                y = y.scatter_add(1, idx_j, contrib_j)
                return y

            probes = self._make_probes(m, device=node_feats.device, dtype=node_feats.dtype)
            n_probe = probes.size(0)
            q = probes.reshape(n_probe, -1)
            dim = q.size(1)
            q_norm = q.norm(dim=-1).clamp_min(1.0e-14)
            q = q / q_norm.view(-1, 1)
            q_prev = torch.zeros_like(q)
            beta_prev = q.new_zeros(n_probe)
            alphas: list[torch.Tensor] = []
            betas: list[torch.Tensor] = []
            basis: list[torch.Tensor] = []
            for step in range(steps):
                z = matvec(q.reshape(n_probe, m, 3)).reshape(n_probe, 3 * m)
                if step > 0:
                    z = z - beta_prev.view(-1, 1) * q_prev
                a = (q * z).sum(dim=-1)
                z = z - a.view(-1, 1) * q
                # Modified Gram-Schmidt is cheap here and keeps the tiny Lanczos matrices stable
                # enough for differentiable training smoke tests.
                for old_q in basis:
                    z = z - (z * old_q).sum(dim=-1, keepdim=True) * old_q
                b = z.norm(dim=-1)
                alphas.append(a)
                if step + 1 < steps:
                    betas.append(b)
                basis.append(q)
                q_prev = q
                q = z / b.clamp_min(1.0e-14).view(-1, 1)
                beta_prev = b

            tri = q.new_zeros(n_probe, steps, steps)
            diag = torch.stack(alphas, dim=1)
            ar = torch.arange(steps, device=node_feats.device)
            tri[:, ar, ar] = diag
            if steps > 1:
                off = torch.stack(betas, dim=1)
                ar0 = torch.arange(steps - 1, device=node_feats.device)
                tri[:, ar0, ar0 + 1] = off
                tri[:, ar0 + 1, ar0] = off
            estimates = q_norm.square() * self._sqrt_first_moment(tri)
            if self.probe_mode == "basis":
                trace_sqrt = estimates.sum()
            elif self.probe_mode == "atom-rademacher":
                trace_sqrt = estimates.reshape(max(int(self.num_probes), 1), 3).sum(dim=1).mean()
            else:
                trace_sqrt = estimates.mean()
            e_graph = 0.5 * trace_sqrt - 1.5 * omega_local.sum()
            per_atom[idx] = e_graph / m
        return per_atom.unsqueeze(-1)


class LongRangeDispersion(nn.Module):
    """Unified long-range dispersion term.

    This wrapper keeps the model forward independent of the concrete dispersion
    implementation. It exposes the learned pairwise-C6 term, the dense MBD
    oracle, and the matrix-free SLQ approximation through one model interface.
    """

    SUPPORTED_MODES = {"pairwise-c6", "mbd", "mbd-slq"}

    def __init__(
        self,
        *,
        feature_dim: int,
        mode: str = "pairwise-c6",
        hidden_dim: int = 32,
        cutoff: float = 10.0,
        pbc: bool = True,
        slq_num_probes: int = 8,
        slq_lanczos_steps: int = 16,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        if self.mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported long-range dispersion mode {self.mode!r}; "
                f"supported modes: {sorted(self.SUPPORTED_MODES)}"
            )
        self.cutoff = float(cutoff)
        self.pbc = bool(pbc)
        self.slq_num_probes = int(slq_num_probes)
        self.slq_lanczos_steps = int(slq_lanczos_steps)
        if self.mode == "pairwise-c6":
            self.term = PairwiseDispersion(feature_dim=feature_dim, hidden_dim=hidden_dim)
        elif self.mode == "mbd":
            self.term = ManyBodyDispersion(feature_dim=feature_dim, hidden_dim=hidden_dim)
        elif self.mode == "mbd-slq":
            self.term = ManyBodyDispersionSLQ(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_probes=self.slq_num_probes,
                lanczos_steps=self.slq_lanczos_steps,
            )
        else:  # pragma: no cover - guarded above; future modes land here explicitly.
            raise ValueError(f"Unsupported long-range dispersion mode {self.mode!r}")

    def forward(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        *,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_lengths: torch.Tensor,
        edge_vec: torch.Tensor | None = None,
        cutoff: float | None = None,
        pbc: bool | None = None,
    ) -> torch.Tensor:
        cutoff_value = self.cutoff if cutoff is None else float(cutoff)
        pbc_value = self.pbc if pbc is None else bool(pbc)
        if cutoff_value and cutoff_value > 0.0:
            d_src, d_dst, d_shift = dispersion_neighbor_list(
                pos, batch, cell, cutoff_value, pbc=pbc_value
            )
            shift_vecs = torch.einsum("ni,nij->nj", d_shift.to(pos.dtype), cell[batch[d_dst]])
            d_vec = pos[d_dst] - pos[d_src] + shift_vecs
            d_len = d_vec.norm(dim=1)
        else:
            d_src, d_dst, d_len, d_vec = edge_src, edge_dst, edge_lengths, edge_vec

        if self.mode == "pairwise-c6":
            return self.term(node_feats, d_src, d_dst, d_len)
        if self.mode in {"mbd", "mbd-slq"}:
            if d_vec is None:
                raise ValueError("MBD dispersion requires edge_vec or cutoff-based neighbor construction")
            return self.term(node_feats, batch, d_src, d_dst, d_vec, num_graphs=int(cell.shape[0]))
        raise ValueError(f"Unsupported long-range dispersion mode {self.mode!r}")


def normalize_dispersion_mode(
    *,
    long_range_dispersion: bool = False,
    long_range_dispersion_mode: str | None = None,
) -> str:
    """Resolve legacy boolean and explicit mode into one stable mode string."""

    if long_range_dispersion_mode is None:
        return "pairwise-c6" if bool(long_range_dispersion) else "none"
    mode = str(long_range_dispersion_mode)
    if mode == "none" and bool(long_range_dispersion):
        return "pairwise-c6"
    return mode


def build_long_range_dispersion(
    *,
    mode: str,
    feature_dim: int,
    hidden_dim: int = 32,
    cutoff: float = 10.0,
    pbc: bool = True,
    slq_num_probes: int = 8,
    slq_lanczos_steps: int = 16,
) -> LongRangeDispersion | None:
    mode = str(mode)
    if mode == "none":
        return None
    return LongRangeDispersion(
        feature_dim=feature_dim,
        mode=mode,
        hidden_dim=hidden_dim,
        cutoff=cutoff,
        pbc=pbc,
        slq_num_probes=slq_num_probes,
        slq_lanczos_steps=slq_lanczos_steps,
    )
