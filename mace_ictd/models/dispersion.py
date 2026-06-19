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
