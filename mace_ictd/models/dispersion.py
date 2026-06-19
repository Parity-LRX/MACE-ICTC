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
