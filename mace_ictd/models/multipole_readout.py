"""Equivariant multipole readout for long-range electrostatics.

A degree-l irreducible ICTD carrier *is* a rank-l multipole moment. Given the
full-SO(3) node features of the last interaction layer (l = 0..lmax, the
"fusion route" tap rather than the scalar-only readout), this module produces
per-atom Cartesian multipole sources -- monopole (l=0), dipole (l=1),
quadrupole (l=2) -- ready for the reciprocal-space ``multipole_energy`` PME path
in :mod:`mace_ictd.models.long_range`.

Equivariance: the channel mixers act on the channel axis only (shared across the
2l+1 angular components), so they commute with rotations; the ICTD-block ->
Cartesian decode is the existing convention-faithful
:class:`PhysicalTensorICTDRecovery`. Hence the monopole is invariant, the dipole
rotates by R, and the quadrupole by R Q R^T.
"""

from __future__ import annotations

import torch
from torch import nn

from mace_ictd.models.pure_cartesian_ictd_layers import (
    PhysicalTensorICTDRecovery,
    _split_irreps,
)


class MultipoleReadout(nn.Module):
    def __init__(
        self,
        channels: int,
        lmax: int,
        max_multipole_l: int,
        source_channels: int = 1,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.max_multipole_l = int(max_multipole_l)
        self.source_channels = int(source_channels)
        if self.max_multipole_l < 0:
            raise ValueError(f"max_multipole_l must be >= 0, got {self.max_multipole_l}")
        if self.max_multipole_l > self.lmax:
            raise ValueError(
                f"max_multipole_l={self.max_multipole_l} exceeds model lmax={self.lmax}; "
                "the last layer must expose at least that degree"
            )
        if self.max_multipole_l > 2:
            raise NotImplementedError(
                "multipole long-range currently supports max_multipole_l <= 2 "
                "(monopole/dipole/quadrupole)"
            )
        # Per-degree channel mixers: C -> source_channels, acting ONLY on the
        # channel axis (so equivariance is preserved). No bias (would break l>0).
        self.mix = nn.ModuleList(
            [nn.Linear(self.channels, self.source_channels, bias=False) for _ in range(self.max_multipole_l + 1)]
        )
        self.dipole_recovery = (
            PhysicalTensorICTDRecovery(
                rank=1, channels_in=self.source_channels, lmax_in=1, include_trace_chain=False
            )
            if self.max_multipole_l >= 1
            else None
        )
        self.quad_recovery = (
            PhysicalTensorICTDRecovery(
                rank=2,
                channels_in=self.source_channels,
                lmax_in=2,
                include_trace_chain=False,
                rank2_mode="symmetric",
            )
            if self.max_multipole_l >= 2
            else None
        )

    def _chan_mix(self, block_l: torch.Tensor, l: int) -> torch.Tensor:
        # block_l: [N, C, 2l+1] -> [N, source_channels, 2l+1]
        return self.mix[l](block_l.transpose(-1, -2)).transpose(-1, -2)

    def forward(
        self, node_feats_so3: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """node_feats_so3: ``[N, channels*(lmax+1)**2]`` (concat over l=0..lmax of
        ``[N, channels, 2l+1]``). Returns ``(monopole [N, S], dipole [N, S, 3] or
        None, quadrupole [N, S, 3, 3] or None)`` with ``S = source_channels``."""
        blocks = _split_irreps(node_feats_so3, self.channels, self.lmax)
        monopole = self._chan_mix(blocks[0], 0).squeeze(-1)  # [N, S]
        dipole = None
        quadrupole = None
        if self.max_multipole_l >= 1:
            d_block = self._chan_mix(blocks[1], 1)  # [N, S, 3]
            dipole = self.dipole_recovery({1: d_block}, squeeze_channel=False)  # [N, S, 3]
        if self.max_multipole_l >= 2:
            q_block = self._chan_mix(blocks[2], 2)  # [N, S, 5]
            quadrupole = self.quad_recovery({2: q_block}, squeeze_channel=False)  # [N, S, 3, 3]
        return monopole, dipole, quadrupole
