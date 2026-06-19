from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mace_ictd.utils.checkpoint_metadata import (
    derive_long_range_far_max_radius_multiplier,
)


def _build_integer_k_lattice(kmax: int, include_k0: bool) -> torch.Tensor:
    values: list[list[float]] = []
    for i in range(-int(kmax), int(kmax) + 1):
        for j in range(-int(kmax), int(kmax) + 1):
            for k in range(-int(kmax), int(kmax) + 1):
                if not include_k0 and i == 0 and j == 0 and k == 0:
                    continue
                values.append([float(i), float(j), float(k)])
    if not values:
        return torch.zeros((0, 3), dtype=torch.float32)
    return torch.tensor(values, dtype=torch.float32)


class LatentSourceHead(nn.Module):
    """Map node invariant features to latent reciprocal-space sources."""

    def __init__(self, feature_dim: int, hidden_dim: int, source_channels: int = 1):
        super().__init__()
        self.source_channels = int(source_channels)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.source_channels),
        )

    def forward(self, invariant_features: torch.Tensor) -> torch.Tensor:
        return self.net(invariant_features)


def _fft_integer_frequencies(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.fft.fftfreq(size, d=1.0 / float(size), device=device).to(dtype=dtype)


def _effective_cell_for_boundary(
    cell: torch.Tensor,
    *,
    boundary: str,
    slab_padding_factor: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    effective_cell = cell.to(dtype=dtype)
    if boundary == "slab":
        effective_cell = effective_cell.clone()
        effective_cell[2] = effective_cell[2] * float(max(int(slab_padding_factor), 1))
    return effective_cell


def _prepare_frac_for_boundary(
    pos: torch.Tensor,
    cell: torch.Tensor,
    *,
    boundary: str,
    slab_padding_factor: int,
) -> torch.Tensor:
    inv_cell = torch.linalg.inv(cell)
    frac = torch.einsum("ni,ij->nj", pos, inv_cell)
    if boundary == "periodic":
        return frac - torch.floor(frac)

    frac = frac.clone()
    frac[:, :2] = frac[:, :2] - torch.floor(frac[:, :2])
    pad = float(max(int(slab_padding_factor), 1))
    z_offset = 0.5 * (pad - 1.0) / pad
    frac[:, 2] = frac[:, 2] / pad + z_offset
    return frac


def _assignment_stencil_size(assignment: str) -> int:
    if assignment == "cic":
        return 2
    if assignment == "tsc":
        return 3
    if assignment == "pcs":
        return 4
    raise ValueError(f"Unsupported mesh assignment: {assignment!r}")


def _build_assignment_offsets(assignment: str) -> torch.Tensor:
    stencil = _assignment_stencil_size(assignment)
    values = torch.arange(stencil, dtype=torch.long)
    return torch.cartesian_prod(values, values, values)


def _assignment_kernel_1d(scaled: torch.Tensor, assignment: str) -> tuple[torch.Tensor, torch.Tensor]:
    if assignment == "cic":
        base = torch.floor(scaled).to(dtype=torch.long)
        frac = scaled - base.to(dtype=scaled.dtype)
        weights = torch.stack([1.0 - frac, frac], dim=-1)
        return base, weights

    if assignment == "tsc":
        shifted = scaled - 0.5
        base = torch.floor(shifted).to(dtype=torch.long)
        local = scaled - base.to(dtype=scaled.dtype)
        weights = torch.stack(
            [
                0.5 * (1.5 - local).square(),
                0.75 - (local - 1.0).square(),
                0.5 * (local - 0.5).square(),
            ],
            dim=-1,
        )
        return base, weights

    if assignment == "pcs":
        floor_scaled = torch.floor(scaled).to(dtype=torch.long)
        base = floor_scaled - 1
        frac = scaled - floor_scaled.to(dtype=scaled.dtype)
        frac2 = frac * frac
        frac3 = frac2 * frac
        weights = torch.stack(
            [
                ((1.0 - frac).pow(3)) / 6.0,
                (3.0 * frac3 - 6.0 * frac2 + 4.0) / 6.0,
                (-3.0 * frac3 + 3.0 * frac2 + 3.0 * frac + 1.0) / 6.0,
                frac3 / 6.0,
            ],
            dim=-1,
        )
        return base, weights

    raise ValueError(f"Unsupported mesh assignment: {assignment!r}")


def _assignment_weights_from_scaled(scaled: torch.Tensor, assignment: str) -> tuple[torch.Tensor, torch.Tensor]:
    base_components: list[torch.Tensor] = []
    weight_components: list[torch.Tensor] = []
    for dim in range(3):
        base_dim, weights_dim = _assignment_kernel_1d(scaled[:, dim], assignment)
        base_components.append(base_dim)
        weight_components.append(weights_dim)
    base = torch.stack(base_components, dim=-1)
    weights = (
        weight_components[0][:, :, None, None]
        * weight_components[1][:, None, :, None]
        * weight_components[2][:, None, None, :]
    ).reshape(scaled.size(0), -1)
    return base, weights


def _assignment_window_1d(
    *,
    mesh_size: int,
    assignment: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    exponent = _assignment_stencil_size(assignment)
    freq = _fft_integer_frequencies(mesh_size, device=device, dtype=dtype)
    return torch.sinc(freq / float(mesh_size)).pow(exponent)


def _apply_mesh_boundary(idx: torch.Tensor, *, mesh_size: int, boundary: str) -> torch.Tensor:
    if boundary == "periodic":
        return torch.remainder(idx, mesh_size)
    idx_wrapped = idx.clone()
    idx_wrapped[..., 0] = torch.remainder(idx_wrapped[..., 0], mesh_size)
    idx_wrapped[..., 1] = torch.remainder(idx_wrapped[..., 1], mesh_size)
    idx_wrapped[..., 2] = idx_wrapped[..., 2].clamp(0, mesh_size - 1)
    return idx_wrapped


def _spread_source_to_mesh(
    frac: torch.Tensor,
    source: torch.Tensor,
    *,
    mesh_size: int,
    assignment: str,
    assignment_offsets: torch.Tensor,
    boundary: str,
) -> torch.Tensor:
    channels = int(source.size(1))
    mesh = source.new_zeros((mesh_size, mesh_size, mesh_size, channels))
    flat_mesh = mesh.view(-1, channels)
    scaled = frac * float(mesh_size)
    base, stencil_weights = _assignment_weights_from_scaled(scaled, assignment)
    idx = _apply_mesh_boundary(
        base.unsqueeze(1) + assignment_offsets.to(device=base.device).unsqueeze(0),
        mesh_size=mesh_size,
        boundary=boundary,
    )
    flat_idx = ((idx[..., 0] * mesh_size) + idx[..., 1]) * mesh_size + idx[..., 2]
    flat_mesh.scatter_add_(
        0,
        flat_idx.reshape(-1, 1).expand(-1, channels),
        (source.unsqueeze(1) * stencil_weights.unsqueeze(-1)).reshape(-1, channels),
    )
    return mesh


def _gather_source_from_mesh(
    frac: torch.Tensor,
    mesh: torch.Tensor,
    *,
    mesh_size: int,
    assignment: str,
    assignment_offsets: torch.Tensor,
    boundary: str,
) -> torch.Tensor:
    channels = int(mesh.size(-1))
    flat_mesh = mesh.view(-1, channels)
    scaled = frac * float(mesh_size)
    base, stencil_weights = _assignment_weights_from_scaled(scaled, assignment)
    idx = _apply_mesh_boundary(
        base.unsqueeze(1) + assignment_offsets.to(device=base.device).unsqueeze(0),
        mesh_size=mesh_size,
        boundary=boundary,
    )
    flat_idx = ((idx[..., 0] * mesh_size) + idx[..., 1]) * mesh_size + idx[..., 2]
    gathered = flat_mesh.index_select(0, flat_idx.reshape(-1)).reshape(frac.size(0), -1, channels)
    return (gathered * stencil_weights.unsqueeze(-1)).sum(dim=1)


class FeatureSpectralFilterGrid(nn.Module):
    """Apply a learnable radial spectral filter on a regular periodic/slab mesh."""

    def __init__(
        self,
        *,
        mesh_size: int,
        channels: int,
        hidden_dim: int,
        boundary: str = "periodic",
        slab_padding_factor: int = 2,
        include_k0: bool = False,
        k_norm_floor: float = 1.0e-6,
    ):
        super().__init__()
        self.mesh_size = int(mesh_size)
        self.channels = int(channels)
        self.boundary = str(boundary)
        if self.boundary not in {"periodic", "slab"}:
            raise ValueError(f"Unsupported feature spectral boundary: {self.boundary!r}")
        self.slab_padding_factor = max(int(slab_padding_factor), 1)
        self.include_k0 = bool(include_k0)
        self.k_norm_floor = float(k_norm_floor)
        self.radial_filter = RadialSpectralFilter(hidden_dim=hidden_dim, k_norm_floor=k_norm_floor)
        self.channel_scale_raw = nn.Parameter(torch.zeros(self.channels))

    def _effective_cell(self, cell: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return _effective_cell_for_boundary(
            cell,
            boundary=self.boundary,
            slab_padding_factor=self.slab_padding_factor,
            dtype=dtype,
        )

    def build_k_norms(self, cell: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        freq = _fft_integer_frequencies(self.mesh_size, device=cell.device, dtype=dtype)
        kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing="ij")
        integer_k = torch.stack([kx, ky, kz], dim=-1).reshape(-1, 3)
        effective_cell = self._effective_cell(cell, dtype=dtype)
        inv_cell = torch.linalg.inv(effective_cell)
        # k = 2*pi * m @ inv(cell)^T (transpose required for O(3) equivariance on non-orthogonal
        # cells; without it |k| -- hence the radial filter -- is not rotation-invariant). No-op for
        # orthogonal cells. Matches the long-range build_k_norms / _build_k_cart_flat convention.
        k_cart = 2.0 * math.pi * torch.matmul(integer_k, inv_cell.transpose(-1, -2))
        return torch.linalg.vector_norm(k_cart, dim=-1).reshape(self.mesh_size, self.mesh_size, self.mesh_size)

    def forward(self, mesh: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        mesh_dtype = mesh.dtype
        mesh_complex = torch.fft.fftn(mesh, dim=(0, 1, 2))
        k_norms = self.build_k_norms(cell, dtype=mesh_dtype)
        spectral_weights = self.radial_filter(k_norms)
        if not self.include_k0:
            spectral_weights = torch.where(
                k_norms > self.k_norm_floor,
                spectral_weights,
                torch.zeros_like(spectral_weights),
            )
        channel_scale = torch.nn.functional.softplus(self.channel_scale_raw).to(dtype=mesh_dtype)
        filtered = torch.fft.ifftn(
            mesh_complex * spectral_weights.unsqueeze(-1) * channel_scale.view(1, 1, 1, -1),
            dim=(0, 1, 2),
        )
        return filtered.real


class FeatureSpectralResidualBlock(nn.Module):
    """Low-rank feature-space spectral filter with FFT mesh projection."""

    def __init__(
        self,
        *,
        feature_dim: int,
        bottleneck_dim: int = 8,
        mesh_size: int = 16,
        filter_hidden_dim: int = 64,
        boundary: str = "periodic",
        slab_padding_factor: int = 2,
        neutralize: bool = True,
        include_k0: bool = False,
        assignment: str = "cic",
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.mesh_size = int(mesh_size)
        self.boundary = str(boundary)
        if self.boundary not in {"periodic", "slab"}:
            raise ValueError(f"Unsupported feature spectral boundary: {self.boundary!r}")
        self.slab_padding_factor = max(int(slab_padding_factor), 1)
        self.neutralize = bool(neutralize)
        self.include_k0 = bool(include_k0)
        self.assignment = str(assignment)
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.in_proj = nn.Linear(self.feature_dim, self.bottleneck_dim)
        self.out_proj = nn.Linear(self.bottleneck_dim, self.feature_dim, bias=False)
        self.mesh_filter = FeatureSpectralFilterGrid(
            mesh_size=self.mesh_size,
            channels=self.bottleneck_dim,
            hidden_dim=int(filter_hidden_dim),
            boundary=self.boundary,
            slab_padding_factor=self.slab_padding_factor,
            include_k0=self.include_k0,
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        assignment_offsets = _build_assignment_offsets(self.assignment)
        self.register_buffer("assignment_offsets", assignment_offsets, persistent=False)

    def _neutralize_source(self, source: torch.Tensor) -> torch.Tensor:
        if not self.neutralize:
            return source
        return source - source.mean(dim=0, keepdim=True)

    def _effective_cell(self, cell: torch.Tensor) -> torch.Tensor:
        return _effective_cell_for_boundary(
            cell,
            boundary=self.boundary,
            slab_padding_factor=self.slab_padding_factor,
            dtype=cell.dtype,
        )

    def _prepare_frac(self, pos: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        return _prepare_frac_for_boundary(
            pos,
            cell,
            boundary=self.boundary,
            slab_padding_factor=self.slab_padding_factor,
        )

    def _apply_boundary(self, idx: torch.Tensor) -> torch.Tensor:
        return _apply_mesh_boundary(idx, mesh_size=self.mesh_size, boundary=self.boundary)

    def _spread_to_mesh(self, frac: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return _spread_source_to_mesh(
            frac,
            source,
            mesh_size=self.mesh_size,
            assignment=self.assignment,
            assignment_offsets=self.assignment_offsets,
            boundary=self.boundary,
        )

    def _gather_from_mesh(self, frac: torch.Tensor, mesh: torch.Tensor) -> torch.Tensor:
        return _gather_source_from_mesh(
            frac,
            mesh,
            mesh_size=self.mesh_size,
            assignment=self.assignment,
            assignment_offsets=self.assignment_offsets,
            boundary=self.boundary,
        )

    def _filter_single_graph(self, pos: torch.Tensor, cell: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        frac = self._prepare_frac(pos, cell)
        mesh = self._spread_to_mesh(frac, source)
        filtered_mesh = self.mesh_filter(mesh, cell)
        return self._gather_from_mesh(frac, filtered_mesh)

    def forward(
        self,
        invariant_features: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.in_proj(self.input_norm(invariant_features))
        filtered_source = torch.zeros_like(source)

        for graph_idx in range(cell.size(0)):
            node_index = torch.nonzero(batch == graph_idx, as_tuple=False).view(-1)
            if node_index.numel() == 0:
                continue
            filtered_source.index_copy_(
                0,
                node_index,
                self._filter_single_graph(
                    pos.index_select(0, node_index),
                    cell[graph_idx],
                    self._neutralize_source(source.index_select(0, node_index)),
                ),
            )

        residual = self.out_proj(filtered_source)
        gated_residual = torch.tanh(self.gate).to(dtype=residual.dtype) * residual
        return invariant_features + gated_residual, source


class RadialSpectralFilter(nn.Module):
    """Learnable radial filter that modulates a Coulomb-like k-space kernel."""

    def __init__(self, hidden_dim: int, k_norm_floor: float = 1.0e-6):
        super().__init__()
        self.k_norm_floor = float(k_norm_floor)
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, k_norms: torch.Tensor) -> torch.Tensor:
        safe_k = k_norms.clamp_min(self.k_norm_floor)
        x0 = torch.log1p(safe_k)
        x1 = x0 * x0
        x2 = torch.reciprocal(safe_k)
        x = torch.stack([x0, x1, x2], dim=-1)
        learned_scale = torch.nn.functional.softplus(self.net(x).squeeze(-1))
        base_kernel = 4.0 * math.pi / (safe_k * safe_k)
        return base_kernel * learned_scale


class ReciprocalGreenKernel(nn.Module):
    """Base Poisson kernel with an optional learnable radial modifier."""

    def __init__(self, *, green_mode: str, hidden_dim: int, k_norm_floor: float = 1.0e-6):
        super().__init__()
        if green_mode not in {"poisson", "learned_poisson"}:
            raise ValueError(f"Unsupported long-range green mode: {green_mode!r}")
        self.green_mode = str(green_mode)
        self.k_norm_floor = float(k_norm_floor)
        self.learned_filter = (
            RadialSpectralFilter(hidden_dim=hidden_dim, k_norm_floor=k_norm_floor)
            if self.green_mode == "learned_poisson"
            else None
        )

    def forward(self, k_norms: torch.Tensor) -> torch.Tensor:
        if self.learned_filter is not None:
            return self.learned_filter(k_norms)
        safe_k = k_norms.clamp_min(self.k_norm_floor)
        return 4.0 * math.pi / (safe_k * safe_k)


class ReciprocalSpectralKernel3D(nn.Module):
    """Direct k-space prototype for periodic reciprocal-space long-range energy."""

    def __init__(
        self,
        *,
        kmax: int = 2,
        filter_hidden_dim: int = 64,
        include_k0: bool = False,
        reciprocal_backend: str = "direct_kspace",
        energy_partition: str = "potential",
        k_norm_floor: float = 1.0e-6,
    ):
        super().__init__()
        if int(kmax) < 0:
            raise ValueError(f"kmax must be >= 0, got {kmax}")
        if reciprocal_backend != "direct_kspace":
            raise ValueError(f"Unsupported reciprocal backend: {reciprocal_backend!r}")
        if energy_partition not in ("potential", "uniform"):
            raise ValueError(f"Unsupported reciprocal energy partition: {energy_partition!r}")
        self.kmax = int(kmax)
        self.include_k0 = bool(include_k0)
        self.reciprocal_backend = str(reciprocal_backend)
        self.energy_partition = str(energy_partition)
        self.k_norm_floor = float(k_norm_floor)
        self.register_buffer("integer_k_lattice", _build_integer_k_lattice(self.kmax, self.include_k0))
        if self.integer_k_lattice.shape[0] == 0:
            raise ValueError("ReciprocalSpectralKernel3D produced an empty k-lattice; increase kmax or enable include_k0")
        self.spectral_filter = RadialSpectralFilter(hidden_dim=filter_hidden_dim, k_norm_floor=k_norm_floor)

    @property
    def num_k(self) -> int:
        return int(self.integer_k_lattice.shape[0])

    def build_k_lattice(self, cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inv_cells = torch.linalg.inv(cell)
        k_lattice = self.integer_k_lattice.to(device=cell.device, dtype=cell.dtype)
        k_cart = 2.0 * math.pi * torch.einsum("kd,bdh->bkh", k_lattice, inv_cells)
        k_norms = torch.linalg.vector_norm(k_cart, dim=-1)
        volumes = torch.abs(torch.linalg.det(cell)).clamp_min(self.k_norm_floor)
        return k_cart, k_norms, volumes

    def compute_structure_factor(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inv_cells = torch.linalg.inv(cell)
        atom_inv_cells = inv_cells.index_select(0, batch)
        frac = torch.einsum("ni,nij->nj", pos, atom_inv_cells)
        k_lattice = self.integer_k_lattice.to(device=pos.device, dtype=pos.dtype)
        phases = 2.0 * math.pi * torch.matmul(frac, k_lattice.transpose(0, 1))
        cos_phase = torch.cos(phases)
        sin_phase = torch.sin(phases)

        graph_ids = torch.arange(cell.size(0), device=batch.device, dtype=batch.dtype)
        graph_mask = (batch.unsqueeze(1) == graph_ids.unsqueeze(0)).to(dtype=pos.dtype)
        weighted_cos = source.unsqueeze(1) * cos_phase.unsqueeze(-1)
        weighted_sin = source.unsqueeze(1) * sin_phase.unsqueeze(-1)
        structure_cos = torch.einsum("nb,nkc->bkc", graph_mask, weighted_cos)
        structure_sin = torch.einsum("nb,nkc->bkc", graph_mask, weighted_sin)
        return structure_cos, structure_sin, cos_phase, sin_phase

    def apply_spectral_filter(self, structure_cos: torch.Tensor, structure_sin: torch.Tensor, k_norms: torch.Tensor) -> torch.Tensor:
        spectral_weights = self.spectral_filter(k_norms)
        if not self.include_k0:
            spectral_weights = torch.where(
                k_norms > self.k_norm_floor,
                spectral_weights,
                torch.zeros_like(spectral_weights),
            )
        return spectral_weights

    def forward(self, pos: torch.Tensor, batch: torch.Tensor, cell: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(batch, minlength=cell.size(0)).to(dtype=pos.dtype).clamp_min(1.0)
        k_cart, k_norms, volumes = self.build_k_lattice(cell)
        structure_cos, structure_sin, cos_phase, sin_phase = self.compute_structure_factor(pos, batch, cell, source)
        del k_cart
        spectral_weights = self.apply_spectral_filter(structure_cos, structure_sin, k_norms)

        source_cos = structure_cos.index_select(0, batch)
        source_sin = structure_sin.index_select(0, batch)
        spectral_per_atom = spectral_weights.index_select(0, batch)
        volume_per_atom = volumes.index_select(0, batch).unsqueeze(-1)
        potential = (
            spectral_per_atom.unsqueeze(-1)
            * (source_cos * cos_phase.unsqueeze(-1) + source_sin * sin_phase.unsqueeze(-1))
        ).sum(dim=1) / volume_per_atom

        if self.energy_partition == "potential":
            atom_energy = 0.5 * (source * potential).sum(dim=-1, keepdim=True)
            return atom_energy

        graph_total = 0.5 * (
            spectral_weights.unsqueeze(-1) * (structure_cos.square() + structure_sin.square())
        ).sum(dim=(1, 2)) / volumes
        atom_energy = graph_total.index_select(0, batch).unsqueeze(-1) / counts.index_select(0, batch).unsqueeze(-1)
        return atom_energy


class MeshLongRangeKernel3D(nn.Module):
    """Mesh/FFT reciprocal kernel with periodic/slab boundary support."""

    def __init__(
        self,
        *,
        mesh_size: int = 16,
        filter_hidden_dim: int = 64,
        boundary: str = "periodic",
        slab_padding_factor: int = 2,
        include_k0: bool = False,
        energy_partition: str = "potential",
        green_mode: str = "poisson",
        assignment: str = "cic",
        full_ewald: bool = False,
        k_norm_floor: float = 1.0e-6,
    ):
        super().__init__()
        if boundary not in {"periodic", "slab"}:
            raise ValueError(f"Unsupported reciprocal mesh boundary: {boundary!r}")
        if energy_partition not in {"potential", "uniform"}:
            raise ValueError(f"Unsupported reciprocal energy partition: {energy_partition!r}")
        if assignment not in {"cic", "tsc", "pcs"}:
            raise ValueError(f"Unsupported mesh assignment: {assignment!r}")
        self.mesh_size = int(mesh_size)
        self.boundary = str(boundary)
        self.slab_padding_factor = max(int(slab_padding_factor), 1)
        self.include_k0 = bool(include_k0)
        self.energy_partition = str(energy_partition)
        self.green_mode = str(green_mode)
        self.assignment = str(assignment)
        self.full_ewald = bool(full_ewald)
        self.k_norm_floor = float(k_norm_floor)
        self.ewald_alpha_prefactor = 5.0
        self.assignment_window_floor = 1.0e-6
        self.green_kernel = ReciprocalGreenKernel(
            green_mode=self.green_mode,
            hidden_dim=int(filter_hidden_dim),
            k_norm_floor=self.k_norm_floor,
        )
        assignment_offsets = _build_assignment_offsets(self.assignment)
        self.register_buffer("assignment_offsets", assignment_offsets, persistent=False)
        self.register_buffer("real_space_shift_index", self._init_real_space_shifts(), persistent=False)
        self.register_buffer("_assignment_window_cache", torch.empty(0), persistent=False)
        self.register_buffer("_cached_spectral_cell", torch.empty(0), persistent=False)
        self.register_buffer("_cached_spectral_weights", torch.empty(0), persistent=False)
        self.register_buffer("_cached_spectral_alpha", torch.empty(0), persistent=False)
        self.register_buffer("_cached_spectral_volume", torch.empty(0), persistent=False)
        self.register_buffer("_cached_spectral_real_cutoff", torch.empty(0), persistent=False)

    @property
    def num_k(self) -> int:
        total = self.mesh_size * self.mesh_size * self.mesh_size
        return total if self.include_k0 else max(total - 1, 0)

    def _periodic_axes(self) -> tuple[bool, bool, bool]:
        if self.boundary == "periodic":
            return True, True, True
        return True, True, False

    def _estimate_real_cutoff(self, cell: torch.Tensor) -> torch.Tensor:
        periodic_axes = torch.tensor(self._periodic_axes(), device=cell.device, dtype=torch.bool)
        periodic_vectors = cell[periodic_axes]
        periodic_lengths = torch.linalg.vector_norm(periodic_vectors, dim=-1)
        return 0.5 * periodic_lengths.min().clamp_min(self.k_norm_floor)

    def _estimate_ewald_alpha(self, real_cutoff: torch.Tensor) -> torch.Tensor:
        return real_cutoff.new_tensor(self.ewald_alpha_prefactor) / real_cutoff.clamp_min(self.k_norm_floor)

    def _init_real_space_shifts(self) -> torch.Tensor:
        ranges: list[torch.Tensor] = []
        for is_periodic in self._periodic_axes():
            values = [-1, 0, 1] if is_periodic else [0]
            ranges.append(torch.tensor(values, dtype=torch.long))
        return torch.cartesian_prod(*ranges)

    def _build_assignment_window(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cached = self._assignment_window_cache
        expected_numel = self.mesh_size * self.mesh_size * self.mesh_size
        if cached.numel() == expected_numel and cached.device == device and cached.dtype == dtype:
            return cached
        window_1d = _assignment_window_1d(
            mesh_size=self.mesh_size,
            assignment=self.assignment,
            device=device,
            dtype=dtype,
        )
        wx, wy, wz = torch.meshgrid(window_1d, window_1d, window_1d, indexing="ij")
        window = wx * wy * wz
        self._assignment_window_cache = window
        return window

    def build_k_norms(self, cell: torch.Tensor, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        effective_cell = _effective_cell_for_boundary(
            cell,
            boundary=self.boundary,
            slab_padding_factor=self.slab_padding_factor,
            dtype=dtype,
        )
        freq = _fft_integer_frequencies(self.mesh_size, device=cell.device, dtype=dtype)
        kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing="ij")
        integer_k = torch.stack([kx, ky, kz], dim=-1).reshape(-1, 3)
        inv_cell = torch.linalg.inv(effective_cell)
        # Physical reciprocal vector for integer index m is k = 2*pi * m @ inv(cell)^T (so that
        # k.r == 2*pi * m . frac). The transpose is required for O(3) equivariance on non-orthogonal
        # cells -- without it |k| is not rotation-invariant (matches _build_k_cart_flat and the C++
        # build_local_k_cart). On orthogonal cells inv(cell) is symmetric so this is a no-op.
        k_cart = 2.0 * math.pi * torch.matmul(integer_k, inv_cell.transpose(-1, -2))
        k_norms = torch.linalg.vector_norm(k_cart, dim=-1).reshape(self.mesh_size, self.mesh_size, self.mesh_size)
        volume = torch.abs(torch.linalg.det(effective_cell)).clamp_min(self.k_norm_floor)
        return k_norms, volume

    def _build_k_cart_flat(self, cell: torch.Tensor, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cartesian k-vectors (mesh^3, 3), their norms (mesh^3,), and the (effective) cell volume.

        Same lattice/convention as build_k_norms, but keeps the k vectors (needed for the multipole
        k.mu / k.Q.k terms)."""
        effective_cell = _effective_cell_for_boundary(
            cell, boundary=self.boundary, slab_padding_factor=self.slab_padding_factor, dtype=dtype
        )
        freq = _fft_integer_frequencies(self.mesh_size, device=cell.device, dtype=dtype)
        kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing="ij")
        integer_k = torch.stack([kx, ky, kz], dim=-1).reshape(-1, 3)
        inv_cell = torch.linalg.inv(effective_cell)
        # Physical reciprocal vector for integer index m is k = 2*pi * m @ inv(cell)^T, so that
        # k.r == 2*pi * m . frac (frac = pos @ inv(cell)) -- matches the spread/FFT phase. Using
        # inv(cell) (no transpose) only coincides for symmetric cells; a rotated cell breaks the
        # k.mu / k.Q.k equivariance otherwise.
        k_cart = 2.0 * math.pi * torch.matmul(integer_k, inv_cell.transpose(-1, -2))
        k_norms = torch.linalg.vector_norm(k_cart, dim=-1)
        volume = torch.abs(torch.linalg.det(effective_cell)).clamp_min(self.k_norm_floor)
        return k_cart, k_norms, volume

    def multipole_energy(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        source: torch.Tensor,
        dipole: torch.Tensor | None,
        quadrupole: torch.Tensor | None,
    ) -> torch.Tensor:
        """Reciprocal energy from latent monopole(+dipole)(+quadrupole) via the |S(k)|^2 PME route.

        S(k) = FFT(spread q) + i k . FFT(spread mu) - 1/2 k . FFT(spread Q) . k
        E    = (1/2V) sum_{k!=0} [green(k) / |W(k)|^2] |S(k)|^2          (W = assignment window)

        This route computes the energy directly in k-space (no iFFT), so it is free of the
        iFFT 1/N normalization that the potential route needs compensated, and it reduces to the
        plain reciprocal monopole sum when dipole/quadrupole are None. Per-graph, uniform partition.
        """
        atom_energy = source.new_zeros((source.size(0), 1))
        for graph_idx in range(cell.size(0)):
            node_index = torch.nonzero(batch == graph_idx, as_tuple=False).view(-1)
            n_local = int(node_index.numel())
            if n_local == 0:
                continue
            local_pos = pos.index_select(0, node_index)
            local_source = source.index_select(0, node_index)
            src_c = int(local_source.size(1))
            frac = _prepare_frac_for_boundary(
                local_pos, cell[graph_idx], boundary=self.boundary, slab_padding_factor=self.slab_padding_factor
            )
            k_cart, k_norms_flat, volume = self._build_k_cart_flat(cell[graph_idx], dtype=local_pos.dtype)
            green = self.green_kernel(k_norms_flat)
            window = self._build_assignment_window(device=local_pos.device, dtype=local_pos.dtype).reshape(-1)
            wdeconv = torch.reciprocal(window.clamp_min(self.assignment_window_floor).square())
            spectral = green / volume * wdeconv
            if self.full_ewald:
                # Ewald Gaussian screening exp(-k^2/4a^2): band-limits the reciprocal sum so
                # the coarse mesh can represent it -> accurate + (sub-grid) translation-stable.
                # (Mirrors the monopole _build_reciprocal_spectral_weights path; without it the
                # bare 4*pi/k^2 kernel has large CIC/mesh translation error.)
                real_cutoff = self._estimate_real_cutoff(cell[graph_idx])
                alpha = self._estimate_ewald_alpha(real_cutoff)
                spectral = spectral * torch.exp(-(k_norms_flat.square()) / (4.0 * alpha * alpha))
            spectral = torch.where(k_norms_flat > self.k_norm_floor, spectral, torch.zeros_like(spectral))

            def _spread_fft(field: torch.Tensor) -> torch.Tensor:
                mesh = _spread_source_to_mesh(
                    frac, field, mesh_size=self.mesh_size, assignment=self.assignment,
                    assignment_offsets=self.assignment_offsets, boundary=self.boundary,
                )
                return torch.fft.fftn(mesh, dim=(0, 1, 2)).reshape(-1, field.size(1))

            S = _spread_fft(local_source).reshape(-1, src_c)  # (K, src) complex
            if dipole is not None:
                mut = _spread_fft(dipole.index_select(0, node_index).reshape(n_local, src_c * 3)).reshape(-1, src_c, 3)
                S = S + 1j * torch.einsum("kx,ksx->ks", k_cart.to(mut.dtype), mut)
            if quadrupole is not None:
                qt = _spread_fft(quadrupole.index_select(0, node_index).reshape(n_local, src_c * 9)).reshape(-1, src_c, 3, 3)
                S = S - 0.5 * torch.einsum("kx,ksxy,ky->ks", k_cart.to(qt.dtype), qt, k_cart.to(qt.dtype))
            e_graph = 0.5 * (spectral.unsqueeze(-1) * S.abs().square()).sum()
            atom_energy.index_copy_(0, node_index, (e_graph / n_local).reshape(1, 1).expand(n_local, 1))
        return atom_energy

    def _can_use_spectral_cache(self, cell: torch.Tensor, *, dtype: torch.dtype) -> bool:
        if self.green_mode != "poisson" or self.training or torch.is_grad_enabled():
            return False
        cached_cell = self._cached_spectral_cell
        return (
            cached_cell.numel() == cell.numel()
            and cached_cell.device == cell.device
            and cached_cell.dtype == dtype
            and torch.equal(cached_cell, cell.to(dtype=dtype))
        )

    def _build_reciprocal_spectral_weights(
        self,
        cell: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        if self._can_use_spectral_cache(cell, dtype=dtype):
            alpha = self._cached_spectral_alpha if self.full_ewald and self._cached_spectral_alpha.numel() else None
            real_cutoff = (
                self._cached_spectral_real_cutoff
                if self.full_ewald and self._cached_spectral_real_cutoff.numel()
                else None
            )
            return self._cached_spectral_weights, alpha, self._cached_spectral_volume, real_cutoff

        real_cutoff = None
        alpha = None
        k_norms, volume = self.build_k_norms(cell, dtype=dtype)
        spectral_weights = self.green_kernel(k_norms) / volume
        if self.full_ewald:
            real_cutoff = self._estimate_real_cutoff(cell.to(dtype=dtype))
            alpha = self._estimate_ewald_alpha(real_cutoff)
            spectral_weights = spectral_weights * torch.exp(-(k_norms.square()) / (4.0 * alpha * alpha))
        if self.full_ewald or self.assignment != "cic":
            assignment_window = self._build_assignment_window(device=cell.device, dtype=dtype)
            assignment_scale = torch.reciprocal(assignment_window.clamp_min(self.assignment_window_floor).square())
            spectral_weights = spectral_weights * assignment_scale
        if self.full_ewald or (not self.include_k0):
            spectral_weights = torch.where(
                k_norms > self.k_norm_floor,
                spectral_weights,
                torch.zeros_like(spectral_weights),
            )
        if self.green_mode == "poisson" and (not self.training) and (not torch.is_grad_enabled()):
            self._cached_spectral_cell = cell.to(dtype=dtype).detach().clone()
            self._cached_spectral_weights = spectral_weights.detach().clone()
            self._cached_spectral_volume = volume.detach().clone()
            if alpha is None:
                self._cached_spectral_alpha = volume.new_empty((0,))
                self._cached_spectral_real_cutoff = volume.new_empty((0,))
            else:
                self._cached_spectral_alpha = alpha.detach().clone().reshape(())
                self._cached_spectral_real_cutoff = real_cutoff.detach().clone().reshape(())
        return spectral_weights, alpha, volume, real_cutoff

    def apply_green_kernel(
        self,
        mesh: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        mesh_dtype = mesh.dtype
        mesh_complex = torch.fft.fftn(mesh, dim=(0, 1, 2))
        spectral_weights, alpha, volume, real_cutoff = self._build_reciprocal_spectral_weights(cell, dtype=mesh_dtype)
        filtered = torch.fft.ifftn(mesh_complex * spectral_weights.unsqueeze(-1), dim=(0, 1, 2))
        # Compensate torch.fft.ifftn's 1/N normalization (N = mesh_size**3). The forward FFT of
        # the spread charges already yields the structure factor S(k), so the iFFT's 1/N factor
        # is spurious here -- without it the reciprocal potential/energy comes out mesh_size**3
        # too small (verified exactly: E_mesh * mesh_size**3 == the analytic bare reciprocal sum,
        # and full_ewald reproduces the NaCl Madelung constant to ~0.1% only with this factor).
        filtered = filtered * (float(self.mesh_size) ** 3)
        return filtered.real, alpha, volume, real_cutoff

    def _compute_real_space_potential(
        self,
        pos: torch.Tensor,
        source: torch.Tensor,
        cell: torch.Tensor,
        *,
        alpha: torch.Tensor,
        real_cutoff: torch.Tensor,
    ) -> torch.Tensor:
        if pos.size(0) == 0:
            return source.new_zeros(source.shape)
        shift_index = self.real_space_shift_index.to(device=cell.device)
        shift_cart = torch.matmul(shift_index.to(dtype=pos.dtype), cell.to(dtype=pos.dtype))
        disp = pos.unsqueeze(1).unsqueeze(2) - pos.unsqueeze(0).unsqueeze(2) - shift_cart.unsqueeze(0).unsqueeze(0)
        distance = torch.linalg.vector_norm(disp, dim=-1)
        kernel = torch.special.erfc(alpha * distance) / distance.clamp_min(self.k_norm_floor)
        valid = distance <= real_cutoff
        zero_shift = (shift_index == 0).all(dim=1)
        self_mask = torch.eye(pos.size(0), device=pos.device, dtype=torch.bool).unsqueeze(-1) & zero_shift.view(1, 1, -1)
        kernel = kernel * (valid & (~self_mask)).to(dtype=source.dtype)
        pair_kernel = kernel.sum(dim=-1)
        return torch.matmul(pair_kernel, source)

    def forward(self, pos: torch.Tensor, batch: torch.Tensor, cell: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        atom_energy = source.new_zeros((source.size(0), 1))
        counts = torch.bincount(batch, minlength=cell.size(0)).to(dtype=source.dtype).clamp_min(1.0)
        for graph_idx in range(cell.size(0)):
            node_index = torch.nonzero(batch == graph_idx, as_tuple=False).view(-1)
            if node_index.numel() == 0:
                continue
            local_pos = pos.index_select(0, node_index)
            local_source = source.index_select(0, node_index)
            local_frac = _prepare_frac_for_boundary(
                local_pos,
                cell[graph_idx],
                boundary=self.boundary,
                slab_padding_factor=self.slab_padding_factor,
            )
            mesh = _spread_source_to_mesh(
                local_frac,
                local_source,
                mesh_size=self.mesh_size,
                assignment=self.assignment,
                assignment_offsets=self.assignment_offsets,
                boundary=self.boundary,
            )
            potential_mesh, alpha, effective_volume, real_cutoff = self.apply_green_kernel(mesh, cell[graph_idx])
            reciprocal_potential = _gather_source_from_mesh(
                local_frac,
                potential_mesh,
                mesh_size=self.mesh_size,
                assignment=self.assignment,
                assignment_offsets=self.assignment_offsets,
                boundary=self.boundary,
            )
            total_potential = reciprocal_potential
            if self.full_ewald:
                assert alpha is not None and real_cutoff is not None
                real_space_potential = self._compute_real_space_potential(
                    local_pos,
                    local_source,
                    cell[graph_idx],
                    alpha=alpha.to(dtype=local_pos.dtype),
                    real_cutoff=real_cutoff.to(dtype=local_pos.dtype),
                )
                self_potential = (
                    -2.0 * alpha.to(dtype=local_source.dtype) / math.sqrt(math.pi)
                ) * local_source
                net_source = local_source.sum(dim=0, keepdim=True)
                background_potential = (
                    -math.pi
                    * net_source
                    / (alpha.to(dtype=local_source.dtype).square() * effective_volume.to(dtype=local_source.dtype))
                )
                total_potential = reciprocal_potential + real_space_potential + self_potential + background_potential
            atom_energy_local = 0.5 * (local_source * total_potential).sum(dim=-1, keepdim=True)
            if self.energy_partition == "uniform":
                graph_total = atom_energy_local.sum()
                atom_energy_local = graph_total.expand_as(atom_energy_local) / counts[graph_idx]
            atom_energy.index_copy_(0, node_index, atom_energy_local)
        return atom_energy


class LatentReciprocalLongRange(nn.Module):
    """Periodic reciprocal-space prototype closer to a learnable Green's function."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        *,
        source_channels: int = 1,
        boundary: str = "periodic",
        neutralize: bool = True,
        kmax: int = 2,
        mesh_size: int = 16,
        filter_hidden_dim: int = 64,
        slab_padding_factor: int = 2,
        include_k0: bool = False,
        reciprocal_backend: str = "direct_kspace",
        energy_partition: str = "potential",
        green_mode: str = "poisson",
        assignment: str = "cic",
        mesh_fft_full_ewald: bool = False,
    ):
        super().__init__()
        if reciprocal_backend not in {"direct_kspace", "mesh_fft"}:
            raise ValueError(f"Unsupported reciprocal backend: {reciprocal_backend!r}")
        if reciprocal_backend == "direct_kspace" and boundary != "periodic":
            raise ValueError("direct_kspace reciprocal backend currently requires boundary='periodic'")
        if boundary not in {"periodic", "slab"}:
            raise ValueError(f"Unsupported long-range boundary mode: {boundary!r}")
        self.source_channels = int(source_channels)
        self.boundary = str(boundary)
        self.neutralize = bool(neutralize)
        self.kmax = int(kmax)
        self.mesh_size = int(mesh_size)
        self.slab_padding_factor = max(int(slab_padding_factor), 1)
        self.include_k0 = bool(include_k0)
        self.reciprocal_backend = str(reciprocal_backend)
        self.energy_partition = str(energy_partition)
        self.green_mode = str(green_mode)
        self.assignment = str(assignment)
        self.mesh_fft_full_ewald = bool(mesh_fft_full_ewald)
        self.source_kind = "latent_charge"
        self.source_layout = "channels_last"
        self.runtime_backend = "mesh_fft" if reciprocal_backend == "mesh_fft" else "none"
        self.source_head = LatentSourceHead(feature_dim, hidden_dim, source_channels=self.source_channels)
        if self.reciprocal_backend == "mesh_fft":
            self.kernel = MeshLongRangeKernel3D(
                mesh_size=self.mesh_size,
                filter_hidden_dim=int(filter_hidden_dim),
                boundary=self.boundary,
                slab_padding_factor=self.slab_padding_factor,
                include_k0=self.include_k0,
                energy_partition=self.energy_partition,
                green_mode=self.green_mode,
                assignment=self.assignment,
                full_ewald=self.mesh_fft_full_ewald,
            )
            self.exports_reciprocal_source = True
            final_linear = self.source_head.net[-1]
            if isinstance(final_linear, nn.Linear):
                nn.init.zeros_(final_linear.weight)
                nn.init.zeros_(final_linear.bias)
            self.energy_scale = None
        else:
            self.kernel = ReciprocalSpectralKernel3D(
                kmax=self.kmax,
                filter_hidden_dim=int(filter_hidden_dim),
                include_k0=self.include_k0,
                reciprocal_backend=self.reciprocal_backend,
                energy_partition=self.energy_partition,
            )
            self.exports_reciprocal_source = False
            # Keep the default contribution near zero so the module can be enabled
            # in existing workflows without destabilizing outputs before training.
            self.energy_scale = nn.Parameter(torch.tensor(0.0))

    @property
    def num_k(self) -> int:
        return self.kernel.num_k

    def _neutralize_source(self, source: torch.Tensor, batch: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        if not self.neutralize:
            return source
        graph_ids = torch.arange(cell.size(0), device=batch.device, dtype=batch.dtype)
        graph_mask = (batch.unsqueeze(1) == graph_ids.unsqueeze(0)).to(dtype=source.dtype)
        counts = graph_mask.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        graph_mean = torch.einsum("nb,nc->bc", graph_mask, source) / counts
        return source - graph_mean.index_select(0, batch)

    def forward(
        self,
        invariant_features: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        *,
        edge_src: torch.Tensor | None = None,
        edge_dst: torch.Tensor | None = None,
        return_source: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        source = self.source_head(invariant_features)
        source = self._neutralize_source(source, batch, cell)
        atom_energy = self.kernel(pos, batch, cell, source)
        if self.energy_scale is not None:
            atom_energy = self.energy_scale * atom_energy
        if return_source:
            return atom_energy, source
        return atom_energy

    def forward_multipole(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        monopole: torch.Tensor,
        dipole: torch.Tensor | None = None,
        quadrupole: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Long-range energy from pre-computed equivariant Cartesian multipole sources.

        Routes through the mesh-FFT kernel's ``multipole_energy``
        (S(k) = q + i k.mu - 1/2 k.Q.k). Only the monopole is charge-neutralized
        per graph; dipole/quadrupole carry no net-charge constraint.
        """
        if not hasattr(self.kernel, "multipole_energy"):
            raise ValueError(
                "multipole long-range requires reciprocal_backend='mesh_fft' "
                "(only the mesh-FFT kernel exposes multipole_energy)"
            )
        source = self._neutralize_source(monopole, batch, cell)
        atom_energy = self.kernel.multipole_energy(pos, batch, cell, source, dipole, quadrupole)
        if self.energy_scale is not None:
            atom_energy = self.energy_scale * atom_energy
        return atom_energy


def build_feature_spectral_module(
    *,
    mode: str,
    feature_dim: int,
    bottleneck_dim: int = 8,
    mesh_size: int = 16,
    filter_hidden_dim: int = 64,
    boundary: str = "periodic",
    slab_padding_factor: int = 2,
    neutralize: bool = True,
    include_k0: bool = False,
    assignment: str = "cic",
    gate_init: float = 0.0,
) -> nn.Module | None:
    if mode == "none":
        return None
    if mode == "fft":
        return FeatureSpectralResidualBlock(
            feature_dim=feature_dim,
            bottleneck_dim=bottleneck_dim,
            mesh_size=mesh_size,
            filter_hidden_dim=filter_hidden_dim,
            boundary=boundary,
            slab_padding_factor=slab_padding_factor,
            neutralize=neutralize,
            include_k0=include_k0,
            assignment=assignment,
            gate_init=gate_init,
        )
    raise ValueError(f"Unsupported feature_spectral_mode: {mode!r}")


def build_long_range_module(
    *,
    mode: str,
    feature_dim: int,
    hidden_dim: int = 64,
    boundary: str = "nonperiodic",
    neutralize: bool = True,
    filter_hidden_dim: int = 64,
    kmax: int = 2,
    mesh_size: int = 16,
    slab_padding_factor: int = 2,
    include_k0: bool = False,
    source_channels: int = 1,
    backend: str = "dense_pairwise",
    reciprocal_backend: str = "direct_kspace",
    energy_partition: str = "potential",
    green_mode: str = "poisson",
    assignment: str = "cic",
    mesh_fft_full_ewald: bool = False,
    theta: float = 0.5,
    leaf_size: int = 32,
    multipole_order: int = 0,
    far_source_dim: int = 16,
    far_num_shells: int = 3,
    far_shell_growth: float = 2.0,
    far_tail: bool = True,
    far_tail_bins: int = 2,
    far_stats: str = "mean,count,mean_r,rms_r",
    far_max_radius_multiplier: float | None = None,
    far_source_norm: bool = True,
    far_gate_init: float = 0.0,
    cutoff_radius: float = 5.0,
    max_multipole_l: int = 0,
    multipole_feature_channels: int = 0,
) -> nn.Module | None:
    if mode == "none":
        return None
    if mode == "reciprocal-spectral-v1":
        return LatentReciprocalLongRange(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            source_channels=source_channels,
            boundary=boundary,
            neutralize=neutralize,
            kmax=kmax,
            mesh_size=mesh_size,
            filter_hidden_dim=filter_hidden_dim,
            slab_padding_factor=slab_padding_factor,
            include_k0=include_k0,
            reciprocal_backend=reciprocal_backend,
            energy_partition=energy_partition,
            green_mode=green_mode,
            assignment=assignment,
            mesh_fft_full_ewald=mesh_fft_full_ewald,
        )
    raise ValueError(f"Unsupported long_range_mode: {mode!r}")


def configure_long_range_modules(
    owner: nn.Module,
    *,
    feature_dim: int,
    cutoff_radius: float,
    long_range_mode: str = "none",
    long_range_hidden_dim: int = 64,
    long_range_boundary: str = "nonperiodic",
    long_range_neutralize: bool = True,
    long_range_filter_hidden_dim: int = 64,
    long_range_kmax: int = 2,
    long_range_mesh_size: int = 16,
    long_range_slab_padding_factor: int = 2,
    long_range_include_k0: bool = False,
    long_range_source_channels: int = 1,
    long_range_backend: str = "dense_pairwise",
    long_range_reciprocal_backend: str = "direct_kspace",
    long_range_energy_partition: str = "potential",
    long_range_green_mode: str = "poisson",
    long_range_assignment: str = "cic",
    long_range_mesh_fft_full_ewald: bool = False,
    long_range_theta: float = 0.5,
    long_range_leaf_size: int = 32,
    long_range_multipole_order: int = 0,
    long_range_far_source_dim: int = 16,
    long_range_far_num_shells: int = 3,
    long_range_far_shell_growth: float = 2.0,
    long_range_far_tail: bool = True,
    long_range_far_tail_bins: int = 2,
    long_range_far_stats: str = "mean,count,mean_r,rms_r",
    long_range_far_max_radius_multiplier: float | None = None,
    long_range_far_source_norm: bool = True,
    long_range_far_gate_init: float = 0.0,
    feature_spectral_mode: str = "none",
    feature_spectral_bottleneck_dim: int = 8,
    feature_spectral_mesh_size: int = 16,
    feature_spectral_filter_hidden_dim: int = 64,
    feature_spectral_boundary: str = "periodic",
    feature_spectral_slab_padding_factor: int = 2,
    feature_spectral_neutralize: bool = True,
    feature_spectral_include_k0: bool = False,
    feature_spectral_assignment: str = "cic",
    feature_spectral_gate_init: float = 0.0,
) -> None:
    owner.long_range_mode = str(long_range_mode)
    owner.long_range_hidden_dim = int(long_range_hidden_dim)
    owner.long_range_boundary = str(long_range_boundary)
    owner.long_range_neutralize = bool(long_range_neutralize)
    owner.long_range_filter_hidden_dim = int(long_range_filter_hidden_dim)
    owner.long_range_kmax = int(long_range_kmax)
    owner.long_range_mesh_size = int(long_range_mesh_size)
    owner.long_range_slab_padding_factor = int(long_range_slab_padding_factor)
    owner.long_range_include_k0 = bool(long_range_include_k0)
    owner.long_range_source_channels = int(long_range_source_channels)
    owner.long_range_backend = str(long_range_backend)
    owner.long_range_reciprocal_backend = str(long_range_reciprocal_backend)
    owner.long_range_energy_partition = str(long_range_energy_partition)
    owner.long_range_green_mode = str(long_range_green_mode)
    owner.long_range_assignment = str(long_range_assignment)
    owner.long_range_mesh_fft_full_ewald = bool(long_range_mesh_fft_full_ewald)
    owner.long_range_theta = float(long_range_theta)
    owner.long_range_leaf_size = int(long_range_leaf_size)
    owner.long_range_multipole_order = int(long_range_multipole_order)
    owner.long_range_far_source_dim = int(long_range_far_source_dim)
    owner.long_range_far_num_shells = int(long_range_far_num_shells)
    owner.long_range_far_shell_growth = float(long_range_far_shell_growth)
    owner.long_range_far_tail = bool(long_range_far_tail)
    owner.long_range_far_tail_bins = int(long_range_far_tail_bins)
    owner.long_range_far_stats = str(long_range_far_stats)
    owner.long_range_far_max_radius_multiplier = (
        None if long_range_far_max_radius_multiplier is None else float(long_range_far_max_radius_multiplier)
    )
    owner.long_range_far_source_norm = bool(long_range_far_source_norm)
    owner.long_range_far_gate_init = float(long_range_far_gate_init)
    owner.feature_spectral_mode = str(feature_spectral_mode)
    owner.feature_spectral_bottleneck_dim = int(feature_spectral_bottleneck_dim)
    owner.feature_spectral_mesh_size = int(feature_spectral_mesh_size)
    owner.feature_spectral_filter_hidden_dim = int(feature_spectral_filter_hidden_dim)
    owner.feature_spectral_boundary = str(feature_spectral_boundary)
    owner.feature_spectral_slab_padding_factor = int(feature_spectral_slab_padding_factor)
    owner.feature_spectral_neutralize = bool(feature_spectral_neutralize)
    owner.feature_spectral_include_k0 = bool(feature_spectral_include_k0)
    owner.feature_spectral_assignment = str(feature_spectral_assignment)
    owner.feature_spectral_gate_init = float(feature_spectral_gate_init)

    owner.long_range_module = build_long_range_module(
        mode=owner.long_range_mode,
        feature_dim=int(feature_dim),
        hidden_dim=owner.long_range_hidden_dim,
        boundary=owner.long_range_boundary,
        neutralize=owner.long_range_neutralize,
        filter_hidden_dim=owner.long_range_filter_hidden_dim,
        kmax=owner.long_range_kmax,
        mesh_size=owner.long_range_mesh_size,
        slab_padding_factor=owner.long_range_slab_padding_factor,
        include_k0=owner.long_range_include_k0,
        source_channels=owner.long_range_source_channels,
        backend=owner.long_range_backend,
        reciprocal_backend=owner.long_range_reciprocal_backend,
        energy_partition=owner.long_range_energy_partition,
        green_mode=owner.long_range_green_mode,
        assignment=owner.long_range_assignment,
        mesh_fft_full_ewald=owner.long_range_mesh_fft_full_ewald,
        theta=owner.long_range_theta,
        leaf_size=owner.long_range_leaf_size,
        multipole_order=owner.long_range_multipole_order,
        far_source_dim=owner.long_range_far_source_dim,
        far_num_shells=owner.long_range_far_num_shells,
        far_shell_growth=owner.long_range_far_shell_growth,
        far_tail=owner.long_range_far_tail,
        far_tail_bins=owner.long_range_far_tail_bins,
        far_stats=owner.long_range_far_stats,
        far_max_radius_multiplier=owner.long_range_far_max_radius_multiplier,
        far_source_norm=owner.long_range_far_source_norm,
        far_gate_init=owner.long_range_far_gate_init,
        cutoff_radius=float(cutoff_radius),
    )
    owner.long_range_num_k = (
        getattr(owner.long_range_module, "num_k", None) if owner.long_range_module is not None else None
    )
    owner.feature_spectral_module = build_feature_spectral_module(
        mode=owner.feature_spectral_mode,
        feature_dim=int(feature_dim),
        bottleneck_dim=owner.feature_spectral_bottleneck_dim,
        mesh_size=owner.feature_spectral_mesh_size,
        filter_hidden_dim=owner.feature_spectral_filter_hidden_dim,
        boundary=owner.feature_spectral_boundary,
        slab_padding_factor=owner.feature_spectral_slab_padding_factor,
        neutralize=owner.feature_spectral_neutralize,
        include_k0=owner.feature_spectral_include_k0,
        assignment=owner.feature_spectral_assignment,
        gate_init=owner.feature_spectral_gate_init,
    )
    owner.long_range_runtime_backend = "none"
    owner.long_range_runtime_source_kind = "none"
    owner.long_range_runtime_source_channels = 0
    owner.long_range_runtime_source_layout = "none"
    owner.long_range_runtime_source_boundary = owner.long_range_boundary
    owner.long_range_runtime_source_slab_padding_factor = owner.long_range_slab_padding_factor
    if owner.long_range_module is not None and bool(getattr(owner.long_range_module, "exports_reciprocal_source", False)):
        owner.long_range_runtime_backend = str(getattr(owner.long_range_module, "runtime_backend", "none"))
        owner.long_range_runtime_source_kind = str(getattr(owner.long_range_module, "source_kind", "latent_charge"))
        owner.long_range_runtime_source_channels = int(
            getattr(owner.long_range_module, "source_channels", owner.long_range_source_channels)
        )
        owner.long_range_runtime_source_layout = str(getattr(owner.long_range_module, "source_layout", "channels_last"))
        owner.long_range_runtime_source_boundary = owner.long_range_boundary
        owner.long_range_runtime_source_slab_padding_factor = owner.long_range_slab_padding_factor
        owner.reciprocal_source_channels = owner.long_range_runtime_source_channels
        owner.reciprocal_source_boundary = owner.long_range_runtime_source_boundary
        owner.reciprocal_source_slab_padding_factor = owner.long_range_runtime_source_slab_padding_factor
    elif owner.feature_spectral_module is not None:
        owner.long_range_runtime_backend = "mesh_fft"
        owner.long_range_runtime_source_kind = "feature_bottleneck"
        owner.long_range_runtime_source_channels = owner.feature_spectral_bottleneck_dim
        owner.long_range_runtime_source_layout = "channels_last"
        owner.long_range_runtime_source_boundary = owner.feature_spectral_boundary
        owner.long_range_runtime_source_slab_padding_factor = owner.feature_spectral_slab_padding_factor
        owner.reciprocal_source_channels = owner.feature_spectral_bottleneck_dim
        owner.reciprocal_source_boundary = owner.feature_spectral_boundary
        owner.reciprocal_source_slab_padding_factor = owner.feature_spectral_slab_padding_factor
    else:
        owner.reciprocal_source_channels = 0
        owner.reciprocal_source_boundary = "periodic"
        owner.reciprocal_source_slab_padding_factor = 2


def apply_long_range_modules(
    owner: nn.Module,
    invariant_features: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    cell: torch.Tensor,
    *,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    return_reciprocal_source: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]:
    feature_reciprocal_source = None
    if getattr(owner, "feature_spectral_module", None) is not None:
        invariant_features, feature_reciprocal_source = owner.feature_spectral_module(
            invariant_features,
            pos,
            batch,
            cell,
        )

    long_range_energy = None
    reciprocal_source = None
    defer_long_range_to_runtime = False
    if getattr(owner, "long_range_module", None) is not None:
        if return_reciprocal_source and bool(getattr(owner.long_range_module, "exports_reciprocal_source", False)):
            long_range_energy, reciprocal_source = owner.long_range_module(
                invariant_features,
                pos,
                batch,
                cell,
                edge_src=edge_src,
                edge_dst=edge_dst,
                return_source=True,
            )
            defer_long_range_to_runtime = reciprocal_source.numel() > 0
        else:
            long_range_energy = owner.long_range_module(
                invariant_features,
                pos,
                batch,
                cell,
                edge_src=edge_src,
                edge_dst=edge_dst,
            )
    if reciprocal_source is None:
        reciprocal_source = feature_reciprocal_source
    return invariant_features, long_range_energy, reciprocal_source, defer_long_range_to_runtime
