"""Many-body dispersion (MBD) building blocks: the dipole-dipole field operator via the shared
ReciprocalBackend (Ewald split), and -- later -- the coupled-dipole matvec + SLQ spectral solver.

The dipole field E_i = sum_j T_ij . mu_j with T_ab = d_a d_b (1/r) (the rank-2 dipole-dipole tensor)
is split Ewald-style into:

    T = T_SR  (real-space, erfc-damped, over a neighbour list)
      + T_LR  (reciprocal, PME: spread mu -> FFT -> x (-4pi/V k_a k_b/k^2 e^{-k^2/4a^2}) -> iFFT -> gather)
      - T_self (the r=0 reciprocal self term, +4a^3/(3 sqrt pi) mu_i)
    tinfoil boundary: drop k=0.

This is the MBD-specific TENSOR kernel that rides on the SAME grid backend as the scalar electrostatic
PME. The total field is exactly alpha-independent (the split is exact), which is the correctness test.
"""

from __future__ import annotations

import math

import torch

from mace_ictd.models.reciprocal_backend import ReciprocalBackend

_SQRT_PI = math.sqrt(math.pi)


def ewald_b_functions(r: torch.Tensor, alpha: float, *, floor: float = 1.0e-12):
    """B_0,B_1,B_2 for the erfc-damped 1/r: B_0=erfc(ar)/r, B_{n+1}=[(2n+1)B_n+(2a^2)^{n+1}/(a*sqrtpi) e^{-a^2 r^2}]/r^2."""
    r = r.clamp_min(floor)
    r2 = r * r
    gauss = torch.exp(-(alpha * alpha) * r2)
    b0 = torch.erfc(alpha * r) / r
    b1 = (b0 + (2.0 * alpha * alpha) / (alpha * _SQRT_PI) * gauss) / r2
    b2 = (3.0 * b1 + ((2.0 * alpha * alpha) ** 2) / (alpha * _SQRT_PI) * gauss) / r2
    return b0, b1, b2


def dipole_field(
    backend: ReciprocalBackend,
    pos: torch.Tensor,
    mu: torch.Tensor,
    cell: torch.Tensor,
    *,
    alpha: float,
    src: torch.Tensor,
    dst: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Periodic Ewald dipole field E_i = sum_j T_ij mu_j (tinfoil, k=0 dropped).

    pos [N,3], mu [N,3], cell [3,3]; (src,dst,shifts) a real-space neighbour list (dst<-src with
    integer cell shift) covering the erfc range. Returns E [N,3].
    """
    N = pos.size(0)
    dtype = pos.dtype
    a3 = alpha ** 3

    # --- reciprocal T_LR . mu  (PME, 3-channel) ---
    frac = backend.frac(pos, cell)
    k_cart, k_norm, volume = backend.k_grid(cell, dtype=dtype)            # [K,3],[K],scalar
    mu_k = backend.fftn(backend.spread(frac, mu)).reshape(-1, 3)          # [K,3] complex
    k_c = k_cart.to(mu_k.dtype)
    kdotmu = (k_c * mu_k).sum(-1)                                         # [K] complex  (k . mu(k))
    screen = torch.exp(-(k_norm.square()) / (4.0 * alpha * alpha))
    wdeconv = backend.assignment_window(device=pos.device, dtype=dtype)
    scale = -(4.0 * math.pi) / volume * screen * wdeconv / k_norm.square()  # T~ = -4pi/V k k /k^2 ...
    scale = torch.where(k_norm > backend.k_norm_floor, scale, torch.zeros_like(scale))
    e_k = (scale.to(mu_k.dtype).unsqueeze(-1) * k_c) * kdotmu.unsqueeze(-1)  # [K,3] = scale k_a (k.mu)
    m = backend.mesh_size
    e_mesh = backend.ifftn(e_k.reshape(m, m, m, 3)).real * (float(m) ** 3)
    field = backend.gather(frac, e_mesh)                                  # [N,3]

    # --- self term: reciprocal included r=0; subtract T_LR_self = -(4a^3/3sqrtpi) I -> +... mu ---
    field = field + (4.0 * a3 / (3.0 * _SQRT_PI)) * mu

    # --- real-space T_SR . mu over the neighbour list (T_ab = -B1 d_ab + B2 r_a r_b) ---
    if src.numel() > 0:
        shift_cart = shifts.to(dtype) @ cell.to(dtype)
        rvec = pos.index_select(0, dst) - pos.index_select(0, src) + shift_cart   # [E,3]  (i<-j)
        r = torch.linalg.vector_norm(rvec, dim=-1)
        b0, b1, b2 = ewald_b_functions(r, alpha)
        mu_src = mu.index_select(0, src)                                  # mu_j  [E,3]
        rdotmu = (rvec * mu_src).sum(-1)                                  # [E]
        contrib = -b1.unsqueeze(-1) * mu_src + b2.unsqueeze(-1) * rvec * rdotmu.unsqueeze(-1)  # T_SR mu_j
        field = field.index_add(0, dst, contrib.to(dtype))
    return field
