"""Task 2: the Ewald dipole-dipole field operator (T.mu via the shared ReciprocalBackend).
Correctness = alpha-independence (the SR+LR+self split is exact) + the isolated-pair limit
reproducing the analytic point-dipole tensor (3 rr - I)/r^3."""
from __future__ import annotations

import torch

from mace_ictd.models.dispersion import dispersion_neighbor_list
from mace_ictd.models.mbd import dipole_field
from mace_ictd.models.reciprocal_backend import ReciprocalBackend


def _field(backend, pos, mu, cell, alpha, cutoff):
    batch = torch.zeros(pos.size(0), dtype=torch.long)
    src, dst, sh = dispersion_neighbor_list(pos, batch, cell.reshape(1, 3, 3), cutoff, pbc=True)
    return dipole_field(backend, pos, mu, cell, alpha=alpha, src=src, dst=dst, shifts=sh)


def test_alpha_independence():
    """The total periodic Ewald dipole field must not depend on the splitting parameter alpha."""
    torch.set_default_dtype(torch.float64)
    be = ReciprocalBackend(mesh_size=48, assignment="pcs", boundary="periodic")
    N, L = 6, 8.0
    g = torch.Generator().manual_seed(0)
    pos = torch.rand(N, 3, generator=g, dtype=torch.float64) * L
    mu = torch.randn(N, 3, generator=g, dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64) * L
    cutoff = 0.5 * L - 1e-6  # < L: no self-image pairs; self handled by the reciprocal self term

    e1 = _field(be, pos, mu, cell, alpha=0.9, cutoff=cutoff)
    e2 = _field(be, pos, mu, cell, alpha=1.4, cutoff=cutoff)
    d = (e1 - e2).abs().max().item()
    scale = e1.abs().max().item()
    # converges cleanly with mesh (2.5e-2@24 -> 1.1e-3@48 -> 3.5e-4@64): an exact split, not a bug
    assert d / scale < 3e-3, f"alpha-dependent (split wrong): rel {d/scale:.2e}"


def test_isolated_pair_matches_point_dipole():
    """Huge box, one pair: E_i ~= T_full(r) mu_j with T_ab=(3 r^_a r^_b - d_ab)/r^3 (LR,self -> 0)."""
    torch.set_default_dtype(torch.float64)
    be = ReciprocalBackend(mesh_size=32, assignment="pcs", boundary="periodic")
    L = 60.0
    pos = torch.tensor([[0.0, 0, 0], [3.2, 0.7, -0.4]], dtype=torch.float64) + L / 2
    mu = torch.tensor([[0.3, -0.5, 0.8], [0.0, 0.0, 0.0]], dtype=torch.float64)  # only mu_1 nonzero
    cell = torch.eye(3, dtype=torch.float64) * L
    e = _field(be, pos, mu, cell, alpha=0.06, cutoff=12.0)

    rvec = pos[1] - pos[0]
    r = rvec.norm()
    rhat = rvec / r
    T = (3.0 * torch.outer(rhat, rhat) - torch.eye(3, dtype=torch.float64)) / r ** 3
    e2_analytic = T @ mu[0]
    d = (e[1] - e2_analytic).abs().max().item()
    assert d / e2_analytic.abs().max().item() < 5e-3, (
        f"field at 2 from dipole 1 wrong: got {e[1].tolist()} vs {e2_analytic.tolist()}"
    )


if __name__ == "__main__":
    test_alpha_independence()
    print("OK: dipole-field Ewald is alpha-independent (SR+LR+self split exact)")
    test_isolated_pair_matches_point_dipole()
    print("OK: isolated-pair field == analytic (3 rr - I)/r^3 point-dipole tensor")
