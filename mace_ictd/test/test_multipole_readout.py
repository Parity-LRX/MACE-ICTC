"""Round-trip + rotation-equivariance test for the multipole readout head.

Strategy: build known Cartesian multipoles (charge / dipole / quadrupole), encode
them into ICTD irreps blocks with the existing embedder, assemble a full-SO(3)
feature, run ``MultipoleReadout`` (channel mixers pinned to identity), and check
(a) round-trip recovery and (b) that under a Cartesian rotation R the outputs
transform as monopole->invariant, dipole->R mu, quadrupole->R Q R^T.
"""

from __future__ import annotations

import torch

from mace_ictd.models.multipole_readout import MultipoleReadout
from mace_ictd.utils.tensor_utils import build_physical_tensor_label_blocks


def _random_rotation(dtype) -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    a = torch.randn(3, 3, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))  # fix QR sign ambiguity
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(dtype)


def _blocks(tensor, rank, dtype):
    return build_physical_tensor_label_blocks(
        tensor, rank=rank, lmax=rank, include_trace_chain=False,
        representation="ictd", device=torch.device("cpu"),
    )


def _assemble(charge, mu, quad, *, lmax, dtype):
    """charge [N], mu [N,3], quad [N,3,3] -> full-SO(3) feature [N, (lmax+1)**2] (C=1)."""
    n = charge.shape[0]
    b1 = _blocks(mu, 1, dtype)[1].reshape(n, 1, 3)
    b2 = _blocks(quad, 2, dtype)[2].reshape(n, 1, 5)
    b0 = charge.reshape(n, 1, 1)
    blocks = {0: b0, 1: b1, 2: b2}
    return torch.cat([blocks[l].reshape(n, 1 * (2 * l + 1)) for l in range(lmax + 1)], dim=-1)


def _pin_identity(head: MultipoleReadout) -> None:
    with torch.no_grad():
        for lin in head.mix:
            lin.weight.copy_(torch.ones_like(lin.weight))


def test_multipole_readout_roundtrip_and_equivariance():
    dtype = torch.float64
    torch.manual_seed(0)
    n = 6
    charge = torch.randn(n, dtype=dtype)
    mu = torch.randn(n, 3, dtype=dtype)
    qsym = torch.randn(n, 3, 3, dtype=dtype)
    qsym = 0.5 * (qsym + qsym.transpose(-1, -2))
    eye = torch.eye(3, dtype=dtype)
    quad = qsym - (qsym.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0).reshape(n, 1, 1) * eye  # traceless

    head = MultipoleReadout(channels=1, lmax=2, max_multipole_l=2, source_channels=1).to(dtype)
    _pin_identity(head)

    feat = _assemble(charge, mu, quad, lmax=2, dtype=dtype)
    m, d, q = head(feat)
    m, d, q = m.reshape(n), d.reshape(n, 3), q.reshape(n, 3, 3)

    # (a) round-trip: decode(encode(.)) == identity (convention-correct)
    assert torch.allclose(m, charge, atol=1e-9), (m - charge).abs().max()
    assert torch.allclose(d, mu, atol=1e-9), (d - mu).abs().max()
    assert torch.allclose(q, quad, atol=1e-9), (q - quad).abs().max()

    # (b) equivariance under a Cartesian rotation R
    R = _random_rotation(dtype)
    mu_r = mu @ R.T
    quad_r = torch.einsum("ij,njk,lk->nil", R, quad, R)
    feat_r = _assemble(charge, mu_r, quad_r, lmax=2, dtype=dtype)
    m2, d2, q2 = head(feat_r)
    m2, d2, q2 = m2.reshape(n), d2.reshape(n, 3), q2.reshape(n, 3, 3)

    assert torch.allclose(m2, charge, atol=1e-9), "monopole not invariant"
    assert torch.allclose(d2, mu_r, atol=1e-9), (d2 - mu_r).abs().max()
    assert torch.allclose(q2, quad_r, atol=1e-9), (q2 - quad_r).abs().max()


if __name__ == "__main__":
    test_multipole_readout_roundtrip_and_equivariance()
    print("OK: multipole readout round-trip + equivariance")
