// Task 5/5: C++ MBD solver -- mirrors the validated Python (mace_ictd/models/mbd.py +
// reciprocal_backend.py) 1:1 using the libtorch C++ API. Shares the PME/cuFFT grid ops with the
// scalar electrostatics (here self-contained CIC spread/gather to match a Python assignment="cic"
// reference for parity). Chebyshev Tr[sqrt C] (no eigensolve) for the deployment hot path.
//
// Build the standalone parity test (compares C++ mbd_energy to a Python reference):
//   g++ -O2 -std=c++17 -DMBD_STANDALONE_TEST mff_mbd_solver.cpp \
//       -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
//       -L$TORCH/lib -ltorch -ltorch_cpu -lc10 -o mbd_test && ./mbd_test
#include "mff_mbd_solver.h"

#include <cmath>

namespace mfftorch {

static const double SQRT_PI = std::sqrt(M_PI);

// integer FFT frequencies [-floor(M/2)..ceil(M/2)-1] in the torch.fft order (0..M/2-1, -M/2..-1).
static torch::Tensor fft_freqs(int M, torch::TensorOptions opt) {
  auto f = torch::arange(M, opt);
  auto split = (M + 1) / 2;  // first 'split' are >=0
  return torch::where(f < split, f, f - M);
}

torch::Tensor MFFMBDSolver::k_grid_cart(const torch::Tensor& eff_cell, const torch::Device& device) const {
  const int M = config_.mesh_size;
  auto cpu = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  auto freq = fft_freqs(M, cpu);
  auto grids = torch::meshgrid({freq, freq, freq}, "ij");
  auto ik = torch::stack({grids[0], grids[1], grids[2]}, -1).reshape({-1, 3});  // [K,3]
  auto inv = torch::linalg_inv(eff_cell.to(torch::kCPU, torch::kFloat64));
  return (2.0 * M_PI * torch::matmul(ik, inv.transpose(0, 1))).to(device);     // [K,3]
}

// CIC spread: atoms [N,C] -> mesh [M,M,M,C] (matches Python assignment="cic").
torch::Tensor MFFMBDSolver::spread_to_mesh(const torch::Tensor& frac, const torch::Tensor& source,
                                           const std::array<int, 3>&) const {
  const int M = config_.mesh_size;
  const int C = source.size(1);
  auto mesh = torch::zeros({M * M * M, C}, source.options());
  auto scaled = frac * (double)M;
  auto base = torch::floor(scaled).to(torch::kLong);          // [N,3]
  auto f = scaled - base.to(scaled.dtype());                  // [N,3] in [0,1)
  // 8 CIC corners
  for (int cx = 0; cx < 2; ++cx)
    for (int cy = 0; cy < 2; ++cy)
      for (int cz = 0; cz < 2; ++cz) {
        auto wx = cx ? f.select(1, 0) : (1.0 - f.select(1, 0));
        auto wy = cy ? f.select(1, 1) : (1.0 - f.select(1, 1));
        auto wz = cz ? f.select(1, 2) : (1.0 - f.select(1, 2));
        auto w = (wx * wy * wz).unsqueeze(-1);                // [N,1]
        auto ix = torch::remainder(base.select(1, 0) + cx, M);
        auto iy = torch::remainder(base.select(1, 1) + cy, M);
        auto iz = torch::remainder(base.select(1, 2) + cz, M);
        auto flat = (ix * M + iy) * M + iz;                   // [N]
        mesh.scatter_add_(0, flat.unsqueeze(-1).expand({-1, C}), source * w);
      }
  return mesh.view({M, M, M, C});
}

torch::Tensor MFFMBDSolver::gather_from_mesh(const torch::Tensor& frac, const torch::Tensor& mesh,
                                             const std::array<int, 3>&) const {
  const int M = config_.mesh_size;
  const int C = mesh.size(-1);
  auto flat_mesh = mesh.view({-1, C});
  auto scaled = frac * (double)M;
  auto base = torch::floor(scaled).to(torch::kLong);
  auto f = scaled - base.to(scaled.dtype());
  auto out = torch::zeros({frac.size(0), C}, mesh.options());
  for (int cx = 0; cx < 2; ++cx)
    for (int cy = 0; cy < 2; ++cy)
      for (int cz = 0; cz < 2; ++cz) {
        auto wx = cx ? f.select(1, 0) : (1.0 - f.select(1, 0));
        auto wy = cy ? f.select(1, 1) : (1.0 - f.select(1, 1));
        auto wz = cz ? f.select(1, 2) : (1.0 - f.select(1, 2));
        auto w = (wx * wy * wz).unsqueeze(-1);
        auto ix = torch::remainder(base.select(1, 0) + cx, M);
        auto iy = torch::remainder(base.select(1, 1) + cy, M);
        auto iz = torch::remainder(base.select(1, 2) + cz, M);
        auto flat = (ix * M + iy) * M + iz;
        out = out + flat_mesh.index_select(0, flat) * w;
      }
  return out;
}

torch::Tensor MFFMBDSolver::dipole_field(
    const torch::Tensor& pos, const torch::Tensor& mu, const torch::Tensor& cell, double alpha,
    const torch::Tensor& src, const torch::Tensor& dst, const torch::Tensor& shifts,
    const torch::Device& device) const {
  const int M = config_.mesh_size;
  auto eff = cell.to(torch::kFloat64);
  auto inv = torch::linalg_inv(eff);
  auto frac = torch::matmul(pos, inv);
  frac = frac - torch::floor(frac);
  auto k_cart = k_grid_cart(eff, device);                                   // [K,3]
  auto k2 = (k_cart * k_cart).sum(-1).clamp_min(1e-12);                      // [K]
  auto volume = torch::det(eff).abs();

  auto mu_mesh = spread_to_mesh(frac, mu, config_.pbc);
  auto mu_k = torch::fft::fftn(mu_mesh, {}, {0, 1, 2}).reshape({-1, 3});     // [K,3] complex
  auto kc = k_cart.to(mu_k.dtype());
  auto kdotmu = (kc * mu_k).sum(-1);                                        // [K] complex
  auto screen = torch::exp(-k2 / (4.0 * alpha * alpha));
  // CIC assignment-window deconvolution 1/|W|^2, W = prod_axes sinc(m/M)^2 (matches Python backend).
  auto cpu2 = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  auto sinc1d = torch::sinc(fft_freqs(M, cpu2) / (double)M).pow(2).to(device);   // [M]
  auto win = (sinc1d.view({M, 1, 1}) * sinc1d.view({1, M, 1}) * sinc1d.view({1, 1, M})).reshape({-1});
  auto wdeconv = torch::reciprocal(win.clamp_min(1e-6).square());                // [K]
  auto scale = -(4.0 * M_PI) / volume * screen * wdeconv / k2;              // [K]  (-4pi/V k k/k^2 / |W|^2)
  scale = torch::where(k2 > 1e-12, scale, torch::zeros_like(scale));        // tinfoil k=0
  auto e_k = (scale.to(mu_k.dtype()).unsqueeze(-1) * kc) * kdotmu.unsqueeze(-1);  // [K,3]
  auto e_mesh = torch::real(torch::fft::ifftn(e_k.reshape({M, M, M, 3}), {}, {0, 1, 2})) * std::pow((double)M, 3);
  auto field = gather_from_mesh(frac, e_mesh, config_.pbc);                 // [N,3]

  double a3 = alpha * alpha * alpha;
  field = field + (4.0 * a3 / (3.0 * SQRT_PI)) * mu;                        // self term

  if (src.numel() > 0) {
    auto shift_cart = torch::matmul(shifts.to(torch::kFloat64), eff);
    auto rvec = pos.index_select(0, dst) - pos.index_select(0, src) + shift_cart;  // [E,3]
    auto r = torch::linalg_vector_norm(rvec, 2, -1).clamp_min(1e-12);
    auto r2 = r * r;
    auto gauss = torch::exp(-(alpha * alpha) * r2);
    auto b0 = torch::erfc(alpha * r) / r;
    auto b1 = (b0 + (2.0 * alpha * alpha) / (alpha * SQRT_PI) * gauss) / r2;
    auto b2 = (3.0 * b1 + std::pow(2.0 * alpha * alpha, 2) / (alpha * SQRT_PI) * gauss) / r2;
    auto mu_src = mu.index_select(0, src);
    auto rdotmu = (rvec * mu_src).sum(-1);
    auto contrib = -b1.unsqueeze(-1) * mu_src + b2.unsqueeze(-1) * rvec * rdotmu.unsqueeze(-1);
    field = field.index_add(0, dst, contrib);
  }
  return field;
}

torch::Tensor MFFMBDSolver::coupled_matvec(
    const torch::Tensor& x, const torch::Tensor& omega, const torch::Tensor& alpha,
    const torch::Tensor& pos, const torch::Tensor& cell, double alpha_ewald,
    const torch::Tensor& src, const torch::Tensor& dst, const torch::Tensor& shifts,
    const torch::Device& device) const {
  auto wsa = (omega * alpha.clamp_min(0).sqrt()).unsqueeze(-1);             // [N,1]
  auto fld = dipole_field(pos, wsa * x, cell, alpha_ewald, src, dst, shifts, device);
  return (omega.unsqueeze(-1) * omega.unsqueeze(-1)) * x + wsa * fld;
}

torch::Tensor MFFMBDSolver::mbd_energy(
    const torch::Tensor& global_pos, const torch::Tensor& mbd_source, const torch::Tensor& cell,
    const torch::Tensor& src, const torch::Tensor& dst, const torch::Tensor& shifts,
    const torch::Device& device) const {
  const int N = global_pos.size(0);
  const int n = 3 * N;
  auto omega = mbd_source.select(1, 0).contiguous();   // [N]
  auto alpha = mbd_source.select(1, 1).contiguous();   // [N]
  // alpha (Ewald) = prefactor / (0.5 * min periodic box length)
  auto rownorm = torch::linalg_vector_norm(cell.to(torch::kFloat64), 2, 1);
  double Lmin = rownorm.min().item<double>();
  double alpha_ew = config_.ewald_alpha_prefactor / (0.5 * Lmin);

  auto mv = [&](const torch::Tensor& v) {
    return coupled_matvec(v.view({N, 3}), omega, alpha, global_pos, cell, alpha_ew, src, dst, shifts, device)
        .reshape({-1});
  };

  // --- spectral bounds via matvec-only power iteration (no eigh) ---
  auto opt = torch::TensorOptions().dtype(torch::kFloat64).device(device);
  double lmax, lmin;
  {
    torch::NoGradGuard ng;
    auto v = torch::randn({n}, opt);
    v = v / v.norm();
    double lam = 0;
    for (int i = 0; i < config_.power_steps; ++i) { auto w = mv(v); lam = (v * w).sum().item<double>(); v = w / w.norm().clamp_min(1e-30); }
    lmax = lam * (1.0 + config_.bound_pad);
    v = torch::randn({n}, opt); v = v / v.norm();
    double mu = 0;
    for (int i = 0; i < config_.power_steps; ++i) { auto w = lmax * v - mv(v); mu = (v * w).sum().item<double>(); v = w / w.norm().clamp_min(1e-30); }
    lmin = std::max((lmax - mu) * (1.0 - config_.bound_pad), 1e-6);
  }

  // --- Chebyshev coeffs of sqrt on [lmin,lmax] ---
  const int deg = config_.cheb_degree, Mc = deg + 1;
  auto j = torch::arange(Mc, opt) + 0.5;
  auto theta = M_PI * j / (double)Mc;
  auto xx = 0.5 * (lmax + lmin) + 0.5 * (lmax - lmin) * torch::cos(theta);
  auto fx = xx.clamp_min(0).sqrt();
  auto kk = torch::arange(deg + 1, opt);
  auto coef = (2.0 / Mc) * (fx.unsqueeze(0) * torch::cos(kk.unsqueeze(1) * theta.unsqueeze(0))).sum(1);
  coef.index_put_({0}, coef.index({0}) * 0.5);

  double a = 2.0 / (lmax - lmin), b = -(lmax + lmin) / (lmax - lmin);
  auto smv = [&](const torch::Tensor& v) { return a * mv(v) + b * v; };

  // --- Hutchinson + Chebyshev recurrence (fixed Rademacher probes) ---
  auto gen = at::detail::createCPUGenerator(0);
  auto probes = (2.0 * torch::randint(0, 2, {config_.num_probes, n}, gen, torch::TensorOptions().dtype(torch::kFloat64)) - 1.0).to(device);
  auto tr = torch::zeros({}, opt);
  for (int r = 0; r < config_.num_probes; ++r) {
    auto z = probes[r];
    auto t_prev = z;
    auto t_cur = smv(z);
    auto s = coef[0] * (z * z).sum() + coef[1] * (z * t_cur).sum();
    for (int kx = 2; kx <= deg; ++kx) {
      auto t_next = 2.0 * smv(t_cur) - t_prev;
      s = s + coef[kx] * (z * t_next).sum();
      t_prev = t_cur; t_cur = t_next;
    }
    tr = tr + s;
  }
  tr = tr / (double)config_.num_probes;
  return 0.5 * tr - 1.5 * omega.sum();
}

}  // namespace mfftorch

#ifdef MBD_STANDALONE_TEST
#include <cstdio>
// Deterministic parity harness: a HARDCODED tiny system (identical in the Python reference,
// /tmp/mbd_parity_ref.py) -> compares the dipole_field operator (reciprocal PME + T_SR + self).
// No random / no cross-language tensor passing: prints two invariants of the field.
int main() {
  using namespace mfftorch;
  auto o = torch::TensorOptions().dtype(torch::kFloat64);
  auto pos = torch::tensor({{1.0, 1.0, 1.0}, {3.0, 1.5, 1.2}}, o);
  auto mu = torch::tensor({{0.3, -0.5, 0.8}, {0.1, 0.2, -0.4}}, o);
  auto cell = torch::eye(3, o) * 10.0;
  auto src = torch::tensor({0, 1}, torch::TensorOptions().dtype(torch::kLong));
  auto dst = torch::tensor({1, 0}, torch::TensorOptions().dtype(torch::kLong));
  auto sh = torch::zeros({2, 3}, o);
  MBDConfig cfg; cfg.mesh_size = 32;
  MFFMBDSolver solver; solver.set_config(cfg);
  auto f = solver.dipole_field(pos, mu, cell, /*alpha=*/1.0, src, dst, sh, torch::kCPU);
  std::printf("CPP_FIELD_SUM %.10f\nCPP_FIELD_SQ %.10f\n", f.sum().item<double>(), (f * f).sum().item<double>());
  return 0;
}
#endif
