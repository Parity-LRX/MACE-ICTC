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

// stencil size + window exponent per assignment (matches Python _assignment_stencil_size).
static int assignment_stencil(int a) { return a == 1 ? 4 : 2; }  // 1=pcs (4-pt cubic), 0=cic (2-pt)

// per-axis assignment weights w[0..S-1] (each [N]) for the in-cell offset f in [0,1).
// Mirrors Python _assignment_kernel_1d (cic: linear; pcs: cubic B-spline).
static std::vector<torch::Tensor> axis_weights(const torch::Tensor& f, int a) {
  if (a == 1) {  // pcs (cubic B-spline) -> C^2 weight, C^1 force, no mesh-cell discontinuity
    auto f2 = f * f, f3 = f2 * f;
    return {torch::pow(1.0 - f, 3) / 6.0,
            (3.0 * f3 - 6.0 * f2 + 4.0) / 6.0,
            (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0) / 6.0,
            f3 / 6.0};
  }
  return {1.0 - f, f};  // cic (linear)
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

// spread atoms [N,C] -> mesh [M,M,M,C]. S^3 stencil (S=2 cic / 4 pcs); matches Python _spread_source_to_mesh.
torch::Tensor MFFMBDSolver::spread_to_mesh(const torch::Tensor& frac, const torch::Tensor& source,
                                           const std::array<int, 3>&) const {
  const int M = config_.mesh_size, C = source.size(1), a = config_.assignment, S = assignment_stencil(a);
  auto mesh = torch::zeros({M * M * M, C}, source.options());
  auto scaled = frac * (double)M;
  auto floor_scaled = torch::floor(scaled);
  auto f = scaled - floor_scaled;                                          // [N,3] in [0,1)
  auto base = floor_scaled.to(torch::kLong) - (a == 1 ? 1 : 0);           // pcs base = floor-1
  std::vector<torch::Tensor> WX = axis_weights(f.select(1, 0), a), WY = axis_weights(f.select(1, 1), a),
                             WZ = axis_weights(f.select(1, 2), a);
  for (int ox = 0; ox < S; ++ox)
    for (int oy = 0; oy < S; ++oy)
      for (int oz = 0; oz < S; ++oz) {
        auto w = (WX[ox] * WY[oy] * WZ[oz]).unsqueeze(-1);                 // [N,1]
        auto ix = torch::remainder(base.select(1, 0) + ox, M);
        auto iy = torch::remainder(base.select(1, 1) + oy, M);
        auto iz = torch::remainder(base.select(1, 2) + oz, M);
        auto flat = (ix * M + iy) * M + iz;                               // [N]
        mesh.scatter_add_(0, flat.unsqueeze(-1).expand({-1, C}), source * w);
      }
  return mesh.view({M, M, M, C});
}

torch::Tensor MFFMBDSolver::gather_from_mesh(const torch::Tensor& frac, const torch::Tensor& mesh,
                                             const std::array<int, 3>&) const {
  const int M = config_.mesh_size, C = mesh.size(-1), a = config_.assignment, S = assignment_stencil(a);
  auto flat_mesh = mesh.view({-1, C});
  auto scaled = frac * (double)M;
  auto floor_scaled = torch::floor(scaled);
  auto f = scaled - floor_scaled;
  auto base = floor_scaled.to(torch::kLong) - (a == 1 ? 1 : 0);
  std::vector<torch::Tensor> WX = axis_weights(f.select(1, 0), a), WY = axis_weights(f.select(1, 1), a),
                             WZ = axis_weights(f.select(1, 2), a);
  auto out = torch::zeros({frac.size(0), C}, mesh.options());
  for (int ox = 0; ox < S; ++ox)
    for (int oy = 0; oy < S; ++oy)
      for (int oz = 0; oz < S; ++oz) {
        auto w = (WX[ox] * WY[oy] * WZ[oz]).unsqueeze(-1);
        auto ix = torch::remainder(base.select(1, 0) + ox, M);
        auto iy = torch::remainder(base.select(1, 1) + oy, M);
        auto iz = torch::remainder(base.select(1, 2) + oz, M);
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
  // assignment-window deconvolution 1/|W|^2, W = prod_axes sinc(m/M)^exp (exp=2 cic / 4 pcs; Python backend).
  auto cpu2 = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  auto sinc1d = torch::sinc(fft_freqs(M, cpu2) / (double)M).pow(assignment_stencil(config_.assignment)).to(device);  // [M]
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
    const torch::Device& device, double alpha_ewald, double* used_lmin, double* used_lmax) const {
  const int N = global_pos.size(0);
  const int n = 3 * N;
  auto omega = mbd_source.select(1, 0).contiguous();   // [N]
  auto alpha = mbd_source.select(1, 1).contiguous();   // [N]
  // alpha (Ewald) = prefactor / (0.5 * min periodic box length)
  auto rownorm = torch::linalg_vector_norm(cell.to(torch::kFloat64), 2, 1);
  double Lmin = rownorm.min().item<double>();
  double alpha_ew = (alpha_ewald > 0.0) ? alpha_ewald : config_.ewald_alpha_prefactor / (0.5 * Lmin);

  auto mv = [&](const torch::Tensor& v) {
    return coupled_matvec(v.view({N, 3}), omega, alpha, global_pos, cell, alpha_ew, src, dst, shifts, device)
        .reshape({-1});
  };

  // --- spectral bounds: fixed (config) for conservative MD, else matvec-only power iteration ---
  auto opt = torch::TensorOptions().dtype(torch::kFloat64).device(device);
  double lmax, lmin;
  if (config_.cheb_lmax > 0.0) {
    lmin = config_.cheb_lmin; lmax = config_.cheb_lmax;  // E becomes a smooth fn of p -> conservative
  } else {
    torch::NoGradGuard ng;
    // deterministic inits (CPU generator -> device) so bounds, hence E, are reproducible across calls.
    auto g1 = at::detail::createCPUGenerator(12345);
    auto v = torch::randn({n}, g1, torch::TensorOptions().dtype(torch::kFloat64)).to(device);
    v = v / v.norm();
    double lam = 0;
    for (int i = 0; i < config_.power_steps; ++i) { auto w = mv(v); lam = (v * w).sum().item<double>(); v = w / w.norm().clamp_min(1e-30); }
    lmax = lam * (1.0 + config_.bound_pad);
    auto g2 = at::detail::createCPUGenerator(67890);
    v = torch::randn({n}, g2, torch::TensorOptions().dtype(torch::kFloat64)).to(device); v = v / v.norm();
    double mu = 0;
    for (int i = 0; i < config_.power_steps; ++i) { auto w = lmax * v - mv(v); mu = (v * w).sum().item<double>(); v = w / w.norm().clamp_min(1e-30); }
    lmin = std::max((lmax - mu) * (1.0 - config_.bound_pad), 1e-6);
  }
  if (used_lmin) *used_lmin = lmin;
  if (used_lmax) *used_lmax = lmax;

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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MFFMBDSolver::build_periodic_neighbors(
    const torch::Tensor& pos, const torch::Tensor& cell, double cutoff) const {
  auto opt = torch::TensorOptions().dtype(torch::kFloat64).device(pos.device());
  auto cellf = cell.to(torch::kFloat64);
  auto rownorm = torch::linalg_vector_norm(cellf, 2, 1);  // [3]
  int na = config_.pbc[0] ? (int)std::ceil(cutoff / rownorm[0].item<double>()) : 0;
  int nb = config_.pbc[1] ? (int)std::ceil(cutoff / rownorm[1].item<double>()) : 0;
  int nc = config_.pbc[2] ? (int)std::ceil(cutoff / rownorm[2].item<double>()) : 0;
  std::vector<torch::Tensor> vsrc, vdst, vshift;
  auto pi = pos.unsqueeze(1);  // [N,1,3]
  auto pj = pos.unsqueeze(0);  // [1,N,3]
  for (int i = -na; i <= na; ++i)
    for (int j = -nb; j <= nb; ++j)
      for (int k = -nc; k <= nc; ++k) {
        auto shift_int = torch::tensor({(double)i, (double)j, (double)k}, opt);     // [3]
        auto shift_cart = torch::matmul(shift_int, cellf);                          // [3]
        auto rij = pj + shift_cart - pi;                                            // [N,N,3]  pos[q]+s-pos[p]
        auto dist = torch::linalg_vector_norm(rij, 2, -1);                          // [N,N]
        auto within = dist < cutoff;
        if (i == 0 && j == 0 && k == 0) within = within & (dist > 1e-8);            // drop self
        auto idx = torch::nonzero(within);                                          // [E,2] (p,q)
        if (idx.size(0) == 0) continue;
        vsrc.push_back(idx.select(1, 0));
        vdst.push_back(idx.select(1, 1));
        vshift.push_back(shift_int.unsqueeze(0).expand({idx.size(0), 3}).contiguous());
      }
  if (vsrc.empty()) {
    auto e = torch::empty({0}, torch::TensorOptions().dtype(torch::kLong).device(pos.device()));
    return {e, e, torch::empty({0, 3}, opt)};
  }
  return {torch::cat(vsrc), torch::cat(vdst), torch::cat(vshift)};
}

MBDOutputs MFFMBDSolver::run_autograd(const torch::Tensor& pos, const torch::Tensor& source,
                                      const torch::Tensor& cell, const torch::Tensor& src,
                                      const torch::Tensor& dst, const torch::Tensor& shift,
                                      double alpha_ewald, const torch::Device& device) const {
  const int N = pos.size(0);
  torch::AutoGradMode grad_on(true);
  auto p = pos.to(device, torch::kFloat64).clone().detach().set_requires_grad(true);
  auto cellf = cell.to(device, torch::kFloat64);
  auto src_ = source.to(device, torch::kFloat64);
  double bl = 0.0, bu = 0.0;
  auto E = mbd_energy(p, src_, cellf, src, dst, shift, device, alpha_ewald, &bl, &bu);
  auto grads = torch::autograd::grad({E}, {p}, /*grad_outputs=*/{}, /*retain_graph=*/false,
                                     /*create_graph=*/false, /*allow_unused=*/true);
  MBDOutputs out;
  out.lmin = bl; out.lmax = bu;
  out.energy = E.item<double>();
  out.forces = grads[0].defined() ? (-grads[0]).detach()
                                  : torch::zeros({N, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
  out.atom_energy = torch::full({N}, out.energy / N, torch::TensorOptions().dtype(torch::kFloat64).device(device));
  return out;
}

int MFFMBDSolver::adaptive_mesh(double alpha, const torch::Tensor& cell) const {
  // FFT Nyquist k_Nyq = pi*M/L must resolve the e^{-k^2/4a^2} screen -> M ~ mesh_per_alpha * alpha * L.
  double Lmax = torch::linalg_vector_norm(cell.to(torch::kFloat64), 2, 1).max().item<double>();
  int M = (int)std::ceil(config_.mesh_per_alpha * alpha * Lmax);
  if (M < config_.mesh_min) M = config_.mesh_min;
  if (M > config_.mesh_max) M = config_.mesh_max;
  return M;
}

// Deployment path: reuse a LAMMPS real-space list at `real_cutoff`; alpha tied to the cutoff, mesh
// adapted to alpha (the reciprocal far field has no cutoff -> "dispersion range > r_cut" is fine).
MBDOutputs MFFMBDSolver::compute(const torch::Tensor& pos, const torch::Tensor& source,
                                 const torch::Tensor& cell, const torch::Tensor& src,
                                 const torch::Tensor& dst, const torch::Tensor& shifts,
                                 double real_cutoff, const torch::Device& device) {
  double alpha_ew = config_.ewald_bound / real_cutoff;        // erfc(ewald_bound) = SR truncation error
  config_.mesh_size = adaptive_mesh(alpha_ew, cell);          // resolve the screen at this alpha
  return run_autograd(pos, source, cell, src, dst, shifts, alpha_ew, device);
}

// Fallback path: build an O(N^2) periodic list, alpha tied to the box.
MBDOutputs MFFMBDSolver::compute(const torch::Tensor& pos, const torch::Tensor& source,
                                 const torch::Tensor& cell, const torch::Device& device) {
  auto cellf = cell.to(torch::kFloat64);
  double Lmin = torch::linalg_vector_norm(cellf, 2, 1).min().item<double>();
  double alpha_ew = config_.ewald_alpha_prefactor / (0.5 * Lmin);
  double cutoff = config_.real_cutoff > 0.0 ? config_.real_cutoff : 5.0 / alpha_ew;  // erfc(5)~1e-12
  config_.mesh_size = adaptive_mesh(alpha_ew, cellf);
  torch::Tensor src, dst, shift;
  {
    torch::NoGradGuard ng;
    std::tie(src, dst, shift) = build_periodic_neighbors(pos.to(device, torch::kFloat64), cellf, cutoff);
  }
  return run_autograd(pos, source, cell, src, dst, shift, alpha_ew, device);
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
  const char* names[2] = {"CIC", "PCS"};
  for (int a = 0; a < 2; ++a) {  // verify BOTH assignments match the Python backend
    MBDConfig cfg; cfg.mesh_size = 32; cfg.assignment = a;
    MFFMBDSolver solver; solver.set_config(cfg);
    auto f = solver.dipole_field(pos, mu, cell, /*alpha=*/1.0, src, dst, sh, torch::kCPU);
    std::printf("CPP_%s_SUM %.10f\n", names[a], f.sum().item<double>());
    std::printf("CPP_%s_SQ %.10f\n", names[a], (f * f).sum().item<double>());
  }
  // full solver end-to-end (pcs): source [N,2] = (omega, alpha); power-iter bounds + Chebyshev energy.
  MBDConfig cfg; cfg.mesh_size = 32; cfg.cheb_degree = 20; cfg.num_probes = 64;  // assignment defaults to pcs
  MFFMBDSolver solver; solver.set_config(cfg);
  auto source = torch::tensor({{1.0, 0.3}, {1.1, 0.3}}, o);  // (omega, alpha_pol)
  auto E = solver.mbd_energy(pos, source, cell, src, dst, sh, torch::kCPU);
  std::printf("CPP_E_MBD %.8f (finite=%d)\n", E.item<double>(), (int)std::isfinite(E.item<double>()));
  return 0;
}
#endif

#ifdef MBD_MD_TEST
#include <cstdio>
#include <vector>
// Demonstrates PCS (4th-order) gives CONSERVATIVE forces at mesh-cell boundaries where 2nd-order CIC
// does NOT. The crystal sits ON the grid (z = k*3 -> scaled 8k = integer at mesh 24; atoms 0=(0,0,0)
// and 26=(6,6,6) fully on boundaries), so the boundary-aligned components are exactly the CIC
// force-discontinuity case. Per assignment: fix spectral bounds (-> deterministic surrogate) then
// check autograd force == central finite difference.
int main() {
  using namespace mfftorch;
  auto o = torch::TensorOptions().dtype(torch::kFloat64);
  const int nx = 3; const double a = 3.0; const double L = nx * a;
  std::vector<std::array<double, 3>> P;
  for (int i = 0; i < nx; ++i) for (int j = 0; j < nx; ++j) for (int k = 0; k < nx; ++k)
    P.push_back({i * a + 0.15 * ((i + j + k) % 2), j * a + 0.1 * (k % 2), k * a});  // z = k*3, ON the grid
  const int N = (int)P.size();
  auto pos = torch::zeros({N, 3}, o);
  for (int n = 0; n < N; ++n) { pos[n][0] = P[n][0]; pos[n][1] = P[n][1]; pos[n][2] = P[n][2]; }
  auto cell = torch::eye(3, o) * L;
  auto source = torch::zeros({N, 2}, o);
  for (int n = 0; n < N; ++n) { source[n][0] = (n % 2 ? 1.1 : 1.0); source[n][1] = 0.30; }

  const char* names[2] = {"CIC", "PCS"};
  for (int asg = 0; asg < 2; ++asg) {
    // mesh 12: z=k*3 -> scaled 4k = integer -> atoms 0,26 ON cell boundaries; small for the FD sweep.
    MBDConfig cfg; cfg.assignment = asg; cfg.mesh_min = cfg.mesh_max = 12;
    cfg.cheb_degree = 14; cfg.num_probes = 16; cfg.ewald_alpha_prefactor = 5.0;
    MFFMBDSolver solver; solver.set_config(cfg);
    auto out = solver.compute(pos, source, cell, torch::kCPU);
    cfg.cheb_lmin = out.lmin / 1.2; cfg.cheb_lmax = out.lmax * 1.2; solver.set_config(cfg);
    out = solver.compute(pos, source, cell, torch::kCPU);
    const double h = 1e-5; double worst = 0.0;
    auto Eat = [&](int atom, int comp, double dh) { auto pp = pos.clone(); pp[atom][comp] += dh;
                                                    return solver.compute(pp, source, cell, torch::kCPU).energy; };
    for (int atom : {0, N / 2, N - 1})
      for (int comp = 0; comp < 3; ++comp) {
        double fd = -(Eat(atom, comp, h) - Eat(atom, comp, -h)) / (2 * h);
        double fa = out.forces[atom][comp].item<double>();
        worst = std::max(worst, std::abs(fa - fd) / (std::abs(fd) + 1e-6));
      }
    std::printf("assignment=%s mesh=%d (atoms ON cell boundaries): E=%.8f worst conservativity rel err = %.2e  %s\n",
                names[asg], cfg.mesh_min, out.energy, worst, worst < 1e-3 ? "CONSERVATIVE" : "DISCONTINUOUS-FORCE");
  }
  return 0;
}
#endif
