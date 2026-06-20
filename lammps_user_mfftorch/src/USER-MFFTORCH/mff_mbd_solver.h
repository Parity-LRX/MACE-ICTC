// Task 5/5 (DRAFT -- pending build+verify on the 4090): C++ many-body-dispersion (MBD) solver for
// USER-MFFTORCH, sharing the cuFFT reciprocal BACKEND with the scalar electrostatics but NOT the
// physics. Mirrors the validated Python (mace_ictd/models/mbd.py + reciprocal_backend.py):
//
//   E_MBD = 1/2 Tr[sqrt C] - 3/2 sum_i omega_i,   C_pq = w_i^2 d_pq + (1-d) w_i w_j sqrt(a_i a_j) T_ij^LR
//
//   * dipole field T.mu (Ewald): reciprocal PME (spread mu[N,3] -> FFT -> -4pi/V k_a k_b/k^2 e^{-k^2/4a^2}
//     -> iFFT -> gather) + real-space T_SR (erfc B-functions) + self (+4a^3/3sqrtpi) ; tinfoil k=0.
//   * Tr[sqrt C] via CHEBYSHEV (deployment: pure matvec + fixed-degree polynomial, NO eigensolve ->
//     no torch::linalg::eigh in the hot path); spectral bounds via matvec-only power iteration.
//
// The model emits a per-atom MBD source [N, 2] = (omega, alpha) as the reciprocal_source (source_kind
// = "mbd"); the pair style routes source_kind=="mbd" here instead of the charge/multipole path.
//
// Shares with mff_reciprocal_solver: spread_to_mesh_full / gather_from_mesh_full / build_integer_
// frequencies / the GridSpec. This header declares the interface; the .cpp mirrors the Python ops 1:1.
#ifndef MFF_MBD_SOLVER_H
#define MFF_MBD_SOLVER_H

#include <torch/torch.h>
#include <array>

namespace mfftorch {

struct MBDConfig {
  int mesh_size = 32;
  double ewald_alpha_prefactor = 5.0;  // alpha = prefactor / (0.5 * min periodic box length)
  int cheb_degree = 24;                // Chebyshev degree for sqrt(x) (no eigensolve)
  int num_probes = 64;                 // Hutchinson trace probes (fixed Rademacher seed)
  int power_steps = 20;                // power-iteration steps for the spectral bounds
  double bound_pad = 0.05;             // pad on [lmin, lmax]
  double real_cutoff = 0.0;            // real-space T_SR cutoff (0 -> derive from alpha)
  std::array<int, 3> pbc{{1, 1, 1}};
};

class MFFMBDSolver {
 public:
  MFFMBDSolver() = default;
  void set_config(const MBDConfig& c) { config_ = c; }
  const MBDConfig& config() const { return config_; }

  // E_MBD for one (replicated) cell. global_pos [N,3]; mbd_source [N,2] = (omega, alpha); cell [3,3];
  // (src,dst,shifts) a real-space neighbour list for T_SR. Returns the scalar energy (autograd-live
  // w.r.t. global_pos and mbd_source so the pair style gets forces by backprop, like the recip path).
  torch::Tensor mbd_energy(
      const torch::Tensor& global_pos,
      const torch::Tensor& mbd_source,
      const torch::Tensor& cell,
      const torch::Tensor& src,
      const torch::Tensor& dst,
      const torch::Tensor& shifts,
      const torch::Device& device) const;

  // T.mu Ewald dipole field [N,3]  (reciprocal PME + real-space T_SR + self). Public for parity tests.
  torch::Tensor dipole_field(
      const torch::Tensor& pos, const torch::Tensor& mu, const torch::Tensor& cell,
      double alpha, const torch::Tensor& src, const torch::Tensor& dst, const torch::Tensor& shifts,
      const torch::Device& device) const;

 private:
  // C.x coupled-dipole matvec [N,3] -> [N,3].
  torch::Tensor coupled_matvec(
      const torch::Tensor& x, const torch::Tensor& omega, const torch::Tensor& alpha,
      const torch::Tensor& pos, const torch::Tensor& cell, double alpha_ewald,
      const torch::Tensor& src, const torch::Tensor& dst, const torch::Tensor& shifts,
      const torch::Device& device) const;

  // shared grid ops (mirror mff_reciprocal_solver) -- spread [N,C]->mesh, gather mesh->[N,C].
  torch::Tensor spread_to_mesh(const torch::Tensor& frac, const torch::Tensor& source, const std::array<int,3>& pbc) const;
  torch::Tensor gather_from_mesh(const torch::Tensor& frac, const torch::Tensor& mesh, const std::array<int,3>& pbc) const;
  torch::Tensor k_grid_cart(const torch::Tensor& eff_cell, const torch::Device& device) const;  // [K,3]

  MBDConfig config_;
};

}  // namespace mfftorch

#endif  // MFF_MBD_SOLVER_H
