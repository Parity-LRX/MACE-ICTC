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
  int assignment = 1;                   // mesh assignment: 0=CIC (2nd order), 1=PCS (4th-order cubic
                                        // B-spline) -> C^1 forces everywhere (no mesh-cell kink)
  int mesh_size = 32;                   // box-tied fallback only; the cutoff path adapts the mesh to alpha
  double ewald_alpha_prefactor = 5.0;  // box-tied fallback: alpha = prefactor / (0.5 * min box length)
  double ewald_bound = 5.0;            // cutoff path: alpha = ewald_bound / r_cut  (erfc(ewald_bound)=SR err)
  double mesh_per_alpha = 2.5;         // adaptive mesh points per (alpha * box length): resolve the screen
  int mesh_min = 16;                   // adaptive-mesh clamp
  int mesh_max = 64;
  int cheb_degree = 24;                // Chebyshev degree for sqrt(x) (no eigensolve)
  int num_probes = 64;                 // Hutchinson trace probes (fixed Rademacher seed)
  int power_steps = 20;                // power-iteration steps for the spectral bounds
  double bound_pad = 0.05;             // pad on [lmin, lmax]
  double cheb_lmin = 0.0;              // fixed spectral bounds for the Chebyshev (0 -> auto power-iter).
  double cheb_lmax = 0.0;              // FIX these over an MD trajectory -> conservative forces.
  double real_cutoff = 0.0;            // real-space T_SR cutoff (0 -> derive from alpha)
  std::array<int, 3> pbc{{1, 1, 1}};
};

// Deployment output -- mirrors mff_reciprocal_solver's ReciprocalOutputs so the pair style consumes
// MBD identically to the electrostatics path (energy + per-atom conservative forces).
struct MBDOutputs {
  double energy = 0.0;
  torch::Tensor forces;       // [N,3] = -dE/dpos (autograd)
  torch::Tensor atom_energy;  // [N] (equal split for now)
  double lmin = 0.0, lmax = 0.0;  // spectral bounds used (fix these across a trajectory -> conservative)
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
      const torch::Device& device,
      double alpha_ewald = -1.0,   // Ewald split parameter; <=0 -> box-tied prefactor/(0.5*Lmin)
      double* used_lmin = nullptr, double* used_lmax = nullptr) const;

  // Deployment entry point: reuse a LAMMPS-provided real-space neighbour list (src,dst,shifts) at
  // `real_cutoff` (e.g. the pair_style 'dispersion <cutoff>' ghost list). The Ewald split parameter is
  // TIED TO THE CUTOFF (alpha = ewald_bound/real_cutoff) so the erfc near field fits inside the list,
  // and the mesh ADAPTS to that alpha so the reciprocal far field (no cutoff) stays converged. Runs
  // mbd_energy under autograd; returns energy + conservative forces. `src,dst,shifts` must be a FULL
  // (both-directions) edge list. NON-const: adapts config_.mesh_size to the box+alpha.
  MBDOutputs compute(const torch::Tensor& pos, const torch::Tensor& source, const torch::Tensor& cell,
                     const torch::Tensor& src, const torch::Tensor& dst, const torch::Tensor& shifts,
                     double real_cutoff, const torch::Device& device);

  // Fallback entry point (no LAMMPS list): builds an O(N^2) periodic neighbour list internally and
  // ties alpha to the box. For small cells / standalone tests.
  MBDOutputs compute(const torch::Tensor& pos, const torch::Tensor& source, const torch::Tensor& cell,
                     const torch::Device& device);

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

  // periodic neighbour list within `cutoff` -> (src, dst, shift_int [E,3]) for T_SR (single rank).
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> build_periodic_neighbors(
      const torch::Tensor& pos, const torch::Tensor& cell, double cutoff) const;

  // adaptive mesh: M points/axis so the FFT Nyquist resolves the e^{-k^2/4a^2} screen at this alpha.
  int adaptive_mesh(double alpha, const torch::Tensor& cell) const;

  // shared core: requires_grad on pos -> mbd_energy(alpha_ewald) -> autograd forces -> MBDOutputs.
  MBDOutputs run_autograd(const torch::Tensor& pos, const torch::Tensor& source, const torch::Tensor& cell,
                          const torch::Tensor& src, const torch::Tensor& dst, const torch::Tensor& shifts,
                          double alpha_ewald, const torch::Device& device) const;

  MBDConfig config_;
};

}  // namespace mfftorch

#endif  // MFF_MBD_SOLVER_H
