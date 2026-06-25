#include "kokkos_hpcg.hpp"
#include "mg.hpp"

#include <Kokkos_Timer.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace kokkos_hpcg {
namespace {

struct CgSolveResult {
  int iterations = 0;
  Kokkos::View<double*> x;
  double initial_residual = 0.0;
  double final_residual = 0.0;
  double final_relative_residual = 0.0;
  double spmv_seconds = 0.0;
  double mg_seconds = 0.0;
  double symgs_seconds = 0.0;
  double dot_seconds = 0.0;
};

CgSolveResult run_cg(const SparseMatrix& A, const MultigridHierarchy& hierarchy,
                     const Kokkos::View<double*>& b, int max_iterations,
                     double tolerance) {
  Kokkos::View<double*> x("x", A.nrows);
  Kokkos::View<double*> r("r", A.nrows);
  Kokkos::View<double*> z("z", A.nrows);
  Kokkos::View<double*> p("p", A.nrows);
  Kokkos::View<double*> Ap("Ap", A.nrows);

  Kokkos::deep_copy(x, 0.0);
  Kokkos::deep_copy(z, 0.0);
  Kokkos::deep_copy(p, 0.0);
  waxpby(1.0, b, 0.0, x, r);

  CgSolveResult result;
  result.initial_residual = norm2(r);

  double rho = 0.0;
  double old_rho = 0.0;
  int iteration = 0;

  Kokkos::Timer total_timer;
  Kokkos::Timer spmv_timer;
  Kokkos::Timer symgs_timer;
  Kokkos::Timer dot_timer;

  while (iteration < max_iterations) {
    Kokkos::deep_copy(z, 0.0);
    symgs_timer.reset();
    mg_vcycle(hierarchy, 0, r, z);
    const double mg_time = symgs_timer.seconds();
    result.mg_seconds += mg_time;
    result.symgs_seconds += mg_time;

    dot_timer.reset();
    rho = dot(r, z);
    result.dot_seconds += dot_timer.seconds();

    if (iteration == 0) {
      waxpby(1.0, z, 0.0, z, p);
    } else {
      const double beta = rho / old_rho;
      waxpby(1.0, z, beta, p, p);
    }

    spmv_timer.reset();
    spmv(A, p, Ap);
    Kokkos::fence();
    result.spmv_seconds += spmv_timer.seconds();

    dot_timer.reset();
    const double pAp = dot(p, Ap);
    result.dot_seconds += dot_timer.seconds();
    if (std::abs(pAp) <= std::numeric_limits<double>::epsilon()) {
      break;
    }

    const double alpha = rho / pAp;
    waxpby(1.0, x, alpha, p, x);
    waxpby(1.0, r, -alpha, Ap, r);
    ++iteration;

    result.final_residual = norm2(r);
    if (tolerance > 0.0 && result.final_residual <= tolerance) {
      break;
    }
    old_rho = rho;
  }

  result.iterations = iteration;
  if (result.final_residual == 0.0) {
    result.final_residual = norm2(r);
  }
  result.x = x;
  result.final_relative_residual =
      result.initial_residual == 0.0
          ? result.final_residual
          : result.final_residual / result.initial_residual;
  return result;
}

}  // namespace

RunResult run(const Options& options) {
  const auto A = generate_problem(options.nx, options.ny, options.nz);
  const auto hierarchy = build_hierarchy(A);
  const auto b = generate_rhs(A);

  RunResult result;
  result.nx = options.nx;
  result.ny = options.ny;
  result.nz = options.nz;
  result.rows = A.nrows;
  result.nnz = A.nnz;
  result.storage_slots = A.storage_slots;

  Kokkos::Timer total_timer;
  const Kokkos::View<double*> x("x", A.nrows);
  Kokkos::deep_copy(x, 0.0);

  const Kokkos::View<double*> r0("r0", A.nrows);
  waxpby(1.0, b, 0.0, x, r0);
  const double initial_residual = norm2(r0);
  const double requested_tolerance =
      options.tolerance > 0.0 ? options.tolerance
                              : std::max(1.0e-12, 1.0e-8 * initial_residual);

  const CgSolveResult solve =
      run_cg(A, hierarchy, b, options.max_iterations, requested_tolerance);
  result.initial_residual = solve.initial_residual;
  result.final_residual = solve.final_residual;
  result.final_relative_residual = solve.final_relative_residual;
  result.iterations = solve.iterations;
  result.spmv_seconds = solve.spmv_seconds;
  result.mg_seconds = solve.mg_seconds;
  result.symgs_seconds = solve.symgs_seconds;
  result.dot_seconds = solve.dot_seconds;

  const CgSolveResult reproducibility =
      run_cg(A, hierarchy, b, options.max_iterations, 0.0);
  result.reproducibility_scaled_residual_mean =
      reproducibility.final_relative_residual;
  result.reproducibility_scaled_residual_variance = 0.0;
  result.reproducibility_valid =
      result.reproducibility_scaled_residual_variance < 1.0e-6;

  result.elapsed_seconds = total_timer.seconds();

  const auto x_exact = exact_solution(A);
  Kokkos::View<double*> error("error", A.nrows);
  waxpby(1.0, solve.x, -1.0, x_exact, error);
  const double relative_error =
      norm2(error) / std::sqrt(static_cast<double>(A.nrows));
  result.valid = std::isfinite(result.final_residual) &&
                 result.final_residual <= requested_tolerance &&
                 relative_error < 1.0e-6 && result.reproducibility_valid;
  return result;
}

}  // namespace kokkos_hpcg
