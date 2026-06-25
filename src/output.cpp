#include "kokkos_hpcg.hpp"

#include <format>
#include <ostream>

namespace kokkos_hpcg {

std::string validation_string(bool valid) {
  return valid ? "PASSED" : "FAILED";
}

void print_result(std::ostream& os, const RunResult& result) {
  os << std::format(
      "HPCG.version=3.1-compatible\n"
      "HPCG.implementation=kokkos-hpcg\n"
      "HPCG.mpi.enabled=0\n"
      "HPCG.problem.nx={}\n"
      "HPCG.problem.ny={}\n"
      "HPCG.problem.nz={}\n"
      "HPCG.problem.rows={}\n"
      "HPCG.problem.nonzeros={}\n"
      "HPCG.problem.storage_slots={}\n"
      "HPCG.result.iterations={}\n"
      "HPCG.result.initial_residual={:.17g}\n"
      "HPCG.result.final_residual={:.17g}\n"
      "HPCG.result.final_relative_residual={:.17g}\n"
      "HPCG.result.validation={}\n"
      "Reproducibility Information::Result={}\n"
      "Reproducibility Information::Scaled residual mean={:.17g}\n"
      "Reproducibility Information::Scaled residual variance={:.17g}\n"
      "HPCG.timing.total_seconds={:.17g}\n"
      "HPCG.timing.spmv_seconds={:.17g}\n"
      "HPCG.timing.mg_seconds={:.17g}\n"
      "HPCG.timing.symgs_seconds={:.17g}\n"
      "HPCG.timing.dot_seconds={:.17g}\n",
      result.nx, result.ny, result.nz, result.rows, result.nnz,
      result.storage_slots, result.iterations, result.initial_residual,
      result.final_residual, result.final_relative_residual,
      validation_string(result.valid),
      validation_string(result.reproducibility_valid),
      result.reproducibility_scaled_residual_mean,
      result.reproducibility_scaled_residual_variance, result.elapsed_seconds,
      result.spmv_seconds, result.mg_seconds, result.symgs_seconds,
      result.dot_seconds);
}

}  // namespace kokkos_hpcg
