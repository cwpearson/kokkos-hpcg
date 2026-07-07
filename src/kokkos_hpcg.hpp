#pragma once

#include "spmv.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>

namespace kokkos_hpcg {

struct Options {
  int nx = 16;
  int ny = 16;
  int nz = 16;
  int max_iterations = 50;
  double tolerance = 0.0;
  double runtime_seconds = 0.0;
};

struct SparseMatrix {
  static constexpr int max_entries_per_row = 27;

  int nx = 0;
  int ny = 0;
  int nz = 0;
  int nrows = 0;
  int nnz = 0;
  int storage_slots = 0;
  Kokkos::View<int*> row_length;
  Kokkos::View<int**, Kokkos::LayoutRight> col_idx;
  Kokkos::View<double**, Kokkos::LayoutRight> values;
  Kokkos::View<int*> diag_slot;
};

struct RunResult {
  int nx = 0;
  int ny = 0;
  int nz = 0;
  int rows = 0;
  int nnz = 0;
  int storage_slots = 0;
  int iterations = 0;
  double initial_residual = 0.0;
  double final_residual = 0.0;
  double final_relative_residual = 0.0;
  double reproducibility_scaled_residual_mean = 0.0;
  double reproducibility_scaled_residual_variance = 0.0;
  double elapsed_seconds = 0.0;
  double spmv_seconds = 0.0;
  double mg_seconds = 0.0;
  double symgs_seconds = 0.0;
  double dot_seconds = 0.0;
  bool valid = false;
  bool reproducibility_valid = false;
};

Options parse_options(int argc, char** argv);
SparseMatrix generate_problem(int nx, int ny, int nz);
Kokkos::View<double*> exact_solution(const SparseMatrix& A);
Kokkos::View<double*> generate_rhs(const SparseMatrix& A);

void waxpby(double alpha, const Kokkos::View<double*>& x, double beta,
            const Kokkos::View<double*>& y, const Kokkos::View<double*>& w);
double dot(const Kokkos::View<double*>& x, const Kokkos::View<double*>& y);
double norm2(const Kokkos::View<double*>& x);
void symgs(const SparseMatrix& A, const Kokkos::View<double*>& b,
           const Kokkos::View<double*>& x);

RunResult run(const Options& options);
void print_result(std::ostream& os, const RunResult& result);

int linear_index(int i, int j, int k, int nx, int ny);
std::string validation_string(bool valid);

}  // namespace kokkos_hpcg
