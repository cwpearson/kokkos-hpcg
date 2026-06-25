#include "kokkos_hpcg.hpp"

#include <stdexcept>

namespace kokkos_hpcg {
namespace {

constexpr double kDiagonalValue = 26.0;
constexpr double kOffDiagonalValue = -1.0;

}  // namespace

int linear_index(int i, int j, int k, int nx, int ny) {
  return i + nx * (j + ny * k);
}

SparseMatrix generate_problem(int nx, int ny, int nz) {
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    throw std::invalid_argument("problem dimensions must be positive");
  }

  const int nrows = nx * ny * nz;

  int nnz = 0;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        for (int dz = -1; dz <= 1; ++dz) {
          for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
              const int ni = i + dx;
              const int nj = j + dy;
              const int nk = k + dz;
              if (ni >= 0 && ni < nx && nj >= 0 && nj < ny && nk >= 0 &&
                  nk < nz) {
                ++nnz;
              }
            }
          }
        }
      }
    }
  }

  SparseMatrix A;
  A.nx = nx;
  A.ny = ny;
  A.nz = nz;
  A.nrows = nrows;
  A.nnz = nnz;
  A.storage_slots = nrows * SparseMatrix::max_entries_per_row;
  A.row_length = Kokkos::View<int*>("row_length", nrows);
  A.col_idx = Kokkos::View<int**, Kokkos::LayoutRight>(
      "col_idx", nrows, SparseMatrix::max_entries_per_row);
  A.values = Kokkos::View<double**, Kokkos::LayoutRight>(
      "values", nrows, SparseMatrix::max_entries_per_row);
  A.diag_slot = Kokkos::View<int*>("diag_slot", nrows);

  auto row_length_h = Kokkos::create_mirror_view(A.row_length);
  auto col_idx_h = Kokkos::create_mirror_view(A.col_idx);
  auto values_h = Kokkos::create_mirror_view(A.values);
  auto diag_slot_h = Kokkos::create_mirror_view(A.diag_slot);

  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const int row = linear_index(i, j, k, nx, ny);
        int slot = 0;
        for (int padded_slot = 0;
             padded_slot < SparseMatrix::max_entries_per_row; ++padded_slot) {
          col_idx_h(row, padded_slot) = row;
          values_h(row, padded_slot) = 0.0;
        }
        for (int dz = -1; dz <= 1; ++dz) {
          for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
              const int ni = i + dx;
              const int nj = j + dy;
              const int nk = k + dz;
              if (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 ||
                  nk >= nz) {
                continue;
              }
              const int col = linear_index(ni, nj, nk, nx, ny);
              col_idx_h(row, slot) = col;
              values_h(row, slot) =
                  (col == row) ? kDiagonalValue : kOffDiagonalValue;
              if (col == row) {
                diag_slot_h(row) = slot;
              }
              ++slot;
            }
          }
        }
        row_length_h(row) = slot;
      }
    }
  }

  Kokkos::deep_copy(A.row_length, row_length_h);
  Kokkos::deep_copy(A.col_idx, col_idx_h);
  Kokkos::deep_copy(A.values, values_h);
  Kokkos::deep_copy(A.diag_slot, diag_slot_h);
  return A;
}

Kokkos::View<double*> exact_solution(const SparseMatrix& A) {
  Kokkos::View<double*> x("x_exact", A.nrows);
  Kokkos::deep_copy(x, 1.0);
  return x;
}

Kokkos::View<double*> generate_rhs(const SparseMatrix& A) {
  auto x = exact_solution(A);
  Kokkos::View<double*> b("b", A.nrows);
  spmv(A, x, b);
  return b;
}

}  // namespace kokkos_hpcg
