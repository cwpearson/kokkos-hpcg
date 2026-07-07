#include "kokkos_hpcg.hpp"
#include "test_common.hpp"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_matrix_generation() {
  const auto A = kokkos_hpcg::generate_problem(3, 3, 3);
  auto row_length = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.row_length);
  auto diag_slot = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.diag_slot);
  require(A.nrows == 27, "3x3x3 grid has 27 rows");
  require(A.nnz == 343, "3x3x3 grid has expected boundary-aware nonzeros");
  require(A.storage_slots == 27 * kokkos_hpcg::SparseMatrix::max_entries_per_row,
          "storage has one padded 27-wide row per equation");
  require(row_length(0) == 8, "corner row has 8 entries");
  const int center = kokkos_hpcg::linear_index(1, 1, 1, 3, 3);
  require(row_length(center) == 27, "center row has 27 entries");
  require(diag_slot(center) >= 0 && diag_slot(center) < row_length(center),
          "center row has a diagonal slot");
}

void test_vector_kernels() {
  Kokkos::View<double*> x("x", 4);
  Kokkos::View<double*> y("y", 4);
  Kokkos::View<double*> w("w", 4);
  auto xh = Kokkos::create_mirror_view(x);
  auto yh = Kokkos::create_mirror_view(y);
  for (int i = 0; i < 4; ++i) {
    xh(i) = static_cast<double>(i + 1);
    yh(i) = 2.0 * static_cast<double>(i + 1);
  }
  Kokkos::deep_copy(x, xh);
  Kokkos::deep_copy(y, yh);
  require_near(kokkos_hpcg::dot(x, y), 60.0, 1.0e-12, "dot product");
  kokkos_hpcg::waxpby(2.0, x, -1.0, y, w);
  auto wh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
  for (int i = 0; i < 4; ++i) {
    require_near(wh(i), 0.0, 1.0e-12, "waxpby result");
  }
}

void test_end_to_end() {
  kokkos_hpcg::Options options;
  options.nx = 4;
  options.ny = 4;
  options.nz = 4;
  options.max_iterations = 80;
  const auto result = kokkos_hpcg::run(options);
  require(result.valid, "4x4x4 solve validates");
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  try {
    test_matrix_generation();
    test_vector_kernels();
    test_end_to_end();
    Kokkos::finalize();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    Kokkos::finalize();
    return 1;
  }
}
