#include "kokkos_hpcg.hpp"
#include "spmv.hpp"
#include "test_common.hpp"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

template <class Function>
void require_invalid_argument(Function &&function, std::string_view message) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

void test_spmv() {
  const auto A = kokkos_hpcg::generate_problem(2, 2, 2);
  Kokkos::View<double *> x("x", A.nrows);
  Kokkos::View<double *> y("y", A.nrows);
  Kokkos::deep_copy(x, 1.0);

  kokkos_hpcg::spmv(A, x, y);

  const auto y_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  for (int row = 0; row < A.nrows; ++row) {
    require_near(y_host(row), 19.0, 1.0e-12, "2x2x2 SpMV result");
  }
}

void test_spmv_rejects_invalid_vector_extents() {
  const auto A = kokkos_hpcg::generate_problem(2, 2, 2);
  Kokkos::View<double *> x("x", A.nrows);
  Kokkos::View<double *> short_y("short_y", A.nrows - 1);
  Kokkos::View<double *> oversized_x("oversized_x", A.nrows + 1);
  Kokkos::View<double *> oversized_y("oversized_y", A.nrows + 1);

  require_invalid_argument([&] { kokkos_hpcg::spmv(A, x, short_y); },
                           "SpMV accepts mismatched vector extents");
  require_invalid_argument(
      [&] { kokkos_hpcg::spmv(A, oversized_x, oversized_y); },
      "SpMV accepts a vector extent that does not match the matrix");
}

} // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  try {
    test_spmv();
    test_spmv_rejects_invalid_vector_extents();
    Kokkos::finalize();
    std::cout << "test passed" << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "test failure: " << error.what() << '\n';
    Kokkos::finalize();
    return 1;
  }
}
