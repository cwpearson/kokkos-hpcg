add_executable(kokkos-hpcg-tests test_kernels.cpp)
target_link_libraries(kokkos-hpcg-tests PRIVATE kokkos_hpcg_core)
add_test(NAME kokkos-hpcg-tests COMMAND kokkos-hpcg-tests)

add_executable(kokkos-hpcg-test-spmv test_spmv.cpp)
target_link_libraries(kokkos-hpcg-test-spmv PRIVATE kokkos_hpcg_core)
add_test(NAME kokkos-hpcg-test-spmv COMMAND kokkos-hpcg-test-spmv)

add_executable(kokkos-hpcg-test-restrict-injection test_restrict_injection.cpp)
target_link_libraries(kokkos-hpcg-test-restrict-injection PRIVATE kokkos_hpcg_core)
add_test(NAME kokkos-hpcg-test-restrict-injection
  COMMAND kokkos-hpcg-test-restrict-injection)

add_executable(kokkos-hpcg-test-prolongate-and-add test_prolongate_and_add.cpp)
target_link_libraries(kokkos-hpcg-test-prolongate-and-add PRIVATE kokkos_hpcg_core)
add_test(NAME kokkos-hpcg-test-prolongate-and-add
  COMMAND kokkos-hpcg-test-prolongate-and-add)
