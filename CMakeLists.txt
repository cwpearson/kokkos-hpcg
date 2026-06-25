cmake_minimum_required(VERSION 3.23)

project(kokkos_hpcg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KOKKOS_HPCG_BUILD_TESTS "Build kokkos-hpcg correctness tests" ON)

find_package(Kokkos REQUIRED)

add_library(kokkos_hpcg_core
  src/dot.cpp
  src/mg.cpp
  src/options.cpp
  src/output.cpp
  src/problem.cpp
  src/run.cpp
  src/spmv.cpp
  src/symgs.cpp
  src/waxpby.cpp
)
target_include_directories(kokkos_hpcg_core
  PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(kokkos_hpcg_core PUBLIC Kokkos::kokkos)

add_executable(kokkos-hpcg src/main.cpp)
target_link_libraries(kokkos-hpcg PRIVATE kokkos_hpcg_core)

if(KOKKOS_HPCG_BUILD_TESTS)
  enable_testing()
  add_executable(kokkos-hpcg-tests tests/test_kernels.cpp)
  target_link_libraries(kokkos-hpcg-tests PRIVATE kokkos_hpcg_core)
  add_test(NAME kokkos-hpcg-tests COMMAND kokkos-hpcg-tests)
endif()
