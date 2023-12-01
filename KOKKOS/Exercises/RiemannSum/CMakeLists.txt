# Kokkos minimally requires 3.16 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.16)

project(TestingKokkos CXX)

# Look for an installed Kokkos
find_package(Kokkos REQUIRED)

# Look for an installed Kokkos but force using the compiler launcher
# to ensure that targets depending on Kokkos use the same compiler
# as when kokkos was installed, e.g. if kokkos was built with
# g++ and the CMAKE_CXX_COMPILER=clang++ then example_with_kokkos
# will be compiled and linked with g++ whereas example_no_kokkos
# will be compiled and linked with clang++
#find_package(Kokkos REQUIRED COMPONENTS launch_compiler)


add_executable(riemann_sum_functor_explicit riemann_sum_functor_explicit.cpp)

target_link_libraries(riemann_sum_functor_explicit Kokkos::kokkos)
