# Kokkos minimally requires 3.16 right now,
# but your project can set it higher
cmake_minimum_required (VERSION 3.16)

project (KokkosExercisesInnerProduct)

# Look for an installed Kokkos
#find_package(KOKKOS REQUIRED)
include(../../common.cmake)

add_executable(inner_product_cpu inner_product_cpu.cpp)
target_link_libraries(inner_product_cpu Kokkos::kokkos)