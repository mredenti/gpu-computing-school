cmake_minimum_required (VERSION 3.10)
project (SYCL_benchmarks LANGUAGES CXX)


set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG " -g -Wall -Wextra")
find_package(OpenSYCL REQUIRED)
 
add_executable(matmul matmul.dp.cpp)
add_sycl_to_target(TARGET matmul)

add_executable(matmul_opt matmul_opt.dp.cpp)
add_sycl_to_target(TARGET matmul_opt)

add_executable(prefix prefix.dp.cpp)
add_sycl_to_target(TARGET prefix)

add_executable(prefix_opt prefix_opt.dp.cpp)
add_sycl_to_target(TARGET prefix_opt)

add_executable(stencil stencil.dp.cpp)
add_sycl_to_target(TARGET stencil)
