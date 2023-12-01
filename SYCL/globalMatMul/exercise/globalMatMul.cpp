#include "common_utils.h"
#include "sycl_utils.h"
#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

constexpr size_t RUNS{   25 };
constexpr size_t SIZE{ 1024 };
constexpr size_t BLOCK{  32 };

namespace chr = std::chrono;

///////////////////////////////////////
// user functions

void globalMatMul(const float* A, const float* B, float* C, const sycl::nd_item<2> item)
{
    // TODO
    // YOUR KERNEL CODE HERE
    
}

///////////////////////////////////////
// main

int main() {

    constexpr size_t nelems{ SIZE * SIZE };

    // define work group size and grid
    const size_t nblocks{ static_cast<size_t>(std::ceil(static_cast<float>(SIZE)/BLOCK)) };
    const sycl::range grid{ nblocks, nblocks };
    const sycl::range block{ BLOCK, BLOCK };

    // init sycl queue with default device
    sycl::queue Q{};

    // print kernel and launch info
    print_kernel_info("Global Matrix Multiplication SYCL", SIZE, Q);
    print_launch_info(grid, block , Q);

    // timers
    chr::time_point<chr::steady_clock> start{}, stop{};
    std::vector<float> timings{};
    timings.reserve(RUNS);

    // allocate host matrices
    float* hstA{ new float[nelems] };
    float* hstB{ new float[nelems] };
    float* hstC{ new float[nelems] };

    // initialize host matrices
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            hstA[i * SIZE + j] =  10.0 * i + j;
            hstB[i * SIZE + j] = (i == j) ? 1.0 : 0.0;
            hstC[i * SIZE + j] = -42.0;
        }
    }

    // allocate device matrices
    float* devA{ sycl::malloc_device<float>(nelems, Q) };
    float* devB{ sycl::malloc_device<float>(nelems, Q) };
    float* devC{ sycl::malloc_device<float>(nelems, Q) };

    // copy A and B to device
    Q.copy<float>(hstA, devA, nelems).wait();
    Q.copy<float>(hstB, devB, nelems).wait();

    // benchmark kernel runs, 5 for warmup
    for (size_t i{}; i < RUNS; ++i) {
        start = chr::steady_clock::now();

        Q.submit([&](sycl::handler &h) {
            
            // TODO
            // CALL YOUR KERNEL HERE

        }).wait();

        stop = chr::steady_clock::now();
        timings.push_back(chr::duration<double, std::milli>(stop - start).count());
    }

    // copy back C and check results
    Q.copy<float>(devC, hstC, nelems).wait();
    
    // check result and print metrics
    check_results(hstA, hstC, SIZE);
    print_metrics(timings);

    // free resources
    delete[] hstA;
    delete[] hstB;
    delete[] hstC;

    sycl::free(devA, Q);
    sycl::free(devB, Q);
    sycl::free(devC, Q);

    return 0;
}
