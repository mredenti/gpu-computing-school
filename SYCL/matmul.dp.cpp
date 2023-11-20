#include <sycl/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <chrono>
#include <cmath>

#define RUNS  25
#define SIZE  8192
#define BLOCK 32

///////////////////////////////////////
// user functions

void matmulGPU(float * __restrict__ A, float * __restrict__ B,
               float * __restrict__ C, sycl::nd_item<2> item)
{
    sycl::id<2> id = item.get_global_id();
    int row = id[0];
    int col = id[1];

    if ((row >= SIZE) || (col >= SIZE)) return;

    float c = 0.0;
    for (int k = 0; k < SIZE; ++k) {
        c += A[row * SIZE + k] * B[k * SIZE + col];
    }
    C[row * SIZE + col] = c;
}

void checkResults(float *R, float *C, float *totErr, float *maxErr) {
    float diff, tot = 0.0, max = 0.0;
    #pragma omp parallel for shared(max) reduction(+:tot)
    for (int i = 0; i < SIZE * SIZE; ++i) {
        diff = fabs(C[i] - R[i]);
        tot += diff;
        #pragma omp critical
        {
            if (diff > max) max = diff;
        }
    }
    *totErr = tot;
    *maxErr = max;
}

///////////////////////////////////////
// main

int main() {

    const size_t nelems = SIZE * SIZE;

    // define work group size and grid
    sycl::range grid(std::ceil((float) SIZE / BLOCK), std::ceil((float) SIZE / BLOCK));
    sycl::range block(BLOCK, BLOCK);

    // timers
    std::chrono::time_point<std::chrono::steady_clock> start, stop;
    std::vector<float> timings;
    timings.reserve(RUNS);

    // init sycl queue with default device
    sycl::queue Q;

    // allocate host matrices
    auto hstA = new float[nelems];
    auto hstB = new float[nelems];
    auto hstC = new float[nelems];

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
    auto devA = sycl::malloc_device<float>(nelems, Q);
    auto devB = sycl::malloc_device<float>(nelems, Q);
    auto devC = sycl::malloc_device<float>(nelems, Q);

    // copy A and B to device
    Q.memcpy(devA, hstA, nelems * sizeof(float)).wait();
    Q.memcpy(devB, hstB, nelems * sizeof(float)).wait();

    // benchmark kernel 100 runs + 5 warmup
    for (int i = 0; i < RUNS; ++i) {
        start = std::chrono::steady_clock::now();

        Q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                            [=](sycl::nd_item<2> item) {
                                matmulGPU(devA, devB, devC, item);
                            });
        });

        Q.wait();
        stop = std::chrono::steady_clock::now();
        timings.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }

    // compute timings
    auto fifth = std::next(timings.begin(), 5);
    auto avg =  std::accumulate( fifth, timings.end(), 0.0) / (timings.size() - 5);
    auto min = *std::min_element(fifth, timings.end());
    auto max = *std::max_element(fifth, timings.end());

    // copy back C and check results
    Q.memcpy(hstC, devC, nelems * sizeof(float)).wait();

    float totErr = -1.0;
    float maxErr = -1.0;
    checkResults(hstA, hstC, &totErr, &maxErr);

    /// output kernel info
    auto dev = Q.get_device();
    std::cout << std::endl;
    std::cout << "General info: " << std::endl;
    std::cout << "KERNEL: Simple Matrix Multiplication SYCL" << std::endl;
    std::cout << "DEVICE: " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "DATA SIZE: " << SIZE << std::endl;
    std::cout << std::endl;
    std::cout << "Launch parameters:" << std::endl;
    std::cout << "GRID SIZE: (" << grid[0]  << ", " << grid[1]  << ")" << std::endl;
    std::cout << "WORKGROUP: (" << block[0] << ", " << block[1] << ")" << std::endl;
    std::cout << "SUBGROUPS: {";
    auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    for (auto size: sg_sizes) std::cout << size << " ";
    std::cout << "}" << std::endl;
    std::cout << std::endl;
    std::cout << "Host Metrics:" << std::endl;
    std::cout << "AVERAGE: " << avg << " ms" << std::endl;
    std::cout << "MINIMUM: " << min << " ms" << std::endl;
    std::cout << "MAXIMUM: " << max << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "Results check:" << std::endl;
    std::cout << "ACC. ERROR: " << totErr << std::endl;
    std::cout << "MAX. ERROR: " << maxErr << std::endl;

    // report first 10 runs
    std::cout << std::endl;
    std::cout << "WARM UP + FIRST 5:" << std::endl;
    for (int i = 0; i < 5; ++i)
        std::cout << timings.at(i) << std::endl;
    std::cout << "--------" << std::endl;
    for (int i = 5; i < 10; ++i)
        std::cout << timings.at(i) << std::endl;

    // free resources
    delete[] hstA;
    delete[] hstB;
    delete[] hstC;

    sycl::free(devA, Q);
    sycl::free(devB, Q);
    sycl::free(devC, Q);

    std::cout << std::endl;
    return 0;
}
