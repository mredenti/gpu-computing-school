#include <sycl/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <chrono>
#include <cmath>

#define RUNS  25
#define SIZE  8192 // use multiple of block
#define BLOCK 32   // use power of 2
#define STEPS 100
#define XSOURCE SIZE/4
#define YSOURCE SIZE/4

///////////////////////////////////////
// user functions

void stencilAdd(float * __restrict__ source, float * __restrict__ target,
                sycl::nd_item<2> item)
{
    int c = item.get_global_id(1);
    int r = item.get_global_id(0);
    int flat_idx = r * SIZE + c;
    float source_val = source[flat_idx];
    // leave source unchanged
    if (c == XSOURCE && r == YSOURCE) {
        target[flat_idx] = source[flat_idx];
    } else {
    // otherwise update heat map
        float temp = 4.0f * source_val;
        temp += (c == 0)        ? source_val : source[flat_idx - 1];
        temp += (c == SIZE - 1) ? source_val : source[flat_idx + 1];
        temp += (r == 0)        ? source_val : source[flat_idx - SIZE];
        temp += (r == SIZE - 1) ? source_val : source[flat_idx + SIZE];
        target[flat_idx] = temp / 8.0;
    }
}

void checkResults(float * __restrict__ map, float min, float max, bool *mismatchFound) {
    bool mismatch = false;
    #pragma omp parallel for
    for (int i = 0; i < SIZE * SIZE; ++i) {
        float val = map[i];
        if (val < min || val > max) {
            #pragma omp critical
            {
                mismatch =  true;
            }
        }
    }
    *mismatchFound = mismatch;
}

///////////////////////////////////////
// main

int main() {

    const int nelems = SIZE * SIZE;
    const float min = 273.0f;
    const float max = 333.0f;

    // define work group size and grid
    sycl::range grid(std::ceil((float) SIZE / BLOCK), std::ceil((float) SIZE / BLOCK));
    sycl::range block(BLOCK, BLOCK);

   // timers
    std::chrono::time_point<std::chrono::steady_clock> start, stop;
    std::vector<float> timings;
    timings.reserve(RUNS);

    // init sycl queue with default device
    sycl::queue Q;

    // allocate host arrays
    auto hstA = new float[nelems];
    auto hstB = new float[nelems];

    // init host arrays
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            if (i == YSOURCE && j == XSOURCE) {
                hstA[i * SIZE + j] = max;
            } else {
                hstA[i * SIZE + j] = min;
            }
            hstB[i * SIZE + j] = -42.0f;
        }
    }

    // allocate device arrays
    auto devA = sycl::malloc_device<float>(nelems, Q);
    auto devB = sycl::malloc_device<float>(nelems, Q);

    // benchmark kernel 100 runs
    for (int i = 0; i < RUNS; ++i) {

        // copy A and B to device (reset inputs)
        Q.memcpy(devA, hstA, nelems * sizeof(float)).wait();
        Q.memcpy(devB, hstB, nelems * sizeof(float)).wait();

        start = std::chrono::steady_clock::now();

        for (int s = 0; s < STEPS; ++s) {
            auto source = devA;
            auto target = devB;
            if (s % 2 == 1) {
                source = devB;
                target = devA;
            }
            Q.submit([&](sycl::handler &h) {
                h.parallel_for(sycl::nd_range<2>(grid * block, block),
                                [=](sycl::nd_item<2> item) {
                                    stencilAdd(source, target, item);
                                });
            });
        }

        Q.wait();
        stop = std::chrono::steady_clock::now();
        timings.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }

    // compute timings
    auto fifth = std::next(timings.begin(), 5);
    auto tavg =  std::accumulate( fifth, timings.end(), 0.0) / (timings.size() - 5);
    auto tmin = *std::min_element(fifth, timings.end());
    auto tmax = *std::max_element(fifth, timings.end());

    // copy back B and check results
    Q.memcpy(hstB, devB, nelems * sizeof(float)).wait();

    bool mismatchFound = true;
    checkResults(hstB, min, max, &mismatchFound);

    // output kernel info
    auto dev = Q.get_device();
    std::cout << std::endl;
    std::cout << "General info: " << std::endl;
    std::cout << "KERNEL: 2D Stencil Heath Map SYCL" << std::endl;
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
    std::cout << "AVERAGE: " << tavg << " ms" << std::endl;
    std::cout << "MINIMUM: " << tmin << " ms" << std::endl;
    std::cout << "MAXIMUM: " << tmax << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "Results check:" << std::endl;
    std::cout << "MISMATCH FOUND: " << mismatchFound << std::endl;

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

    sycl::free(devA, Q);
    sycl::free(devB, Q);

    std::cout << std::endl;
    return 0;
}
