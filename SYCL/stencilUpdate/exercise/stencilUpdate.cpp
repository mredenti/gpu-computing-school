#include "common_utils.h"
#include "sycl_utils.h"
#include <sycl/sycl.hpp>
#include <chrono>
#include <vector>

constexpr size_t RUNS{   25 };
constexpr size_t SIZE{ 1024 }; // use multiple of block
constexpr size_t BLOCK{  32 }; // use power of 2
constexpr size_t STEPS{ 100 };
constexpr size_t XSOURCE{ SIZE/4 };
constexpr size_t YSOURCE{ SIZE/4 };

namespace chr = std::chrono;

///////////////////////////////////////
// user functions

void stencilAdd(const float* source, float* target, const sycl::nd_item<2> item)
{
    // TODO
    // YOUR KERNEL CODE HERE
}

///////////////////////////////////////
// main

int main() {

    constexpr size_t nelems{ SIZE * SIZE };
    constexpr float min{ 273.0f };
    constexpr float max{ 333.0f };

    // define work group size and grid
    const size_t nblocks{ static_cast<size_t>(std::ceil(static_cast<float>(SIZE)/BLOCK)) };
    const sycl::range grid{ nblocks, nblocks };
    const sycl::range block{ BLOCK, BLOCK };;

    // init sycl queue with default device
    sycl::queue Q{};

    // print kernel and launch info
    print_kernel_info("2D Stencil Heat Map SYCL", SIZE, Q);
    print_launch_info(grid, block, Q);
    
    // timers
    chr::time_point<chr::steady_clock> start, stop;
    std::vector<float> timings{};
    timings.reserve(RUNS);

    // allocate host arrays
    float* hstA{ new float[nelems] };
    float* hstB{ new float[nelems] };

    // init host arrays
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < SIZE; ++i) {
        for (size_t j = 0; j < SIZE; ++j) {
            if (i == YSOURCE && j == XSOURCE) {
                hstA[i * SIZE + j] = max;
            } else {
                hstA[i * SIZE + j] = min;
            }
            hstB[i * SIZE + j] = -42.0f;
        }
    }

    // allocate device arrays
    float* devA{ sycl::malloc_device<float>(nelems, Q) };
    float* devB{ sycl::malloc_device<float>(nelems, Q) };

    // benchmark kernel runs, 5 for warmup
    for (size_t i{}; i < RUNS; ++i) {

        // copy A and B to device (reset inputs)
        Q.copy<float>(hstA, devA, nelems).wait();
        Q.copy<float>(hstB, devB, nelems).wait();

        start = chr::steady_clock::now();

        for (size_t s{}; s < STEPS; ++s) {
            float* source{ devA };
            float* target{ devB };
            // swap old and updated map
            if (s % 2 == 1) {
                target = devA;
                source = devB;
            }
            Q.submit([&](sycl::handler &h) {
            
                // TODO
                // CALL GPU KERNEL

            }).wait();
        }

        stop = chr::steady_clock::now();
        timings.push_back(chr::duration<double, std::milli>(stop - start).count());
    }

    // copy back B
    Q.copy<float>(devB, hstB, nelems).wait();

    // check results and print metrics
    check_results(hstB, SIZE, min, max);
    print_metrics(timings);

    // free resources
    delete[] hstA;
    delete[] hstB;

    sycl::free(devA, Q);
    sycl::free(devB, Q);

    return 0;
}
