#include "common_utils.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <chrono>
#include <vector>

constexpr size_t RUNS{   25 };
constexpr size_t SIZE{ 1024 };

namespace chr = std::chrono;

///////////////////////////////////////
// main

int main() {

    constexpr size_t nelems{ SIZE * SIZE };

    // define thread block size and grid
    const dim3 grid{ SIZE / (TILE*VECT), SIZE / TILE };
    const dim3 block{ TILE, VECT };

    // print kernel and launch info
    print_kernel_info("CUSTOM KERNEL", SIZE, 0);
    print_launch_info(grid, block, 0);

    // timers
    chr::time_point<chr::steady_clock> start{}, stop{};
    cudaEvent_t kernelEnd;
    cudaEventCreate(&kernelEnd);
    std::vector<float> timings;{}
    timings.reserve(RUNS);

    // allocate host memory

    // initialize host memory

    // allocate device memory

    // copy memory from host to device

    // benchmark kernel runs, 5 for warmup
    for (size_t i{}; i < RUNS; ++i) {
        start = chr::steady_clock::now();

        // YOUR CUSTOM KERNEL<<< >>>

        cudaEventRecord(kernelEnd, 0);
        cudaEventSynchronize(kernelEnd);
        stop = chr::steady_clock::now();
        timings.push_back(chr::duration<double, std::milli>(stop - start).count());
    }

    // copy results back to host

    // check results

    // print metrics
    print_metrics(timings);

    // free resources
    cudaEventDestroy(kernelEnd);

    return 0;
}
