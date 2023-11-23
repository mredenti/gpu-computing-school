#include "common_utils.h"
#include "sycl_utils.h"
#include <sycl/sycl.hpp>
#include <chrono>
#include <vector>

constexpr size_t SIZE{ 1024 };
constexpr size_t RUNS{   25 };
namespace chr = std::chrono;

///////////////////////////////////////
// main

int main() {

    constexpr size_t nelems{ SIZE * SIZE };

    // define work group size and grid
    const sycl::range grid{   };
    const sycl::range block{  };

    // init sycl queue with default device
    sycl::queue Q{};

    // print kernel and launch info
    print_kernel_info("CUSTOM KERNEL", SIZE, Q);
    print_launch_info(grid, block, Q);

    // timers
    chr::time_point<chr::steady_clock> start{}, stop{};
    std::vector<float> timings{};
    timings.reserve(RUNS);

    // allocate host memory

    // initialize host memory

    // allocate device memory

    // copy data from host to device

    // benchmark kernel runs, 5 for warmup
    for (size_t i{}; i < RUNS; ++i) {
        start = chr::steady_clock::now();

        Q.submit([&](sycl::handler &h) {
            //
            // YOUR CUSTOM KERNEL CODE 
            //
        }).wait();

        stop = chr::steady_clock::now();
        timings.push_back(chr::duration<double, std::milli>(stop - start).count());
    }

    // copy results back to host

    // check results 

    // print metrics
    print_metrics(timings);

    // free resources

    return 0;
}
