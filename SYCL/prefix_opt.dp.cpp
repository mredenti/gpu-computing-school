#include <sycl/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <chrono>
#include <cmath>

#define RUNS  25
#define SIZE  2048 // <= BLOCK
#define BLOCK 2048 // max 2048

///////////////////////////////////////
// user functions

void prescan(float * __restrict__ in, float * __restrict__ out,
             float * __restrict__ blockSums, float * __restrict__ temp,
             int totalSize, sycl::nd_item<1> item)
{
    int block_idx = item.get_group(0);
    int block_off = block_idx * BLOCK;
    int thread_idx = item.get_local_id(0);

    // load input in shared mem (one thread fills two)
    int idx0 = 2 * thread_idx + block_off;
    int idx1 = idx0 + 1;
    temp[2 * thread_idx]     = (idx0 < totalSize) ? in[idx0] : 0.0;
    temp[2 * thread_idx + 1] = (idx1 < totalSize) ? in[idx1] : 0.0;

    int offset = 1; // start with offset 1, double at each step of upsweep

    // up sweep phase in local mem
    for (int d = BLOCK >> 1; d > 0; d >>= 1) {
        // wait for shared mem to be consistent
        item.barrier(sycl::access::fence_space::local_space);
        // start with SIZE/2 threads, halve at each iteration
        if (thread_idx < d) {
            int ai = offset * (2 * thread_idx + 1) - 1;
            int bi = offset * (2 * thread_idx + 2) - 1;
            temp[bi] += temp[ai];
        }
        // double offset to add to farther elements
        offset *= 2;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // clear last element
    if (thread_idx == (BLOCK / 2 - 1)) temp[BLOCK - 1] = 0.0;

    // traverse down and build block sums
    for (int d = 1; d < BLOCK; d *= 2) {
        // start with half offset from before
        offset >>= 1;
        // wait for consistency
        item.barrier(sycl::access::fence_space::local_space);
        // start with one thread, double at each iteration
        // up to SIZE/2 threads (starting number)
        if (thread_idx < d) {
            int ai = offset*(2 * thread_idx + 1) - 1;
            int bi = offset*(2 * thread_idx + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    item.barrier(sycl::access::fence_space::local_space);

    // write block sums in blockSums array (using single thread)
    if (thread_idx == (BLOCK / 2 - 1)) {
        int last_idx = BLOCK - 1 + block_off;
        int last_val = (last_idx < totalSize) ? in[last_idx] : 0.0;
        blockSums[block_idx] = temp[BLOCK - 1] + last_val;
    }

    // write output array
    if (idx0 < totalSize) out[idx0] = temp[2 * thread_idx];
    if (idx1 < totalSize) out[idx1] = temp[2 * thread_idx + 1];
}

void addBlockSum(float * __restrict__ inout, float * __restrict__ blockSums,
                 sycl::nd_item<1> item)
{
    int block_idx = item.get_group(0);
    int block_off = block_idx * BLOCK;
    int thread_idx = item.get_local_id(0);

    // get sum of previous blocks
    float mySum = blockSums[block_idx];
    item.barrier();

    // get indices and check bounds
    int idx0 = 2 * thread_idx + block_off;
    int idx1 = idx0 + 1;
    if (idx0 < SIZE * SIZE) inout[idx0] += mySum;
    if (idx1 < SIZE * SIZE) inout[idx1] += mySum;
}

void checkResults(float * __restrict__ out, bool *mismatchFound) {
    auto mismatch = false;
    for (int i = 0; i < SIZE * SIZE; ++i) {
        if (out[i] != i) {
            mismatch = true;
            std::cout << "Expected: " << i << "\n";
            std::cout << "Obtained: " << out[i] << std::endl;
            break;
        }
    }
    *mismatchFound = mismatch;
}

///////////////////////////////////////
// main

int main() {

    // arg checks
    if (SIZE > BLOCK) {
        std::cout << "Invalid argument: SIZE must be <= BLOCK" << std::endl;
        return -1;
    }

    const int totalSize = SIZE * SIZE;

    // define work group size and grid
    const int nblocks = std::ceil((float) totalSize / (float) BLOCK);
    sycl::range grid(nblocks);
    sycl::range block(BLOCK/2);

    // timers
    std::chrono::time_point<std::chrono::steady_clock> start, stop;
    std::vector<float> timings;
    timings.reserve(RUNS);

    // init sycl queue with default device
    sycl::queue Q{sycl::property::queue::in_order()};

    // allocate host arrays
    auto hstIn  = new float[totalSize];
    auto hstOut = new float[totalSize];

    // initialize host arrays
    #pragma omp parallel for
    for (int i = 0; i < totalSize; ++i) {
        hstIn[i]  = 1.0;
    }

    // allocate device arrays
    auto devIn  = sycl::malloc_device<float>(totalSize, Q);
    auto devOut = sycl::malloc_device<float>(totalSize, Q);
    auto preSum = sycl::malloc_device<float>(nblocks, Q);

    // benchmark kernel 100 runs + 5 warmup
    for (int i = 0; i < RUNS; ++i) {

        // copy In to device (reset input)
        Q.memcpy(devIn, hstIn, totalSize * sizeof(float)).wait();

        start = std::chrono::steady_clock::now();

        Q.submit([&](sycl::handler &h) {
            sycl::accessor<float, 1,
                sycl::access::mode::read_write,
                sycl::access::target::local
            > temp(sycl::range<1>(BLOCK), h);

            h.parallel_for(sycl::nd_range<1>(grid * block, block),
                            [=](sycl::nd_item<1> item) {
                                prescan(devIn, devOut, preSum, temp.get_pointer(), totalSize, item);
                            });
        });

        Q.submit([&](sycl::handler &h) {
            sycl::accessor<float, 1,
                sycl::access::mode::read_write,
                sycl::access::target::local
            > temp(sycl::range<1>(BLOCK), h);

            h.parallel_for(sycl::nd_range<1>(block, block),
                            [=](sycl::nd_item<1> item) {
                                prescan(preSum, preSum, devIn, temp.get_pointer(), nblocks, item);
                            });
        });

        Q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::nd_range<1>(grid * block, block),
                            [=](sycl::nd_item<1> item) {
                                addBlockSum(devOut, preSum, item);
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

    // copy back Out and check results
    Q.memcpy(hstOut, devOut, totalSize * sizeof(float)).wait();

    bool mismatchFound = true;
    checkResults(hstOut, &mismatchFound);

    // output kernel info
    auto dev = Q.get_device();
    std::cout << std::endl;
    std::cout << "General info: " << std::endl;
    std::cout << "KERNEL: Optimized Prefix Sum SYCL" << std::endl;
    std::cout << "DEVICE: " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "DATA SIZE: " << SIZE << std::endl;
    std::cout << std::endl;
    std::cout << "Launch parameters:" << std::endl;
    std::cout << "GRID SIZE: (" << grid[0]  << ", 1)" << std::endl;
    std::cout << "WORKGROUP: (" << block[0] << ", 1)" << std::endl;
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
    delete[] hstIn;
    delete[] hstOut;

    sycl::free(devIn, Q);
    sycl::free(devOut, Q);
    sycl::free(preSum, Q);

    std::cout << std::endl;
    return 0;
}
