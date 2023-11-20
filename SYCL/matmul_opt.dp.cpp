#include <sycl/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <chrono>
#include <cmath>

#define RUNS 25
#define SIZE 8192
#define TILE 16
#define VECT 4

///////////////////////////////////////
// user functions

void matmulOptGPU(float * __restrict__ A, float * __restrict__ B,
                  float * __restrict__ C, float * __restrict__ As,
                  sycl::nd_item<2> item)
{
    // workgroup index
    sycl::id<2> group_id = item.get_group().get_group_id();
    int bx = group_id[0];
    int by = group_id[1];

    // workitem index in group
    sycl::id<2> local_id = item.get_local_id();
    int tx = local_id[0];
    int ty = local_id[1];

    // vector to accumulate results
    float cv[TILE] = { 0 };

    int aBegin = SIZE * TILE * by;
    int aEnd   = aBegin + SIZE - 1;
    int aStep  = TILE;

    int bBegin = TILE * VECT * bx;
    int bStep  = TILE * SIZE;

    for (int a = aBegin, b = bBegin; a <= aEnd; a+=aStep, b+=bStep) {
        for (int i = 0; i < TILE/VECT; ++i) {
            As[(i * VECT + ty) + TILE * tx] = A[a + SIZE * (i * VECT + ty) + tx];
        }
        item.barrier(sycl::access::fence_space::local_space);

        float *ap = &As[0];
        float *bp = &B[b + TILE * ty + tx];

        for (int i = 0; i < TILE; ++i) {
            float bv = bp[0] ;
            for (int j = 0; j < TILE; ++j) {
                cv[j] += ap[j] * bv;
            }
            ap += TILE;
            bp += SIZE;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    int c = (SIZE * TILE * by) + (TILE * VECT * bx);
    c += TILE * ty + tx;
    for (int i = 0; i < TILE; ++i) {
        C[c] = cv[i];
        c += SIZE;
    }
}

void checkResults(float *R, float *C, float *totErr, float *maxErr) {
    float diff, tot = 0.0, max = 0.0;
    // #pragma omp parallel for collapse(2) shared(max) reduction(+:tot)
    for (int i = 0; i < SIZE * SIZE; ++i) {
        diff = fabs(C[i] - R[i]);
        tot += diff;
        // #pragma omp critical
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
    sycl::range grid(SIZE / (TILE * VECT), SIZE / TILE);
    sycl::range block(TILE, VECT);

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
            sycl::accessor<float,  1,
                sycl::access::mode::read_write,
                sycl::access::target::local> As {sycl::range<1>(TILE * TILE), h};

            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                            [=](sycl::nd_item<2> item) {
                                matmulOptGPU(devA, devB, devC, As.get_pointer(), item);
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

    // output kernel info
    auto dev = Q.get_device();
    std::cout << std::endl;
    std::cout << "General info: " << std::endl;
    std::cout << "KERNEL: Optimized Matrix Multiplication SYCL" << std::endl;
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
    free(hstA);
    free(hstB);
    free(hstC);

    sycl::free(devA, Q);
    sycl::free(devB, Q);
    sycl::free(devC, Q);

    std::cout << std::endl;
    return 0;
}
