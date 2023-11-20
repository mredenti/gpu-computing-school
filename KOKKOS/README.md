### GPU computing school: todo list
- [ ] Building Kokkos inline on M100 for Serial and Cuda execution spaces (+openMP)
- [ ] Building Kokkos with CMake and then linking the scripts

- [ ] Matrix multiplication: kokkos implementation
- [ ] Matrix multiplication: report results and compare to SYCL (implementation dependent: true dat)
- [ ] Array access or reduction as in the example - play with host and cuda execution/memory space
- [ ] Array access: play around with layout on different execution spaces - compare performances
- [x] example of a done task

It can be useful also for the presentation to define a typical program/an abstraction of the typical workflow in scientific applications:
1. Allocation on host (think I/O operations)
2. **Copy to device for offloading computations on the accellerators**
3. **"Kernel" execution**
4. Copy back to host

Note: the highlighted items require the most care

# Instructions on M100


# Building Kokkos (v4.0.00) on M100
module load hp
