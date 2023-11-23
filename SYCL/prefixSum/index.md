# Parallel Prefix Sum
This example shows the implementation of a GPU-based prefix sum algorithm. The prefix-sum algorithm takes a vector, and for each entry of its entries computes the sum of all previous entries. 

The serial version of the algorith is extremely easy (it is a simple for loop), but it becomes much more complicated in its parallel version. A very good explanation of the parallel algorithm can be found at [this link](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) 

## Compiling
```shell
# enter the example directory
cd prefixSum

# SYCL CPU
make

# SYCL GPU
make BACKEND=cuda-leo

# CUDA
make TARGET=cuda
```

## Running
```shell
# run CPU version
./prefixSum_sycl_cpu

# run SYCL GPU version
./prefixSum_sycl_cuda-leo

# run CUDA GPU version
./prefixSum_cuda
```

## Do Try it at Home!
If you want to try to implement your own version of this example, in the `extras` folder there are template files for SYCL and CUDA. The following instructions should allow you to compile your code without any problems:
```shell
# create a folder in the directory SYCL
mkdir customKernel

# copy the makefile from any of the other examples
cp globalMatMul/Makefile customKernel/Makefile

# copy the template
cp extras/template.cpp custoKernel/customKernel.cpp

# modify the template as you want
# and compile for the required target of backend
make BACKEND=cuda-leo
```
The templates already contain the code required to compute metrics for your kernel, if you wish to compare with the provided implementation.
