# Tiled Matrix Multiplication
This example shows a (very fancy) implementation of a tiled matrix multiplication algorithm. The performance of this algorithm is significantly increased with respect to the `globalMatMul` case. 

The code is based on [this article](https://siboehm.com/articles/22/CUDA-MMM), and the following [GitHub repository](https://github.com/siboehm/SGEMM_CUDA/tree/master).

## Compiling
```shell
# enter the example directory
cd tiledMatMul

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
./tiledMatMul_sycl_cpu

# run SYCL GPU version
./tiledMatMul_sycl_cuda-leo

# run CUDA GPU version
./tiledMatMul_cuda
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
