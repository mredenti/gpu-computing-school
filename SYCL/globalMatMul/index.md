# Global Matrix Multiplication
In this exercise you are going to implement the classic matrix multiplication using SYCL.

The `exercise` folder contains the `globalMatMul.cpp` file, which is a partially filled template. You have to complete it with the code to perform the matrix multiplication, and the code to offload the kernel to the device. The spots that have to be filled are marked with a `//TODO` comment.

The code in the template already initializes the matrices and performs the memory transfers required. Furthermore it checks the results of the multiplication and reports the error in output, together with some execution metrics. 

## Compiling
In order to compile the exercise it is sufficient to run the following commands inside the exercise directory
```shell
# compile for CPU
make

# compile for GPU
make BACKEND=cuda-leo
```

## Running
In order to run the exercise on Leonardo using GPUs you need to request a GPU node
```shell
srun -X -t 10 -N 1 -n 1 --mem=8GB --gres=gpu:1 -p boost_usr_prod -A tra23_advgpu --reservation s_tra_advgpu --qos=qos_lowprio --pty /usr/bin/bash
./globalMatMul_sycl_cuda-leo
```
The CPU version can be run on the login node, but the `SIZE` must be set to ad adequate value, otherwise the execution will be interrupted
```shell
# setting SIZE = 512
# recompiling
./globalMatMul_sycl_cpu
```

## Compiling and running the solution
The folder `solution` contains a solution to the exercise. The commands for compiling and running are the same, except that they must be run in the `solution` folder.

There is also a CUDA version of the solution, that can be compiled and run as follows
```shell
# compile CUDA version
make TARGET=cuda

# run on GPU node
srun -X -t 10 -N 1 -n 1 --mem=8GB --gres=gpu:1 -p boost_usr_prod -A tra23_advgpu --reservation s_tra_advgpu --qos=qos_lowprio --pty /usr/bin/bash
./globalMatMul_cuda
```
