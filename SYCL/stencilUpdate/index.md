# 2D Stencil Heat Map Update
In this exercise we will write a simple code that computes a stencil update to a 2D heat map. There is an heat source placed at `( SIZE/4, SIZE/4)`, where `SIZE` is the dimension of the domain we are considering (the number sf stencil locations per side, assuming a square stencil).

The update conditions are as follows:
- if we are at the source, i.e. `x,y = source_x, source_y` the value of the heat map is unchanged
- in any other point `(i,j)` it should be updated as follows:
```
map[i,j] = (4 * map[i,j] + map[i-1,j] + map[i+1,j] + map[i,j-1] + map[i,j+1]) /8
```
- if the point is on the border, the value of the map in the direction of the border is substituted with the value at the point:
```
if (i==0)      map[i-1,j] => map[i,j]
if (i==SIZE-1) map[i+1,j] => map[i,j]
```

The file `exercise/stencilUpdate.cpp` contains a partially compiled template for the exercise. You only need to fill the code for the kernel function, and the code to call it from host. The spots that have to be filled are marked with a `//TODO` comment.

The templated code already sets up the memory on host and device, and performs the memory tranfers. Furthermore, it checks the result at the end, to ensure that the final results are consistent. The result of the check, together with some execution metrics are printed in the program output.

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
./stencilUpdate_sycl_cuda-leo
```
The CPU version can be run on the login node, but the `SIZE` must be set to ad adequate value, otherwise the execution will be interrupted
```shell
# setting SIZE = 512
# recompiling
./stencilUpdate_sycl_cpu
```

## Compiling and running the solution
The folder `solution` contains a solution to the exercise. The commands for compiling and running are the same, except that they must be run in the `solution` folder.

There is also a CUDA version of the solution, that can be compiled and run as follows
```shell
# compile CUDA version
make TARGET=cuda

# run on GPU node
srun -X -t 10 -N 1 -n 1 --mem=8GB --gres=gpu:1 -p boost_usr_prod -A tra23_advgpu --reservation s_tra_advgpu --qos=qos_lowprio --pty /usr/bin/bash
./stencilUpdate_cuda
```
