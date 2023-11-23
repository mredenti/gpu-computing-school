# Setup for Running the Exercises

## Installing SYCL
In order to compile the code for the exercises we need to have SYCL installed on our machine. There are various SYCL distributions that can be used for this purpose, but it is definitely easier if we stick with one of the main three: 

- [Data Parallel C++ (DPC++)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/data-parallel-c-plus-plus.html): Intel's implentation of SYCL, distributed as part of their oneAPI Base Toolkit.  
- [AdaptiveCpp](https://adaptivecpp.github.io/): An independent SYCL implementation, supporting a multitude of compilation flows. See [here](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md) for more info.
- [ComputeCpp](https://developer.codeplay.com/products/computecpp/ce/home/): Based on SYCL 1.2.1 (not SYCL2020), developed by Codeplay. Has native support for NVIDIA GPUs.

Detailed instructions on how to install them can be found on the corresponding product pages (linked). DPC++ and ComputeCpp are provided as binaries (in the form of a SYCL-enabled compiler), while AdaptiveCpp has to be compiled from source.

Intel's DPC++ is the easiest to get (at least for SYCL2020). AdaptiveCpp is very flexible and has more backends available, but it is also not-so-easy to compile, depending on the compilation flow you want, and the software dependencies it requires.

## Tutorial environment on Leonardo HPC Cluster
If you are compiling these exercises on Leonardo, you can use the `setenv-leo.sh` script in the `SYCL` folder inside the repository to set up the environment required for the compiler to work.

The compiler is Intel's `icpx` found in the oneAPI Toolkit. In order to make it work also on NVIDIA GPUs (natively it only works with Intel's GPUs) there is a free plugin maintained by Codeplay. The page for the plugin, with links for the download and a detailed guid on how to use it, can be found at the [following link](https://developer.codeplay.com/products/oneapi/nvidia/home/).

The following commands will clone the repo and acivate the environment
```shell
# clone the repository
git clone https://gitlab.hpc.cineca.it/amasini0/gpu-computing-sycl-kokkos.git

# enter the SYCL folder
cd gpu-computing-sycl-kokkos/SYCL
cd SYCL

# source the setenv file
. setenv-leo.sh
```

If you did everything correclty you should see a message on the terminal that says that a module has been loaded. This script loads the `oneapi/compiler/latest` module and its dependencies, together with the `nvhpc/23.5` module, required to compile the CUDA versions of the exercises that we will use as reference for the performance.
Furthermore, the script also exports an environment variable required to allow the SYCL compiler to find the required CUDA libraries to run the code on the GPUs (this is only required since the CUDA Toolkit on the cluster is installed on non-standard locations.

## Compiling on Leonardo
The exercise can be compiled simply by running one of the following three commands in the corresponding folder.

- `make`: to compile the SYCL code using CPU as backend (executes only on CPU)
- `make BACKEND=cuda-leo`: to compile the SYCL code using CUDA backend (executes on Leonardo's GPUs).
- `make TARGET=cuda`: to compile the CUDA code (executed on Leonardo's GPUs).

For example, to compile the `globalMatMul` SYCL example, and run it on GPU, we can do the following:
```shell
cd globalMatMul
make BACKEND=cuda-leo
```
We should see the message `building globalMatMul_sycl_cuda-leo ...` appear on the terminal.

## Compiling on other machines
There is also another compilation target available, namely `make BACKEND=cuda`, which targets generic CUDA backend, but it requires all the software to be installed in standard locations, and targets the CUDA sm_50 architecture by default, to enable the code to run on the majority of NVIDIA GPUs. 

This target does not work on Leonardo (thus it has its own), but it can be useful when installing SYCL on your laptop (only Linux, only using distribution packages). This is not tested however.

If these targets are not suitable for the your target machine, it suffices to add a fel lines to the `Makefile.in`, and set the required compilation flags for the SYCL compiler.
