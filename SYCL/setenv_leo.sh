# The following file sets the environment to enable compilation on LEONARDO

# Add OneAPI compiler modules to modulepath
export MODULEPATH=/leonardo/pub/userinternal/amasini0/modulefiles/oneapi:$MODULEPATH

# Load required modules
module load compiler/latest # for SYCL (.cpp files)
module load nvhpc/23.5      # for CUDA (.cu files)

# Environment variable to locate specific CUDA required for SYCL compilation
# (only if cuda libraries are in non-standard location)
export CUDA_ROOT=/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.5-pdmwq3k5perrhdqyrv2hspv4poqrb2dr/Linux_x86_64/2023/cuda
