# Path to the Kokkos library installation
KOKKOS_PATH ?= $(KOKKOS_PATH)

# Automatically collect all .cpp source files in the current directory
SRC = $(wildcard *.cpp)

# Set the C++ standard for Kokkos compilation to C++17
KOKKOS_CXX_STANDARD=c++17

# Default OpenMP flags, can be overridden
OFFLOAD_FLAGS ?= -fopenmp


# Backend-specific settings
# For CUDA backend
ifeq ($(BACKEND), cuda)
    # Default architecture setting, can be overridden
    # Default is set for Leonardo (Ampere80 architecture)
    arch ?=
    KOKKOS_ARCH ?= Ampere80 
    # Override Kokkos architecture for NVIDIA V100 GPUs (G100)
    ifeq ($(arch), V100)
        KOKKOS_ARCH=Volta70
    endif
    # Use nvcc_wrapper for CUDA compilation and enable debugging
    CXX = $(KOKKOS_PATH)/bin/nvcc_wrapper
    # Enable lambda expressions in CUDA
    KOKKOS_CUDA_OPTIONS += "enable_lambda"
    # Set Kokkos to use CUDA devices
    KOKKOS_DEVICES = "Cuda"

# For OpenMP Target backend
else ifeq ($(BACKEND), ompt)
    # Set Kokkos to use OpenMP Target devices
    KOKKOS_DEVICES = "OpenMPTarget"
    # Additional options and flags for OpenMP Target
    KOKKOS_OPTIONS += KOKKOS_INTERNAL_DISABLE_DESUL_ATOMICS
    # Compiler settings for OpenMP Target
    CXX = nvc++
    OFFLOAD_FLAGS = -mp=gpu -gpu=cc80 -Minfo=all

# For OpenMP backend
else ifeq ($(BACKEND), omp)
    # Set Kokkos to use OpenMP
    KOKKOS_DEVICES = "OpenMP"
    # Use g++ for OpenMP compilation
    CXX = g++

# Default to Serial backend if no backend is specified
else
    BACKEND=serial
    KOKKOS_DEVICES = "Serial"
    CXX = g++
endif


