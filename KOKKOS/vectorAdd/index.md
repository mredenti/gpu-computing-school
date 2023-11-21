This guide will walk you through the process of compiling and running a vector addition program using Kokkos. 
The program demonstrates basic usage of Kokkos for heterogeneous computing, including the use of different execution backends.

???+ note "Step 0: Navigate to the `vectorAdd` directory"

    ```shell
    gpu-computing-sycl-kokkos/KOKKOS/vectorAdd/begin
    ```

???+ note "Step 1: Complete vector addition program"

    The program is partially implemented, and you will need to complete the sections marked with `@TASK@`.
    These comments indicate parts of the code that you need to implement. 

???+ note "Step 2: Compilation with Make"

    You can compile the program using the Makefile provided. By default, the Makefile compiles for serial execution. To compile the program, simply run:

    ```shell
    make
    ```
    ???+ note 

        Take a note of the compiler being used

    Kokkos supports various backends such as CUDA for GPUs. To compile the program for a specific backend, you can pass the BACKEND variable to Make. For example, to compile with CUDA support, use:

    ```shell
    make BACKEND=cuda
    ```

    ???+ question
        
        What do you note about the compilation?

???+ tip "Step 3: Comparing performances"

    - Implement your own version of vector addition in CUDA (perhaps retrieve one from the exercises during the school)
    - Time the execution in a similar manner as for the Kokkos vector program and compare the execution times

    **You may just compare performance to a SYCL version as well**
