This guide will walk you through the process of compiling and running a vector addition program using Kokkos. 
The program demonstrates basic usage of Kokkos for heterogeneous computing, including the use of different execution backends.

???+ note "Step 0: Navigate to the `vectorAdd` directory"

    ```shell
    cd gpu-computing-sycl-kokkos/KOKKOS/vectorAdd/begin
    ```

???+ note "Step 1: Complete the vector addition program"

    The program is partially implemented, and you will need to complete the sections marked with `@TASK@`.

???+ note "Step 2: Compile and Run"

    You can compile the program using the Makefile provided. 

    ??? example "Serial compilation and execution"
    
        By default, the Makefile compiles for serial execution. To compile the program, simply run:

        ```shell
        make
        ```

        The compilation step should give you yield an executable named `vecAdd.serialx`

        To run the serial Kokkos version of the vector addition program launch an interactive session and run the program 

        ```shell
        srun -N 1 --ntasks-per-node=1 --cpus-per-task=1 -p boost_usr_prod --gres=gpu:0 -A <your_project_name> --time=00:10:00 --pty /bin/bash
        ./vecAdd.serialx
        ```

    ??? example "Cuda compilation and execution"

        Kokkos supports various backends such as CUDA for GPUs. To compile the program for a specific backend, you can pass the BACKEND variable to Make. 
        
        For example, to compile with CUDA support, use:

        ```shell
        make BACKEND=cuda
        ```

        ???+ question

            What do you note different about the compilation w.r.t. the serial compilation?

        The compilation step should give you yield an executable named `vecAdd.cudax`

        Launch an interactive session and run the program 

        ```shell
        srun -N 1 --ntasks-per-node=1 --cpus-per-task=1 -p boost_usr_prod --gres=gpu:1 -A <your_project_name> --time=00:10:00 --pty /bin/bash
        ./vecAdd.cudax
        ```


??? tip "Optional: Comparing performances against Cuda/SYCL version"

    - Implement your own version of vector addition in CUDA (perhaps retrieve one from the exercises during the school)
    - Time the execution in a similar manner as for the Kokkos vector program and compare the execution times

    **You may just compare performance to a SYCL version as well**


??? success "Next"
    
    Great! We can now turn to our next exercise [Tutorial 02: Matrix Vector Multiplication](../matVecMul/index.md)
