This exercise asks you to investigate of the performance of matrix-vector multiplication
$$
    y = Ax, \text{where} A \in \mathbb{R}^{N \times M}
$$


 More precisely, you will witness how performance varies with the shape of the matrix while keeping the overall size fixed.



???+ note "Step 0: Navigate to the `matVecMul` directory"

    ```shell
    cd gpu-computing-sycl-kokkos/KOKKOS/matVecMul/begin
    ```

???+ note "Step 1: Complete the matrix vector multiplication program"

    The program is partially implemented, and you will need to complete the sections marked with `@TASK@`.

???+ note "Step 2: Investigate the performance as you vary the shape of the matrix"

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
    
    Great! We can now turn to running our first Kokkos program [Tutorial 02: Matrix Vector Multiplication](./matVecMul/index.md)