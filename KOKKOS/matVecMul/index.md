This exercise asks you to investigate of the performance of matrix-vector multiplication.
More precisely, you will witness how performance varies with the shape of the matrix while keeping the overall size fixed.


???+ note "Step 0: Navigate to the `matVecMul` directory"

    ```shell
    cd gpu-computing-sycl-kokkos/KOKKOS/matVecMul/begin
    ```

???+ note "Step 1: Complete the matrix vector multiplication program"

    The program is partially implemented, and you will need to complete the sections marked with `@TASK@`.

???+ note "Step 2: Investigate the performance as you vary the shape of the matrix"

    You can compile the program using the Makefile provided. 


    Compile with CUDA support

    ```shell
    make BACKEND=cuda
    ```

    Launch an interactive session and run the matrix vector multiplication varying the number of rows N while keeping constant the size S of the matrix

    ```shell
    srun -X -t 10 -N 1 -n 1 --mem=8GB --gres=gpu:1 -p boost_usr_prod -A tra23_advgpu --reservation s_tra_advgpu --qos=qos_lowprio --pty /usr/bin/bash
    ```
    ```shell
    ./matVecMul.cudax -N 12
    ```
    ```shell
    ./matVecMul.cudax -N 5
    ```

    ???+ question

        What do you note about the execution time and/or the bandwidth?

        ???+ example "Task"
            Take a note of the bandwidths or the execution time. You will compare them to the ones obtained in the next exercise when you will re-write the Matrix Vector Multiplication kernel using team policies

??? success "Next"
    
    Great! We can now turn to the next exercise [Tutorial 03: Matrix Vector Multiplication + Team Policy](../matVecMulTeamPolicy/index.md)