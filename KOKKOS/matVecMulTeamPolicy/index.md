This exercise asks you to improve the performance of the matrix-vector multiplication for rectangular matrices using hierarchical parallelism.

???+ note "Step 0: Navigate to the `matVecMulTeamPolicy` directory"

    ```shell
    cd gpu-computing-sycl-kokkos/KOKKOS/matVecMulTeamPolicy/begin
    ```

???+ note "Step 1: Complete the matrix vector multiplication program"

    The starting point is the matrix vector multiplication program that you completed previously. 
    Now, you have to re-write it using Team and TeamThreadRange policies. The section of interest is marked by `TASK@`.

???+ note "Step 2: Compare the performance to the previous version of the matrix vector multiplication kernel"

    Compile with CUDA support

    ```shell
    make BACKEND=cuda
    ```

    Launch an interactive session and run with the same parameters as before

    ```shell
    srun -N 1 --ntasks-per-node=1 --cpus-per-task=1 -p boost_usr_prod --gres=gpu:1 -A <your_project_name> --time=00:10:00 --pty /bin/bash
    ```
    ```shell
    ./matVecMulTeamPolicy.cudax -N 12
    ```
    ```shell
    ./matVecMulTeamPolicy.cudax -N 5
    ```

    ???+ question

        Do you note an improvement in the execution time and/or the bandwidth?

    ???+ question

        Think about memory access patterns. Do you think the default layout is optimal for this version of the matrix vector multiplication kernel?
        If not, can you explain why?

        ???+ example "Task"
            Try change the layout from `Kokkos::LayoutLeft` to `Kokkos::LayoutRight`. 
            Then, re-compile the program 
            
            ```shell
            make BACKEND=cuda
            ```

            ???+ danger 
                Make sure to be running the example on a compute node
            
            and run with the same parameters 

            ```shell
            ./matVecMulTeamPolicy.cudax -N 12
            ```
            ```shell
            ./matVecMulTeamPolicy.cudax -N 5
            ```

            ???+ question

                Do you observe any improvement?

        