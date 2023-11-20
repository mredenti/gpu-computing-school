# Building Kokkos inline 

For the tutorial, we will compile our Kokkos programs via a Makefile while **building Kokkos inline**. This allows us to easily swap between different default execution spaces and memory spaces.


???+ note "Instructions: Cloning Kokkos Core repository"

    **Change into your work area on Leonardo...**
    ```shell
    cd $WORK
    ```

    **...define Kokkos release version/tag to clone...** 
    ```shell
    export KOKKOS_TAG=4.1.00
    ```

    **...clone the Kokkos repository...**
    ```shell
    git clone --branch $KOKKOS_TAG https://github.com/kokkos/kokkos.git kokkos-$KOKKOS_TAG 
    ```

    **...and finally export the path to the Kokkos folder.**
    ```shell
    export KOKKOS_PATH=$PWD/kokkos-$KOKKOS_TAG
    ```

    ??? tip
        To avoid having to export this environment variable every time you open a new shell, you might want to add it to your `~/.bashrc` file

    ??? info "Installing Kokkos as shared library/package"
        You may consult the documentation to learn about:    
        [Building Kokkos as an intalled package](https://kokkos.github.io/kokkos-core-wiki/building.html)   
        [Building Kokkos via Spack package manager](https://kokkos.github.io/kokkos-core-wiki/building.html#:~:text=a%20single%20process.-,Spack,-%23)
        but for the tutorial we will compile Kokkos programs inline via a Makefile

!!! success "Next"
    Great! We can now turn to running our first Kokkos program [Tutorial 01: Vector Addition](./vectorAdd/index.md)

