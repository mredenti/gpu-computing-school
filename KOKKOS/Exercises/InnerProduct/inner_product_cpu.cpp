/**
 *
 *  KOKKOS IMPLEMENTATION OF INNER PRODUCT  <y, Ax>
 * 
 *  The implementation allocates vectors and arrays on the host. This means 
 *  unless we move data to the device, running on the GPU will fail. There are 
 *  several ways to approach this:
 *  1. Allocate data on the CPU -> copy data to the device -> launch parallel execution. 
 *      The problem with this first approach is that if in a second moment I do not want to 
 *      use the GPU (I am on a cluster with no GPUs), then 
 *
 **/

#include <Kokkos_Core.hpp>

#define N 10000 // feed this as an input parameter
#define NREPEAT 1000

int main(int argc, char *argv[])
{

    Kokkos::initialize(argc, argv);
    /*
     * I think the following brackets have something to
     * do with scope but I am not sure why
     */
    {
        /*
         * ALLOCATE VECTORS X,Y AND MATRIX A
         */
        auto x = static_cast<double *>(std::malloc(N * sizeof(double)));
        auto y = static_cast<double *>(std::malloc(N * sizeof(double)));
        auto A = static_cast<double *>(std::malloc(N * N * sizeof(double)));

        /* 
        * 
        * I AM NOT SURE HOW THIS WILL WORK WITH CMAKE AND THE REST...?
        *
        */
        
        /*
        #ifdef USE_UDA
        #define MemSpace Kokkos::CudaSpace
        #endif
        #ifndef MemSpace
        #define MemSpace Kokkos::HostSpace
        #endif 
        using ExecSpace = MemSpace::execution_space;
        */
        //using ExecSpace = Kokkos::DefaultHostExecutionSpace; // Defaul Execution Space could be OMP over Serial if enabled?
        // I WANT TO ENABLE THIS TO BE CHANGED EASILY AT COMPILATION TIME
        using ExecSpace = Kokkos::Serial;
        //using range_policy = Kokkos::RangePolicy<ExecSpace>;

        /*
         * INITIALIZE VECTORS X,Y AND MATRIX A
         */

        Kokkos::parallel_for(
            "y_init", Kokkos::RangePolicy<ExecSpace>(ExecSpace(), 0, N), KOKKOS_LAMBDA(int64_t i) {
                y[i] = 1;
            });

        Kokkos::parallel_for(
            "x_init", Kokkos::RangePolicy<ExecSpace>(ExecSpace(), 0, N), KOKKOS_LAMBDA(int64_t i) {
                x[i] = 1;
            });

        // the following loop should be optimised for CPU but poor for GPU (CHECK)
        Kokkos::parallel_for(
            "A_init", Kokkos::RangePolicy<ExecSpace>(ExecSpace(), 0, N), KOKKOS_LAMBDA(int64_t i) {
                for (int j = 0; j < N; ++j)
                {
                    A[i * N + j] = 1;
                }
            });

        // Instrumentation, initialize timer
        Kokkos::Timer timer;
        // repeat computation multiple times

        // ? is there any benefit from decomposing the operation first doint Ax and then y . (Ax) ?
        for (int repeat = 0; repeat < NREPEAT; repeat++)
        {

            double result = 0;
            Kokkos::parallel_reduce(
                "yAx", Kokkos::RangePolicy<ExecSpace>(ExecSpace(), 0, N), KOKKOS_LAMBDA(int64_t i, double &update) {
                    double temp2 = 0;
                    for (int j = 0; j < N; j++)
                    {
                        temp2 += A[i * N + j] * x[j];
                    }

                    update += y[i] * temp2;
                },
                result);
        }

        // has result gone out of scope? Otherwise, check solution...?

        double time = timer.seconds();

        /*
         * COMPUTE BANDWIDTH:
         *
         */

        double Gbytes = 1.0e-9 * double(sizeof(double) * (2 * N * N + N));

        printf("  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
               N, N, NREPEAT, Gbytes * 1000, time, Gbytes * NREPEAT / time);

        std::free(x);
        std::free(y);
        std::free(A);
    }

    Kokkos::finalize();

    return 0;
}
