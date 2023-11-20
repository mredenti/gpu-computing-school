#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <Kokkos_Core.hpp>

void checkSizes(int &N, int &M, int &S, int &nrepeat);

int main(int argc, char *argv[])
{
  
  int nrepeat = 100; // number of repeats of the test
  int N = argc > 1 ? std::atoi(argv[1]) : pow(2, 12); // Set N to argument value or 10 if no argument

  std::cout << "Value of N is: " << N << std::endl;
  
  Kokkos::initialize(argc, argv);
  {

    /* ------------------ Print default memory and execution spaces ----------------------*/
    using executionSpace = typename Kokkos::DefaultExecutionSpace;
    auto memorySpace = executionSpace::memory_space();

    std::cout << "Default Execution Space: " << executionSpace::name() << std::endl;
    std::cout << "Default Memory Space: " << memorySpace.name() << std::endl;
    std::cout << "-------\n";


    /* ------------------ Allocate a, b, c vectors on device (if GPU backend enabled) ----------------------*/
    typedef Kokkos::View<double *> ViewVectorType;


    // @TASK@ Allocate one dimensional views views for a, b and c


    /* ------------------ @TASK@ Create mirrors of the views on host. ----------------------*/

    //ViewVectorType::HostMirror h_a = Kokkos::create_mirror_view(a);
    //ViewMatrixType::HostMirror h_b = Kokkos::create_mirror_view(b);
    //ViewVectorType::HostMirror h_c = Kokkos::create_mirror_view(c);

    // @TASK@ Initialize a vector on host.
    for (int i = 0; i < h_a.extent(0); ++i)
    {
      a(i) = 1;
    }

    // @TASK@ Initialize b vector on host.
    for (int i = 0; i < h_b.extent(0); ++i)
    {
      b(i) = 1;
    }

    // @TASK@ Deep copy host views to device views.

    // Timer products.
    Kokkos::Timer timer;

    for (int repeat = 0; repeat < nrepeat; repeat++)
    {

      Kokkos::parallel_for(
          "c=a+b", c.extent(0), KOKKOS_LAMBDA(const int i) {
            
            c(i) = a(i) + b(i);

          });
    }

    // syncrhonize
    Kokkos::fence();

    // Calculate time.
    double time = timer.seconds();

    /* ------------ Verify results ----------------*/

    const double solution = (double)N;
    double result = 0;

    Kokkos::parallel_reduce(
        "sum(c)", N, KOKKOS_LAMBDA(const int i, double &partial_sum) {
          partial_sum += c(i);
        },
        result);

    // synchronize 
    Kokkos::fence();

    if (result != solution)
    {
      printf("  Error: result( %lf ) != solution( %lf )\n", result, solution);
    }

    /* ------------ Print Avg Time ----------------*/

    // Print results (problem size, time and bandwidth in GB/s).
    printf("Avg time( %g s )\n", time/nrepeat);
  }
  Kokkos::finalize();

  return 0;
}

