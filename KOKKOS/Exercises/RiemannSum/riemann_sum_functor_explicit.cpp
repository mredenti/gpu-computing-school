/*
 *
 * Demonstrating use of Kokkos to compute a Riemann integral (left index function value)
 *
 */

#include <cmath>
#include <cstdio>
#include <typeinfo>
#include <Kokkos_Core.hpp>

/*
 * A functor is just a class or struct with a public operator() instance method
 */
class RiemannSum{
   
/*
 * Specify the type of the reduction value with a "value type" alias. 
 */
    
    //using value_type = double;
      
    private: 
        double a, b, dx; 

    public:
        // constructor
        RiemannSum(double a_, double b_, double dx_) : a(a_), b(b_), dx(dx_) {}

        KOKKOS_INLINE_FUNCTION
        void operator() (const int64_t i, double& partialSum) const{
            const double x = a + i * dx;
            // is the exponential map available on the device?
            partialSum += exp(x);
        }
}; 

int main(int argc, char* argv[]){
    
    long int N = 10000000; 
    double totalIntegral=0; 
    double a = 0.0, b = 1.0; 
    
    // grid spacing
    double dx = (b - a) / (N * 1.0);
    
    // initialises Kokkos internal objects and all enabled Kokkos backends 
    Kokkos::initialize(argc, argv);
    
/*
 *
 * KOKKOS & FUNCTORS EXPLICIT
 *
 *
 */

    std::cout << "\n\nKOKKOS PARALLEL VERSION WITH FUNCTORS" << std::endl; 
    
    // Kokkos default execution space
    printf("\n\nKokkos execution space %s\n", typeid(Kokkos::DefaultExecutionSpace).name());
    printf("Kokkos host execution space %s\n\n", typeid(Kokkos::DefaultHostExecutionSpace).name());

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::Cuda>(
            Kokkos::Cuda(), 0, N
        ),
        RiemannSum(a, b, dx), totalIntegral
    );
    
    totalIntegral *= dx; 
    std::cout << "Int of exp(x) on [a,b] = [" << a << ", " << b << "] is " << totalIntegral << std::endl;
    
    // Shutdown Kokkos initialized execution spaces and release internally managed resources
    Kokkos::finalize();

    return 0; 
}
