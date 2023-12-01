/*
 *
 * Demonstrating use of Kokkos to compute a Riemann integral (left index function value)
 *
 */

#include <cmath>
#include <cstdio>
#include <typeinfo>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]){
    
    int N = 100; 
    double totalIntegral=0; 
    double a = 0.0, b = 1.0; 
    
    // grid spacing
    double dx = (b - a) / (N * 1.0);
    
/*
 *
 * KOKKOS & LAMBDAS
 *
 *
 */
    std::cout << "\n\nKOKKOS PARALLEL VERSION WITH LAMBDAS" << std::endl; 
    
    // the second argument is a thread-private value that is managed by Kokkos 
    Kokkos::parallel_reduce("RiemannSumLambdas", 
        N, KOKKOS_LAMBDA(const int64_t index, double & partialSum) {
            const double x = a + index * (b - a) / (N * 1.0);
            partialSum += exp(x);
        }, totalIntegral);
    
    // Must call finalize() after using Kokkos
    Kokkos::finalize();

    totalIntegral *= dx; 
    
    std::cout << "Int of exp(x) on [a,b] = [" << a << ", " << b << "] is " << totalIntegral << std::endl;
    

    return 0; 

}
