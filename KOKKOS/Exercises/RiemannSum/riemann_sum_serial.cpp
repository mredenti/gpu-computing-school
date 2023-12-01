/*
 *
 * Demonstrating use of Kokkos to compute a Riemann integral (left index function value)
 *
 */

#include <cmath>

int main(int argc, char* argv[]){
    
    int N = 100; 
    double totalIntegral=0; 
    double a = 0.0, b = 1.0; 
    
    // grid spacing
    double dx = (b - a) / (N * 1.0);
    
/*
 *
 * RIEMANN SUMMATION SERIAL IMPLEMENTATION 
 *
 *
 */
    
    std::cout << "\n\nSERIAL VERSION " << std::endl; 
    
    
    // add contribution from each interval 
    for (int64_t i = 0; i < N; i++){
        // left point
        const double x = a + i * (b - a) / (N * 1.0);
        totalIntegral += exp(x);
    }
    // scale by the grid spacing
    totalIntegral *= dx; 
    
    std::cout << "Int of exp(x) on [a,b] = [" << a << ", " << b << "] is " << totalIntegral << std::endl;
    
    return 0; 
}
