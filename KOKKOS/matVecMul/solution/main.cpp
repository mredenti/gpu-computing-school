//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

// ************************************************************************
// MODIFICATION NOTICE:
// This file has been modified from its original version. It has been
// formatted, edited, or otherwise altered in a way that changes the
// content or structure of the original version. The modifications are
// not endorsed by the original authors or the copyright holder.
// ************************************************************************

#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <Kokkos_Core.hpp>

void checkSizes(int &N, int &M, int &S, int &nrepeat);

int main(int argc, char *argv[])
{
  int N = -1;        // number of rows 2^12
  int M = -1;        // number of columns 2^10
  int S = -1;        // total size 2^22
  int nrepeat = 100; // number of repeats of the test

  // Read command line arguments.
  for (int i = 0; i < argc; i++)
  {
    if ((strcmp(argv[i], "-N") == 0) || (strcmp(argv[i], "-Rows") == 0))
    {
      N = pow(2, atoi(argv[++i]));
      printf("  User N is %d\n", N);
    }
    else if ((strcmp(argv[i], "-M") == 0) || (strcmp(argv[i], "-Columns") == 0))
    {
      M = pow(2, atof(argv[++i]));
      printf("  User M is %d\n", M);
    }
    else if ((strcmp(argv[i], "-S") == 0) || (strcmp(argv[i], "-Size") == 0))
    {
      S = pow(2, atof(argv[++i]));
      printf("  User S is %d\n", S);
    }
    else if (strcmp(argv[i], "-nrepeat") == 0)
    {
      nrepeat = atoi(argv[++i]);
    }
    else if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0))
    {
      printf("  y=Ax Options:\n");
      printf("  -Rows (-N) <int>:      exponent num, determines number of rows 2^num (default: 2^12 = 4096)\n");
      printf("  -Columns (-M) <int>:   exponent num, determines number of columns 2^num (default: 2^10 = 1024)\n");
      printf("  -Size (-S) <int>:      exponent num, determines total matrix size 2^num (default: 2^22 = 4096*1024 )\n");
      printf("  -nrepeat <int>:        number of repetitions (default: 100)\n");
      printf("  -help (-h):            print this message\n\n");
      exit(1);
    }
  }

  // Check sizes.
  checkSizes(N, M, S, nrepeat);

  Kokkos::initialize(argc, argv);
  {

    /* ------------------ Print default memory and execution spaces ----------------------*/
    using executionSpace = typename Kokkos::DefaultExecutionSpace;
    auto memorySpace = executionSpace::memory_space();

    std::cout << "Default Execution Space: " << executionSpace::name() << std::endl;
    std::cout << "Default Memory Space: " << memorySpace.name() << std::endl;
    std::cout << "-------\n";

    /* ------------------ Allocate y, x vectors and Matrix A on device. ----------------------*/
    typedef Kokkos::View<double *> ViewVectorType;
    typedef Kokkos::View<double **> ViewMatrixType;

    ViewVectorType y("y", N);
    ViewMatrixType A("A", N, M);
    ViewVectorType x("x", M);

    /* ------------------ TASK: Create mirrors of the views on host. ----------------------*/

    ViewVectorType::HostMirror h_y = Kokkos::create_mirror_view(y);
    ViewMatrixType::HostMirror h_A = Kokkos::create_mirror_view(A);
    ViewVectorType::HostMirror h_x = Kokkos::create_mirror_view(x);

    // Initialize x vector on host.
    for (int i = 0; i < M; ++i)
    {
      h_x(i) = 1;
    }

    // Initialize A matrix on host.
    for (int j = 0; j < N; ++j)
    {
      for (int i = 0; i < M; ++i)
      {
        h_A(j, i) = 1;
      }
    }

    // Deep copy host views to device views.
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(A, h_A);

    // Timer products.
    Kokkos::Timer timer;

    for (int repeat = 0; repeat < nrepeat; repeat++)
    {

      Kokkos::parallel_for(
          "y=Ax", N, KOKKOS_LAMBDA(const int i) {
            
            double sum = 0;

            for (int j = 0; j < M; j++)
            {
              sum += A(i, j) * x(j);
            }

            y(i) = sum;
          });
    }

    // syncrhonize
    Kokkos::fence();

    // Calculate time.
    double time = timer.seconds();

    /* ------------ Verify results ----------------*/

    const double solution = (double)M * (double)N;
    double result = 0;

    Kokkos::parallel_reduce(
        "sum(y)", N, KOKKOS_LAMBDA(const int i, double &partial_sum) {
          partial_sum += y(i);
        },
        result);

    // synchronize 
    Kokkos::fence();

    if (result != solution)
    {
      printf("  Error: result( %lf ) != solution( %lf )\n", result, solution);
    }

    /* ------------ Calculate bandwidth ----------------*/
    // Each matrix A row (each of length M) is read once.
    // The x vector (of length M) is read N times.
    // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N ) );
    double Gbytes = 1.0e-9 * double(sizeof(double) * (M + M * N));

    // Print results (problem size, time and bandwidth in GB/s).
    printf("  S (%d) = N( %d ) x M( %d )  nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
           N*M, N, M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time);
  }
  Kokkos::finalize();

  return 0;
}

void checkSizes(int &N, int &M, int &S, int &nrepeat)
{
  // If S is undefined and N or M is undefined, set S to 2^22 or the bigger of N and M.
  if (S == -1 && (N == -1 || M == -1))
  {
    S = pow(2, 22);
    if (S < N)
      S = N;
    if (S < M)
      S = M;
  }

  // If S is undefined and both N and M are defined, set S = N * M.
  if (S == -1)
    S = N * M;

  // If both N and M are undefined, fix row length to the smaller of S and 2^10 = 1024.
  if (N == -1 && M == -1)
  {
    if (S > 1024)
    {
      M = 1024;
    }
    else
    {
      M = S;
    }
  }

  // If only M is undefined, set it.
  if (M == -1)
    M = S / N;

  // If N is undefined, set it.
  if (N == -1)
    N = S / M;

  printf("  A is (N = %d) X (M = %d)\n", N, M);

  // Check sizes.
  if ((S < 0) || (N < 0) || (M < 0) || (nrepeat < 0))
  {
    printf("  Sizes must be greater than 0.\n");
    exit(1);
  }

  if ((N * M) != S)
  {
    printf("  N * M != S\n");
    exit(1);
  }
}