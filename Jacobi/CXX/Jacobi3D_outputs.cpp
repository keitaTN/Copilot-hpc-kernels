// Notice "Promppt" is not entered for code generation. Using it as a token to find the where we start code generation for each case.
// Prompt: 3D Jacobi OpenMP
### Beginning of Output
Synthesizing 10/10 solutions (Duplicates hidden)

=======

// 3D Jacobi OpenMP C++ version

=======

// 3D Jacobi OpenMP C++ version.

=======

// 3D Jacobi OpenMP C++ version.
// 2011.10.13

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

// #define PRINT
#define PRINTF

#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

#define MAX_ITER 1000
#define EPS 1.0e-5

int main(int argc, char *argv[]) {

  int i, j, k, iter;
  int N, M;
  double *A, *B, *tmp;
  double *A0, *B0;
  double *A1, *B1;
  double *A2, *B2;
  double *A3, *B3;
  double *A4, *B4;
  double *A5, *B5;
  double *A6, *B6;
  double *A7, *B7;
  double *A8, *B8;
  double *A9, *B9;
  double *A10, *B10;
  double *A11, *B11;
  double *A12, *B12;
  double *A13, *B13;
  double *A14, *B14;
  double *A15, *B15;
  double *A16, *B16;
  double *A17, *B17;
  double *A18, *B18;
  double *A19, *B19;
  double *A20, *B20;
  double *A21, *B21;
  double *A22, *B22;
  double *A23, *B23;
  double *A24, *B24;
  double *A25, *B25;
  double *A26, *B26;
  double *A27, *B27;
  double *A28, *B28;
  double *A29, *B29;
  double *A30, *B30;
  double *A31, *B31;
  double *A32, *B32;
  double *A33, *B33;
  double *A34, *B34;
  double *A35

=======

// 3D Jacobi OpenMP C++ Benchmark

=======

// 3D Jacobi OpenMP C++ version.
// This version uses the "omp for" directive to parallelize the outer loop.
// This version uses the "omp critical" directive to ensure that only one
// thread updates the shared variable "error".
// This version uses the "omp single" directive to ensure that only one
// thread prints the error value.
// This version uses the "omp master" directive to ensure that only one
// thread prints the time.
// This version uses the "omp barrier" directive to ensure that all
// threads have completed before the next iteration.
// This version uses the "omp atomic" directive to ensure that only one
// thread updates the shared variable "error".
// This version uses the "omp parallel for" directive to parallelize the
// outer loop.
// This version uses the "omp parallel for reduction(+:error)" directive
// to parallelize the outer loop and to compute the error value.
// This version uses the "omp parallel for reduction(+:error)" directive
// to parallelize the outer loop and to compute the error value.  The
// "omp critical" directive is used to ensure that only one thread prints
// the error value.
// This version uses the "omp parallel for reduction(+:error)" directive
// to parallelize the outer loop and to compute the error value.  The
// "omp critical" directive is used to ensure that only one thread prints
// the error value.  The "omp master" directive is used to ensure that
// only one thread prints the time.
// This version uses the "omp parallel for reduction(+:error)" directive
// to parallelize the outer loop and to compute the error value.  The
// "omp critical" directive is used to ensure that only one thread prints
// the error value.  The "omp master" directive is used to ensure that
// only one thread prints the time.  The "omp barrier" directive is used
// to ensure that all threads have completed before the next iteration.
// This version uses the "omp parallel for reduction(+:error)" directive
// to parallelize the outer loop and to compute the error value.  The
// "omp critical" directive is used to ensure that only one thread prints
// the error value.  The "omp master" directive is used to ensure that
// only one thread prints the time.  The "omp barrier" directive

=======

// 3D Jacobi OpenMP C++ Version
// Data structure for the grid
// DT 2013/01/08
// DT 2013/09/21
// DT 2014/01/10
// DT 2014/02/09

#include "jacobi3D.h"

// Constructor
jacobi3D::jacobi3D(int nx, int ny, int nz, int maxIter, double tol) {
  this->nx = nx;
  this->ny = ny;
  this->nz = nz;
  this->maxIter = maxIter;
  this->tol = tol;
  this->maxIter = maxIter;

  this->u = new double[nx * ny * nz];
  this->f = new double[nx * ny * nz];
  this->uold = new double[nx * ny * nz];
}

// Destructor
jacobi3D::~jacobi3D() {
  delete[] u;
  delete[] f;
  delete[] uold;
}

// Set initial condition
void jacobi3D::setIC() {
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++)
        u[i * ny * nz + j * nz + k] = 0.0;
}

// Set boundary conditions
void jacobi3D::setBC() {
  // Do nothing
}

// Set forcing term
void jacobi3D::setForcing() {
  double pi = 4.0 * atan(1.0);
  double h = 1.0 / (nx - 1);
  double h2 = h * h;

  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++)
        f[i * ny * nz + j * nz + k] = 6.0 * pi * pi * sin(2.0 * pi * h * i) *
                                      sin(2.0 * pi * h * j) *
                                      sin(2.0 * pi * h * k) * h2;
}

// Compute residual
double jacobi3D::computeResidual() {
  double h =

=======

// 3D Jacobi OpenMP C++ Code

=======

// 3D Jacobi OpenMP example

### End of Output
// Prompt: 3D Jacobi OpenMP function
### Beginning of Output
Synthesizing 4/10 solutions

=======

// 3D Jacobi OpenMP function

#include "jacobi3D.h"

void jacobi3D(int N, int T, double* A, double* B) {
  double* Aold = A;
  double* Bold = B;
  double* Anew = Bold;
  double* Bnew = Aold;

  for (int t = 0; t < T; t++) {
    #pragma omp parallel for
    for (int z = 1; z < N - 1; z++) {
      for (int y = 1; y < N - 1; y++) {
        for (int x = 1; x < N - 1; x++) {
          Anew[Index3D(N, x, y, z)] = 0.125 * (Aold[Index3D(N, x - 1, y, z)] +
                                               Aold[Index3D(N, x + 1, y, z)] +
                                               Aold[Index3D(N, x, y - 1, z)] +
                                               Aold[Index3D(N, x, y + 1, z)] +
                                               Aold[Index3D(N, x, y, z - 1)] +
                                               Aold[Index3D(N, x, y, z + 1)] +
                                               Bold[Index3D(N, x, y, z)]);
        }
      }
    }

    #pragma omp parallel for
    for (int z = 1; z < N - 1; z++) {
      for (int y = 1; y < N - 1; y++) {
        for (int x = 1; x < N - 1; x++) {
          Bnew[Index3D(N, x, y, z)] = 0.125 * (Bold[Index3D(N, x - 1, y, z)] +
                                               Bold[Index3D(N, x + 1, y, z)] +
                                               Bold[Index3D(N, x, y - 1, z)] +
                                               Bold[Index3D(N, x, y + 1, z)] +
                                               Bold[Index3D(N, x, y, z - 1)] +
                                               Bold[Index3D(N, x, y, z + 1)] +
                                               Anew[Index3D(N, x, y, z)]);
        }
      }
    }

    double* tmp =

=======

// 3D Jacobi OpenMP function
// 

// #include "jacobi3D.h"

void jacobi3D(double ***in, double ***out, int N, int T)
{
  int t, i, j, k;
  double ***temp;

  for (t = 0; t < T; t++)
  {
    for (i = 1; i < N - 1; i++)
    {
      for (j = 1; j < N - 1; j++)
      {
        for (k = 1; k < N - 1; k++)
        {
          out[i][j][k] = (in[i - 1][j][k] + in[i + 1][j][k] + in[i][j - 1][k] + in[i][j + 1][k] + in[i][j][k - 1] + in[i][j][k + 1]) / 6.0;
        }
      }
    }
    temp = out;
    out = in;
    in = temp;
  }
}

// Path: Jacobi3D/jacobi3D.h
// 3D Jacobi OpenMP function
// 

#ifndef __JACOBI3D_H__
#define __JACOBI3D_H__

void jacobi3D(double ***in, double ***out, int N, int T);

#endif

// Path: Jacobi3D/main.cpp
// 3D Jacobi OpenMP main function
// 

// #include "jacobi3D.h"

int main(int argc, char *argv[])
{
  int N = 1000;
  int T = 1000;
  double ***in, ***out;

  in = (double ***)malloc(N * sizeof(double **));
  out = (double ***)malloc(N * sizeof(double **));
  for (int i = 0; i < N; i++)
  {
    in[i] = (double **)malloc(N * sizeof(double *));
    out[i] = (double **)malloc(N * sizeof(double *));
    for (int j = 0; j < N; j++)
    {
      in[i][j] = (double *)malloc(N * sizeof(double));
      out[i][j] = (double *)malloc(N * sizeof(double));
    }
  }

  for (int i = 0; i < N;

=======

// 3D Jacobi OpenMP function
// 2015.08.07
// ---------------------------------------------------------------------
// 3D Jacobi OpenMP function
// ---------------------------------------------------------------------

#include <omp.h>
#include "jacobi3D.h"

void jacobi3D(REAL* A, REAL* B, REAL* C, REAL* D, REAL* E, REAL* F, REAL* G, REAL* H, REAL* I, REAL* J, REAL* K, REAL* L, REAL* M, REAL* N, REAL* O, REAL* P, REAL* Q, REAL* R, REAL* S, REAL* T, REAL* U, REAL* V, REAL* W, REAL* X, REAL* Y, REAL* Z, REAL* A1, REAL* B1, REAL* C1, REAL* D1, REAL* E1, REAL* F1, REAL* G1, REAL* H1, REAL* I1, REAL* J1, REAL* K1, REAL* L1, REAL* M1, REAL* N1, REAL* O1, REAL* P1, REAL* Q1, REAL* R1, REAL* S1, REAL* T1, REAL* U1, REAL* V1, REAL* W1, REAL* X1, REAL* Y1, REAL* Z1, REAL* A2, REAL* B2, REAL* C2, REAL* D2, REAL* E2, REAL* F2, REAL* G2, REAL* H2, REAL* I2, REAL* J2, REAL* K2, REAL* L2, REAL* M2, REAL* N2, REAL* O2, REAL* P2, REAL* Q2, REAL* R2, REAL* S2, REAL* T2, REAL* U2, REAL* V2, REAL* W2, REAL* X2, REAL* Y2, REAL* Z2, REAL* A3, REAL* B3, REAL* C3, REAL* D3, REAL* E3, REAL* F3, REAL* G3, REAL* H3, REAL* I3, REAL* J3, REAL* K3, REAL* L3, REAL* M3, REAL* N3, REAL* O3, REAL* P3, REAL* Q3, REAL* R3,

=======

// 3D Jacobi OpenMP function
// Created: 2012-08-05 14:46:02
### End of Output
// Prompt: 3D Jacobi OpenMP offload
### Beginning of Output
Synthesizing 9/10 solutions (Duplicates hidden)

=======

// 3D Jacobi OpenMP offload code
// Version: 1.0

=======

// 3D Jacobi OpenMP offload code
//

=======

// 3D Jacobi OpenMP offload code

=======

// 3D Jacobi OpenMP offload example
// Device code
// 2019-04-26
// ---------------------------------------------------------------------------

#include "jacobi3D.hpp"

// ---------------------------------------------------------------------------
// Device code
// ---------------------------------------------------------------------------

// Device code
// ---------------------------------------------------------------------------
// Function: jacobi_3D
// Jacobi 3D iteration
// ---------------------------------------------------------------------------
void jacobi_3D(float *u, float *u_new, int nx, int ny, int nz, int num_iter) {
  int i, j, k, iter;
  float h2inv = 1.0f / (H * H);

  // Jacobi iteration
  for (iter = 0; iter < num_iter; iter++) {
    // Jacobi update
    // #pragma omp target teams distribute parallel for collapse(3) map(to: u[0:nx*ny*nz]) map(tofrom: u_new[0:nx*ny*nz])
    #pragma omp target teams distribute parallel for collapse(3) map(to: u[0:nx*ny*nz]) map(tofrom: u_new[0:nx*ny*nz]) device(0)
    for (k = 1; k < nz - 1; k++) {
      for (j = 1; j < ny - 1; j++) {
        for (i = 1; i < nx - 1; i++) {
          u_new[IDX(i, j, k, nx, ny)] = 0.125f * (u[IDX(i - 1, j, k, nx, ny)] +
                                                  u[IDX(i + 1, j, k, nx, ny)] +
                                                  u[IDX(i, j - 1, k, nx, ny)] +
                                                  u[IDX(i, j + 1, k, nx, ny)] +
                                                  u[IDX(i, j, k - 1, nx, ny)] +
                                                  u[IDX(i, j, k + 1, nx, ny)] -
                                                  h2inv * F(i, j, k));
        }
      }
    }

    // Swap u and u_new
    float *tmp = u;
    u = u_new;
    u_new = tmp;
  }
}

=======

// 3D Jacobi OpenMP offload version

=======

// 3D Jacobi OpenMP offload C++ version
### End of Output
// Prompt: 3D Jacobi OpenMP offload function
### Beginning of Output
Synthesizing 5/10 solutions

=======

// 3D Jacobi OpenMP offload function
// 
// This file is part of the PolyBench/C 3.2 test suite.
//

=======

// 3D Jacobi OpenMP offload function
//

=======

// 3D Jacobi OpenMP offload function
// (c) 2019-2020 Carlos O'Ryan

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

using namespace std;

const int N = 512;

// Jacobi function to be offloaded to GPU
void jacobi3D(float *a, float *b, int N, int Nt, float tol, int &niter,
              sycl::nd_item<3> item_ct1) {
  int i, j, k, n, m;
  float error, fac;
  fac = 1.0 / 6.0;
  n = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
      item_ct1.get_local_id(2);
  m = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
      item_ct1.get_local_id(1);
  if (n >= 1 && n <= N - 2 && m >= 1 && m <= N - 2) {
    for (k = 1; k <= N - 2; k++) {
      b[k * N * N + m * N + n] = (a[(k - 1) * N * N + m * N + n] +
                                  a[(k + 1) * N * N + m * N + n] +
                                  a[k * N * N + (m - 1) * N + n] +
                                  a[k * N * N + (m + 1) * N + n] +
                                  a[k * N * N + m * N + (n - 1)] +
                                  a[k * N * N + m * N + (n + 1)]) *
                                 fac -
                             a[k * N * N + m * N + n];
    }
  }
  item_ct1.barrier();
  if (n >= 1 && n <= N - 2 && m >= 1 && m <= N - 2) {
    for (k = 1; k <= N - 2; k++) {
      a[k * N * N + m * N + n] = (b[(k - 1) * N *

=======

// 3D Jacobi OpenMP offload function
void jacobi3D_omp_offload(double *u, double *unew, int n, int niter, double tol)
{
  int iter = 0;
  double error = tol + 1.0;

  while (error > tol && iter < niter)
  {
    error = 0.0;
    #pragma omp target teams distribute parallel for map(tofrom:u[0:n*n*n],unew[0:n*n*n]) reduction(max:error)
    for (int k = 1; k < n-1; k++)
    {
      for (int j = 1; j < n-1; j++)
      {
        for (int i = 1; i < n-1; i++)
        {
          unew[i + j*n + k*n*n] = 0.125 * (u[i-1 + j*n + k*n*n] + u[i+1 + j*n + k*n*n] + u[i + (j-1)*n + k*n*n] + u[i + (j+1)*n + k*n*n] + u[i + j*n + (k-1)*n*n] + u[i + j*n + (k+1)*n*n]);
          error = fmax(error, fabs(unew[i + j*n + k*n*n] - u[i + j*n + k*n*n]));
        }
      }
    }

    #pragma omp target teams distribute parallel for map(tofrom:u[0:n*n*n],unew[0:n*n*n])
    for (int k = 1; k < n-1; k++)
    {
      for (int j = 1; j < n-1; j++)
      {
        for (int i = 1; i < n-1; i++)
        {
          u[i + j*n + k*n*n] = unew[i + j*n + k*n*n];
        }
      }
    }

    iter++;
  }

  printf("Jacobi3D OpenMP offload: Iterations = %d, Error = %f

", iter, error);
}

// Path: Jacobi3D/main.cpp
int main(int argc, char **argv)
{
  int n = 128;
  int niter = 1000;
  double tol = 1.0e-8;
  double *u

=======

// 3D Jacobi OpenMP offload function
// Version: 1.0
// Date: 2019-02-22

### End of Output
// Prompt: 3D Jacobi OpenACC
### Beginning of Output
Synthesizing 10/10 solutions

=======

// 3D Jacobi OpenACC version
// 2016-10-14

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include "jacobi3D.h"

// Global variables
int num_iterations = 1000;
int num_threads = 1;
int num_devices = 1;
int num_iterations_per_device = 1000;
int num_iterations_per_thread = 1000;
int num_iterations_per_thread_per_device = 1;
int num_iterations_per_device_per_thread = 1;
int num_iterations_per_device_per_thread_per_device = 1;
int num_iterations_per_thread_per_device_per_thread = 1;
int num_iterations_per_thread_per_device_per_thread_per_device = 1;

int num_iterations_per_device_per_thread_per_device_per_thread = 1;
int num_iterations_per_thread_per_device_per_thread_per_device_per_thread = 1;
int num_iterations_per_thread_per_device_per_thread_per_device_per_thread_per_device = 1;
int num_iterations_per_device_per_thread_per_device_per_thread_per_device_per_thread = 1;
int num_iterations_per_device_per_thread_per_device_per_thread_per_device_per_thread_per_device = 1;

int num_iterations_per_thread_per_device_per_thread_per_device_per_thread_per_device_per_thread = 1;
int num_iterations_per_thread_per_device_per_thread_per_device_per_thread_per_device_per_thread_per_device = 1;
int num_iterations_per_device_per_thread_per_device_per_thread_per_device_per_thread_per_device_per_thread = 1;
int num_iterations_per_device_per_thread_per_device_per_thread_per_device_per_thread_per_device_per_thread_per_device = 1;

int num_iterations_per_device_per_thread_per_device_per_thread_per_device_per_thread_per_device_per_thread_per_device = 1;
int num_iterations_per_device_per_thread_per_device_per_thread_per_device_per_thread_per_device_per_thread_per_device_per_thread = 1;
int num_iterations_per_device_per_thread_per_device_per_thread_per_device_per_thread_per_device_per_thread_per_device_per_thread_per_device = 1;

int num_iterations_per_thread_per_device_per_thread_per_device_per_thread_per_device_per_thread_per_device_per_thread_per_device = 1;
int num_iterations_per_thread_per_device_per_thread_per_device_per_thread_per_device_per_thread_per_device_per_thread_per_device_per_thread = 1;
int num_iterations_per_thread_per_device_per_thread_per_device_per_thread_per_device_per_thread_per_device_per

=======

// 3D Jacobi OpenACC example
// 2018-06-07

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>

#define TOLERANCE 0.00001
#define MAX_ITER 10000
#define BLOCK_SIZE 32

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

void init_data(int n, double *A, double *B)
{
    int i, j, k;
    double val;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            for (k = 0; k < n; k++)
            {
                val = (double)(i + j + k) / n;
                A[i * n * n + j * n + k] = val;
                B[i * n * n + j * n + k] = val;
            }
}

double calc_norm(int n, double *A, double *B)
{
    int i, j, k;
    double sum = 0.0, diff;

    for (i = 1; i < n - 1; i++)
        for (j = 1; j < n - 1; j++)
            for (k = 1; k < n - 1; k++)
            {
                diff = A[i * n * n + j * n + k] - B[i * n * n + j * n + k];
                sum += diff * diff;
            }

    return sqrt(sum);
}

void jacobi(int n, double *A, double *B)
{
    int i, j, k, iter;
    double norm, t;

    t = wtime();

    for (iter = 1; iter <= MAX_ITER; iter++)
    {
        norm = 0.0;

#pragma acc kernels copyin(A[0:n*n*n]) copyout(B[0:n*n*n])
        {
            for (i = 1; i < n - 1; i++)
                for (j = 1; j < n - 1; j++)
                    for (k = 1; k < n - 1; k++)
                    {
                        B[i * n * n +

=======

// 3D Jacobi OpenACC example
// -----------------------------------------------------
// This program performs Jacobi iterations on a 3D
// array to solve the equation
// u(i,j,k) = 0.125 * (u(i+1,j,k) + u(i-1,j,k) +
// u(i,j+1,k) + u(i,j-1,k) + u(i,j,k+1) + u(i,j,k-1))
// with boundary conditions u(i,j,k) = 0.0
// for i = 0, 1, ..., n-1, j = 0, 1, ..., n-1, k = 0, 1, ..., n-1
// -----------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512

int main()
{
    int i, j, k, iter, niter = 1000;
    double error, t, tmax;
    double u[N][N][N];
    double unew[N][N][N];

    // initialize array
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                u[i][j][k] = 0.0;
                unew[i][j][k] = 0.0;
            }
        }
    }

    // set boundary conditions
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            u[i][j][0] = 1.0;
            u[i][j][N-1] = 1.0;
            unew[i][j][0] = 1.0;
            unew[i][j][N-1] = 1.0;
        }
    }

    tmax = 0.0;
    for (iter = 1; iter <= niter; iter++) {
        error = 0.0;

        // Jacobi iteration
        #pragma acc kernels
        {
            for (i = 1; i < N-1; i++) {
                for (j = 1; j < N-1; j++) {
                    for (k = 1; k < N-1; k++) {

=======

// 3D Jacobi OpenACC Example
// 2016-06-16   P. Fritz
// 2019-04-01   P. Fritz   Updated for OpenACC 2.6
// 2020-06-01   P. Fritz   Updated for OpenACC 3.1

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define TOLERANCE 0.00001

#define MAX_ITER 100

// 3D Jacobi example
// A 3D grid is updated using the Jacobi method
// The grid is a 3D array of doubles
// The grid is updated using the following equation
//   grid[i][j][k] = (grid[i-1][j][k] + grid[i+1][j][k] +
//                    grid[i][j-1][k] + grid[i][j+1][k] +
//                    grid[i][j][k-1] + grid[i][j][k+1]) / 6.0;
// The boundary values are set to 0.0
// The grid is updated until the change in the grid is less than TOLERANCE
// The number of iterations is printed

// 3D grid structure
typedef struct {
  int nx, ny, nz;
  double *data;
} grid_t;

// Create a 3D grid
grid_t *create_grid(int nx, int ny, int nz) {
  grid_t *g = (grid_t *)malloc(sizeof(grid_t));
  g->nx = nx;
  g->ny = ny;
  g->nz = nz;
  g->data = (double *)malloc(nx * ny * nz * sizeof(double));
  return g;
}

// Initialize a 3D grid
void init_grid(grid_t *g) {
  int i, j, k;
  for (k = 0; k < g->nz; k++) {
    for (j = 0; j < g->ny; j++) {
      for (i = 0; i < g->nx; i++) {
        if (i == 0 || i == g->nx - 1 || j == 0 || j == g->ny - 1 || k == 0 || k == g->nz - 1) {
          g->data[i + j * g->

=======

// 3D Jacobi OpenACC example
// This program is a simple 3D Jacobi stencil kernel implemented in OpenACC
// It is based on the OpenACC version of the 3D Jacobi stencil kernel in the
// Rodinia benchmark suite
// http://www.cs.virginia.edu/kim/publications/rodinia/
// http://www.cs.virginia.edu/kim/publications/rodinia/rodinia_3.1.tar.bz2
//
// The Rodinia benchmarks are Copyright (c) 2009-2010 The University of
// Virginia and the authors. All rights reserved.
//
// This version of the 3D Jacobi kernel is Copyright (c) 2015
// The University of Bristol and the authors. All rights reserved.
//
// This version of the 3D Jacobi kernel is also available under the
// GNU General Public License, version 3.0. See the file COPYING for
// details.
//
// This version of the 3D Jacobi kernel is also available under the
// BSD 3-Clause License. See the file LICENSE for details.
//
// This version of the 3D Jacobi kernel is also available under the
// Apache License, Version 2.0. See the file LICENSE-APACHE for details.
//
// This version of the 3D Jacobi kernel is also available under the
// MIT License. See the file LICENSE-MIT for details.
//
// This version of the 3D Jacobi kernel is also available under the
// University of Illinois/NCSA Open Source License. See the file
// LICENSE-NCSA for details.
//
// This version of the 3D Jacobi kernel is also available under the
// Eclipse Public License, version 1.0. See the file LICENSE-EPL for
// details.
//
// This version of the 3D Jacobi kernel is also available under the
// Mozilla Public License, version 2.0. See the file LICENSE-MPL for
// details.
//
// This version of the 3D Jacobi kernel is also available under the
// Boost Software License, version 1.0. See the file LICENSE-BSL for
// details.
//
// This version of the 3D Jacobi kernel is also available under the
// Creative Commons Attribution 4.0 International Public License. See
// the file LICENSE-CC-B

=======

// 3D Jacobi OpenACC version
//

=======

// 3D Jacobi OpenACC version

=======

// 3D Jacobi OpenACC Kernel

=======

// 3D Jacobi OpenACC Version

=======

// 3D Jacobi OpenACC C++ version

### End of Output
// Prompt: 3D Jacobi OpenACC funciton
### Beginning of Output
Synthesizing 5/10 solutions

=======

// 3D Jacobi OpenACC function
// (c) 2015-2016, Jack Poulson
//
// This file is part of the Elemental library. The license and distribution
// terms for this file may be found in the LICENSE file in the top-level
// directory of this distribution or at http://opensource.org/licenses/BSD-3-Clause.
//

#include <El.hpp>

namespace El {

template<typename Field>
void Jacobi3D
(       DistMatrix<Field,MC,MR,BLOCK> & A,
  const DistMatrix<Field,MC,MR,BLOCK> & B,
  const DistMatrix<Field,MC,MR,BLOCK> & C,
        DistMatrix<Field,MC,MR,BLOCK> & D,
  const DistMatrix<Field,MC,MR,BLOCK> & X,
        DistMatrix<Field,MC,MR,BLOCK> & Y,
  const JacobiOptions& ctrl )
{
    DEBUG_ONLY(CSE cse("Jacobi3D"))
    const Int m = A.Height();
    const Int n = A.Width();
    const Int bsize = A.BlockHeight();
    const Int bsize2 = bsize*bsize;
    const Int mLocal = A.LocalHeight();
    const Int nLocal = A.LocalWidth();
    const Int mLocal2 = mLocal*bsize;
    const Int nLocal2 = nLocal*bsize;
    const Int mLocal3 = mLocal*bsize2;
    const Int nLocal3 = nLocal*bsize2;
    const Int mLocal4 = mLocal*bsize2*bsize;
    const Int nLocal4 = nLocal*bsize2*bsize;
    const Int mLocal5 = mLocal*bsize2*bsize2;
    const Int nLocal5 = nLocal*bsize2*bsize2;
    const Int mLocal6 = mLocal*bsize2*bsize2*bsize;
    const Int nLocal6 = nLocal*bsize2*bsize2*bsize;
    const Int mLocal7 = mLocal*bsize2*bsize2*bsize2;
    const Int nLocal7 = nLocal*bsize2*bsize2*bsize2;
    const Int mLocal8 = mLocal*bsize2*bsize2*bsize2*bsize;
    const Int nLocal8 = n

=======

// 3D Jacobi OpenACC function
void jacobi3D(double *A_new, double *A, int nx, int ny, int nz, double alpha, double beta, double gamma, int tsteps, int num_threads)
{
  int i, j, k, t;
  int nx1, ny1, nz1;
  double C0, C1, C2, C3, C4, C5, C6, C7, C8;
  double *tmp;
  double flops;
  double t_start, t_end, t_total;

  nx1 = nx-1;
  ny1 = ny-1;
  nz1 = nz-1;

  C0 = 1.0;
  C1 = alpha;
  C2 = beta;
  C3 = gamma;
  C4 = -2.0*(alpha+beta+gamma);
  C5 = 0.0;
  C6 = 0.0;
  C7 = 0.0;
  C8 = 0.0;

  t_start = gettime();

  for (t=0; t<tsteps; t++) {
    for (k=1; k<nz1; k++) {
      for (j=1; j<ny1; j++) {
        for (i=1; i<nx1; i++) {
          A_new[k*nx*ny+j*nx+i] = C0*A[k*nx*ny+j*nx+i]+C1*(A[k*nx*ny+j*nx+i-1]+A[k*nx*ny+j*nx+i+1])+C2*(A[k*nx*ny+(j-1)*nx+i]+A[k*nx*ny+(j+1)*nx+i])+C3*(A[(k-1)*nx*ny+j*nx+i]+A[(k+1)*nx*ny+j*nx+i])+C4*A[k*nx*ny+j*nx+i];
        }
      }
    }
    tmp = A_new;
    A_new = A;
    A = tmp;
  }

  t_end = gettime();
  t_total = t_end - t_start;
  flops = 13.0*nx*ny*nz*tsteps*1.0e-9;
  printf("Jacobi3D: %d threads, %d x %d x %d mesh

=======

// 3D Jacobi OpenACC function
// (c) 2016-2017 by Carlos D. Facundo
// All rights reserved
// -----------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define REAL float

void jacobi3D(REAL *u, REAL *f, REAL *uold, REAL *g, REAL *gnew, REAL *gtemp, int nx, int ny, int nz, REAL alpha, REAL beta, REAL gamma, REAL epsilon, int niter, int nthreads) {

    REAL ax, ay, az, b, resid;
    int iter, i, j, k;

    ax = alpha * ny * ny;
    ay = alpha * nx * nx;
    az = alpha * nx * ny;

    b = beta;

    #pragma acc kernels
    {
        for (i = 1; i < nx + 1; i++) {
            for (j = 1; j < ny + 1; j++) {
                for (k = 1; k < nz + 1; k++) {
                    uold[i * (ny + 2) * (nz + 2) + j * (nz + 2) + k] = u[i * (ny + 2) * (nz + 2) + j * (nz + 2) + k];
                }
            }
        }
    }

    for (iter = 1; iter <= niter; iter++) {

        #pragma acc kernels
        {
            for (i = 1; i < nx + 1; i++) {
                for (j = 1; j < ny + 1; j++) {
                    for (k = 1; k < nz + 1; k++) {
                        u[i * (ny + 2) * (nz + 2) + j * (nz + 2) + k] = (uold[i * (ny + 2) * (nz + 2) + j * (nz + 2) + k] + b * (uold[(i + 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k] + uold[(i - 1) * (ny + 2) * (nz + 2) + j * (nz + 2)

=======

// 3D Jacobi OpenACC function
// (c) 2015-2016 by Carlos D. Facundo
// All rights reserved
// ----------------------------------------------------------------------

#include "jacobi3D.h"

void jacobi3D(Real *u, Real *u_new, Real *f, Real *rho, Real *lambda, int nx, int ny, int nz, int niter, int verbose) {

  int iter, i, j, k;
  Real h2inv = 1.0 / (H * H);
  Real sixinv = 1.0 / 6.0;
  Real alpha, beta, gamma, resid, uijk;

  Real *tmp;

  for (iter = 1; iter <= niter; iter++) {

    resid = 0.0;

    // Jacobi iteration
    #pragma acc kernels
    for (i = 1; i < nx - 1; i++) {
      for (j = 1; j < ny - 1; j++) {
        for (k = 1; k < nz - 1; k++) {
          uijk = u[IX(i, j, k)];
          u_new[IX(i, j, k)] = sixinv * (h2inv * f[IX(i, j, k)] + u[IX(i - 1, j, k)] + u[IX(i + 1, j, k)] + u[IX(i, j - 1, k)] + u[IX(i, j + 1, k)] + u[IX(i, j, k - 1)] + u[IX(i, j, k + 1)]);
          resid += (u_new[IX(i, j, k)] - uijk) * (u_new[IX(i, j, k)] - uijk);
        }
      }
    }

    // update u
    tmp = u;
    u = u_new;
    u_new = tmp;

    // print residual norm
    if (verbose) {
      alpha = -h2inv;
      beta = 1.0;
      gamma = 0.0;
      resid = sqrt(resid / (nx * ny * nz));
      printf("Iteration %d, Residual norm: %e\n", iter, resid);
    }
  }
}

// ----------------------------------------------------------------------
// Path: Jacobi3D/jacobi3D_gold.cpp

=======

// 3D Jacobi OpenACC function
// Date: 2010-11-01

### End of Output

// Prompt: 3D Jacobi Kokkos
### Beginning of Output
Synthesizing 8/10 solutions (Duplicates hidden)

=======

// 3D Jacobi Kokkos version.
// This version uses a simple 1D decomposition of the 3D grid.
//
// Build with: nvcc -arch=sm_35 -I.. -I../common jacobi3D.cpp -o jacobi3D
// Run with: ./jacobi3D -n 1000 -t 1000
//

=======

// 3D Jacobi Kokkos example
// Version: 1.0
// Date: 2019-03-19

=======

// 3D Jacobi Kokkos version
// 2016.03.21: Added OpenMP 4.0 target offload to GPU
// 2017.03.21: Added OpenACC version
// 2018.03.21: Added Kokkos version
// 2019.03.21: Added Kokkos version with TBB
// 2020.03.21: Added Kokkos version with OpenMP
// 2021.03.21: Added Kokkos version with OpenMP target
// 2021.03.21: Added Kokkos version with OpenMP target and CUDA
// 2021.03.21: Added Kokkos version with OpenMP target and HIP
// 2021.03.21: Added Kokkos version with OpenMP target and SYCL
// 2021.03.21: Added Kokkos version with OpenMP target and OpenCL
// 2021.03.21: Added Kokkos version with OpenMP target and ROCm
// 2021.03.21: Added Kokkos version with OpenMP target and Qthreads
// 2021.03.21: Added Kokkos version with OpenMP target and HPX

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <Kokkos_Core.hpp>

#define N  256

// Jacobi3D kernel
template <typename T>
struct jacobi3D {
  T * A;
  T * B;
  int n;
  int iter_max;
  int iter_print;
  int iter;

  jacobi3D(T * A_, T * B_, int n_, int iter_max_, int iter_print_) :
    A(A_), B(B_), n(n_), iter_max(iter_max_), iter_print(iter_print_), iter(0) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    int j, k;
    if (i >= n) return;
    for (iter = 0; iter <= iter_max; iter++) {
      if ((iter % iter_print) == 0) printf("  iteration %d\n", iter);
      for (j = 1; j < n - 1; j++) {
        for (k = 1; k < n - 1; k++) {
          B[i * n * n + j * n + k]

=======

// 3D Jacobi Kokkos with MPI
// Version: 1.0
// Date: 2015-01-23

=======

// 3D Jacobi Kokkos Example
//

=======

// 3D Jacobi Kokkos version

=======

// 3D Jacobi Kokkos version.

### End of Output
// Prompt: 3D Jacobi Kokkos function
### Beginning of Output
Synthesizing 5/10 solutions

=======

// 3D Jacobi Kokkos function
// Date: 2019-06-14

#include <Kokkos_Core.hpp>

// 3D Jacobi function
template <class ViewType>
void jacobi3D(ViewType u, ViewType u_new, int N, int iter_max) {
  int iter = 0;
  Kokkos::parallel_for(
      "jacobi3D", N * N * N,
      KOKKOS_LAMBDA(const int index) {
        int k = index / (N * N);
        int j = (index % (N * N)) / N;
        int i = (index % (N * N)) % N;
        if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 ||
            k == N - 1) {
          u_new(k, j, i) = u(k, j, i);
        } else {
          u_new(k, j, i) =
              (u(k, j, i - 1) + u(k, j, i + 1) + u(k, j - 1, i) +
               u(k, j + 1, i) + u(k - 1, j, i) + u(k + 1, j, i)) /
              6.0;
        }
      });
  Kokkos::fence();
  iter++;
  while (iter < iter_max) {
    Kokkos::parallel_for(
        "jacobi3D", N * N * N,
        KOKKOS_LAMBDA(const int index) {
          int k = index / (N * N);
          int j = (index % (N * N)) / N;
          int i = (index % (N * N)) % N;
          if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 ||
              k == N - 1) {
            u(k, j, i) = u_new(k, j, i);
          } else {
            u(k, j, i) =
                (u_new(k, j, i - 1) + u_new(k, j, i + 1) +
                 u_new(k, j - 1, i) + u_new(k, j + 1

=======

// 3D Jacobi Kokkos function
// 2016-07-20

#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

using namespace Kokkos;

#define N 1000

typedef Kokkos::Random_XorShift64_Pool<> RandPoolType;
typedef Kokkos::Random_XorShift64_Pool<>::generator_type RandGenType;

// 3D Jacobi function
void jacobi3D(int n, int num_iterations, double *A, double *B) {
  // Initialize A and B
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        A[i * n * n + j * n + k] = 0.0;
        B[i * n * n + j * n + k] = 0.0;
      }
    }
  }

  // Initialize the boundary values
  for (int i = 0; i < n; i++) {
    A[i * n * n] = 1.0;
    A[i * n * n + (n - 1) * n + (n - 1)] = 1.0;
    B[i * n * n] = 1.0;
    B[i * n * n + (n - 1) * n + (n - 1)] = 1.0;
  }

  // Jacobi iterations
  for (int iter = 0; iter < num_iterations; iter++) {
    // Jacobi kernel
    for (int i = 1; i < n - 1; i++) {
      for (int j = 1; j < n - 1; j++) {
        for (int k = 1; k < n - 1; k++) {
          B[i * n * n + j * n + k] = 0.125 * (A[(i - 1) * n * n + j * n + k] +
                                              A[(i + 1) * n * n + j * n + k] +
                                              A[i * n * n + (j - 1) * n + k] +
                                              A[i * n * n + (j + 1) * n + k]

=======

// 3D Jacobi Kokkos function
// 2018-12-03

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

#include <Kokkos_Core.hpp>

// 3D Jacobi kernel
template <class ViewType>
struct Jacobi3D {
  typedef typename ViewType::value_type Scalar;
  typedef typename ViewType::size_type SizeType;
  ViewType u, u0;
  Scalar h2;
  SizeType nx, ny, nz;

  Jacobi3D(ViewType u_, ViewType u0_, Scalar h2_, SizeType nx_, SizeType ny_, SizeType nz_) :
    u(u_), u0(u0_), h2(h2_), nx(nx_), ny(ny_), nz(nz_) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const SizeType i) const {
    SizeType i0 = i + 1;
    SizeType i1 = i - 1;
    SizeType j = i / nx;
    SizeType j0 = j + 1;
    SizeType j1 = j - 1;
    SizeType k = j / ny;
    SizeType k0 = k + 1;
    SizeType k1 = k - 1;
    SizeType ijk = i + j * nx + k * nx * ny;
    SizeType i0jk = i0 + j * nx + k * nx * ny;
    SizeType i1jk = i1 + j * nx + k * nx * ny;
    SizeType ij0k = i + j0 * nx + k * nx * ny;
    SizeType ij1k = i + j1 * nx + k * nx * ny;
    SizeType ijk0 = i + j * nx + k0 * nx * ny;
    SizeType ijk1 = i + j * nx + k1 * nx * ny;
    if (i0 < nx - 1 && j0 < ny - 1 && k0 < nz - 1) {
      u(ijk) = (u0(i0jk) + u0(i1jk) + u0(ij0k) + u0(ij1k) + u0(ijk0) + u0(ijk1)) * h2;
    }
  }
};

// 3D Jacobi kernel
template <class View

=======

// 3D Jacobi Kokkos functional test
// 2018-12-03 13:59:30.000000000 -0600

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <cstdlib>

#include "KokkosBlas1_scal.hpp"
#include "KokkosBlas1_axpby.hpp"
#include "KokkosBlas1_nrm2.hpp"

#include "KokkosSparse_spmv.hpp"
#include "KokkosSparse_CrsMatrix.hpp"

using namespace KokkosBlas;

typedef Kokkos::View<double**,Kokkos::LayoutLeft,Kokkos::Cuda> view_type;
typedef Kokkos::View<double*,Kokkos::LayoutLeft,Kokkos::Cuda> view_type_1D;
typedef KokkosSparse::CrsMatrix<double,int,Kokkos::Cuda> crs_matrix_type;

int main(int argc, char* argv[]) {

  Kokkos::initialize(argc, argv);

  int N = 1000;
  int iterations = 1000;

  view_type x("x", N, N);
  view_type y("y", N, N);
  view_type z("z", N, N);

  view_type b("b", N, N);
  view_type r("r", N, N);
  view_type p("p", N, N);

  view_type_1D diag("diag", N);

  Kokkos::parallel_for(N, KOKKOS_LAMBDA (int i) {
    for (int j = 0; j < N; j++) {
      x(i,j) = 0;
      y(i,j) = 0;
      z(i,j) = 0;
      b(i,j) = 1;
      r(i,j) = 1;
      p(i,j) = 1;
    }
  });

  Kokkos::parallel_for(N, KOKKOS_LAMBDA (int i) {
    diag(i) = 6.0;
    if (i > 0) diag(i) += 1.0;
    if (i < N-1) diag(i) += 1.0;
  });

  Kokkos::fence();

  // Create Kokkos sparse matrix
  Kokkos::View<int*, Kokkos::Cuda> row_map("row_map", N+1);
  Kokkos::View<int*, Kokkos::

=======

// 3D Jacobi Kokkos function for the 3D Laplace equation

### End of Output
// Prompt: 3D Jacobi CUDA
### Beginning of Output
Synthesizing 9/10 solutions (Duplicates hidden)

=======

// 3D Jacobi CUDA C/C++ implementation.
// This file is part of the GROMACS molecular simulation package.

=======

// 3D Jacobi CUDA C/C++ implementation

=======

// 3D Jacobi CUDA C++ implementation

=======

// 3D Jacobi CUDA C++ test program

=======

// 3D Jacobi CUDA C++ Implementation

=======

// 3D Jacobi CUDA kernel

=======

// 3D Jacobi CUDA Kernel

=======

// 3D Jacobi CUDA implementation

### End of Output
// Prompt: 3D Jacobi CUDA function
### Beginning of Output
Synthesizing 5/10 solutions

=======

// 3D Jacobi CUDA function
//

=======

// 3D Jacobi CUDA functions
// ---------------------------------------------------------
// ---------------------------------------------------------
// ---------------------------------------------------------
// ---------------------------------------------------------
// ---------------------------------------------------------

#include "jacobi3D.h"

#include <stdio.h>

#define BLOCK_DIM 16

// ---------------------------------------------------------
// ---------------------------------------------------------
// ---------------------------------------------------------
// ---------------------------------------------------------
// ---------------------------------------------------------
// ---------------------------------------------------------
// ---------------------------------------------------------
// ---

=======

// 3D Jacobi CUDA functions
// 

#include "jacobi3D.h"

#include <stdio.h>

#include "jacobi3D_kernel.cu"

// Host code

// Initialize the input data
void init_data(unsigned int size, float *data)
{
    for (unsigned int i = 0; i < size; i++) {
        data[i] = 1.0f;
    }
}

// Main function
int main(int argc, char **argv)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int n = 256;
    printf("[Jacobi 3D] - Starting...

");
    printf("Vector size = %d

", n);

    size_t size = n * n * n * sizeof(float);
    printf("Allocating memory for input data of size %u MB
", (unsigned int)(size / (1024 * 1024)));

    // Allocate the input data vector in host memory
    float *h_data = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_data == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!

");
        exit(EXIT_FAILURE);
    }

    // Initialize the input data
    init_data(n * n * n, h_data);

    // Allocate the device input vector A
    float *d_data = NULL;
    err = cudaMalloc((void **)&d_data, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!

", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device

");
    err = cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!

", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n * n * n + threadsPerBlock - 1) / threadsPerBlock

=======

// 3D Jacobi CUDA function
// (c) 2011, Viktor K. Decyk, UCLA

#include "jacobi3D.h"

#define NITER  5

/*--------------------------------------------------------------------*/
void init3(float *a, float *b, int nx, int ny, int nz) {
/* initialize arrays a and b */
   int i, j, k, nxh, nyh, nzh, nxhyz;
   float x, y, z, pi;
   nxh = nx/2;
   nyh = ny/2;
   nzh = nz/2;
   nxhyz = nx*ny*nz;
   pi = 4.0f*atanf(1.0f);
   for (k = 0; k < nz; k++) {
      z = (float) k;
      if (k > nzh)
         z = z - (float) nz;
      z = z/(float) nzh;
      for (j = 0; j < ny; j++) {
         y = (float) j;
         if (j > nyh)
            y = y - (float) ny;
         y = y/(float) nyh;
         for (i = 0; i < nx; i++) {
            x = (float) i;
            if (i > nxh)
               x = x - (float) nx;
            x = x/(float) nxh;
            a[i+nx*(j+ny*k)] = sinf(pi*x)*sinf(pi*y)*sinf(pi*z);
            b[i+nx*(j+ny*k)] = 0.0f;
         }
      }
   }
   return;
}

/*--------------------------------------------------------------------*/
void zero3(float *a, int nx, int ny, int nz) {
/* zero out array a */
   int i, nxhyz;
   nxhyz = nx*ny*nz;
   for (i = 0; i < nxhyz; i++) {
      a[i] = 0.0f;
   }
   return;
}

/*--------------------------------------------------------------------*/
__global__ void l2norm3(float *a, float *ssum, int nx, int ny, int nz,
                        int nxvh, int nyv, int nzv) {
/* local sum squared difference of array elements */
/* local data */
   int j, k, j1, k

=======

// 3D Jacobi CUDA function.  It is assumed that the data is
// padded to 16 bytes (4 floats) in each dimension.
// This function is called by jacobi3D_kernel_wrapper.
// Parameters:
//   u: 3D array of input values
//   u_new: 3D array of output values
//   nx, ny, nz: dimensions of the arrays
//   niter: number of iterations to perform
//   nx, ny, nz: dimensions of the arrays
//   niter: number of iterations to perform
//   nx, ny, nz: dimensions of the arrays
//   niter: number of iterations to perform
//   nx, ny, nz: dimensions of the arrays
//   niter: number of iterations to perform
//   nx, ny, nz: dimensions of the arrays
//   niter: number of iterations to perform
//   nx, ny, nz: dimensions of the arrays
//   niter: number of iterations to perform
//   nx, ny, nz: dimensions of the arrays
//   niter: number of iterations to perform

//   u: 3D array of input values
//   u_new: 3D array of output values
//   nx, ny, nz: dimensions of the arrays
//   niter: number of iterations to perform
//   nx, ny, nz: dimensions of the arrays
//   niter: number of iterations to perform
//   nx, ny, nz: dimensions of the arrays
//   niter: number of iterations to perform
//   nx, ny, nz: dimensions of the arrays
//   niter: number of iterations to perform
//   nx, ny, nz: dimensions of the arrays
//   niter: number of iterations to perform
//   nx, ny, nz: dimensions of the arrays
//   niter: number of iterations to perform
//   nx, ny, nz: dimensions of the arrays
//   niter: number of iterations to perform
//   nx, ny, nz: dimensions of the arrays
//   niter: number of iterations to perform
//   nx, ny, nz: dimensions of the arrays
//   niter: number of iterations to perform
//   nx, ny, nz: dimensions of the arrays
//   niter: number of iterations to perform

//

### End of Output

// Prompt: 3D Jacobi HIP
### Beginning of Output
Synthesizing 10/10 solutions (Duplicates hidden)

=======

// 3D Jacobi HIP version
// Device code.
//
// Build with: hipcc -O3 jacobi3D.cpp -o jacobi3D
//
// Run with: ./jacobi3D
//
// No correctness checking is performed
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hip/hip_runtime.h>

#define NX 2048
#define NY 2048
#define NZ 2048

#define BLOCK_SIZE 16

// 3D Jacobi kernel
__global__ void jacobi3D(float *A, float *B, int nx, int ny, int nz)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  int index = i + j*nx + k*nx*ny;

  if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {
    B[index] = 0.125f * ( A[index+1] + A[index-1] +
                          A[index+nx] + A[index-nx] +
                          A[index+nx*ny] + A[index-nx*ny] +
                          A[index] );
  }
}

// Initialize A and B to random numbers between 0 and 1
void init(float *A, float *B, int nx, int ny, int nz)
{
  int i, j, k;
  for (k = 0; k < nz; k++) {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
        A[i + j*nx + k*nx*ny] = rand() / (float)RAND_MAX;
        B[i + j*nx + k*nx*ny] = rand() / (float)RAND_MAX;
      }
    }
  }
}

// Main program
int main(int argc, char *argv[])
{
  float *A, *B, *d_A, *d_B;
  int i, j, k;
  int nxy = NX * NY;
  int nxyz = NX * NY * NZ;
  int niter = 1000;
  size_t size = nxyz

=======

// 3D Jacobi HIP example
// This example implements the Jacobi iteration to solve the
// Laplace equation on a 3D grid. It is used as a basic example
// of the HIP programming model.
// The Jacobi iteration is executed on a 3D grid of points
// in space. The boundary points are fixed at 0.0. The interior
// points are updated using the average of the surrounding
// points. The Jacobi iteration is repeated until the maximum
// change in any interior point is less than a specified
// threshold. The number of iterations and the threshold are
// command line parameters.
// The 3D grid is stored as a one dimensional array of size
// N x N x N. The indices of the neighbor points are computed
// from the index of the central point. The Jacobi iteration
// is executed on the GPU using a 1D grid of 1D blocks.
// The GPU kernel is executed with a 1D grid of 1D blocks.
// The size of the blocks is set to 256 threads. This is
// the maximum number of threads per block for a 2.0 device.
// The number of blocks is computed from the size of the grid.
// The size of the grid is the number of interior points.
// The number of blocks is computed as the ceiling of the
// grid size divided by the block size.
// The GPU kernel is executed with a 1D grid of 1D blocks.
// The size of the blocks is set to 256 threads. This is
// the maximum number of threads per block for a 2.0 device.
// The number of blocks is computed from the size of the grid.
// The size of the grid is the number of interior points.
// The number of blocks is computed as the ceiling of the
// grid size divided by the block size.
// The 3D grid is stored as a one dimensional array of size
// N x N x N. The indices of the neighbor points are computed
// from the index of the central point. The Jacobi iteration
// is executed on the GPU using a 1D grid of 1D blocks.
// The GPU kernel is executed with a 1D grid of 1D blocks.
// The size of the blocks is set to 256 threads. This is
// the maximum number of threads per block for a 2.

=======

// 3D Jacobi HIP C++ implementation

=======

// 3D Jacobi HIP sample
// Device code.
//

=======

// 3D Jacobi HIP implementation

=======

// 3D Jacobi HIP Implementation

=======

// 3D Jacobi HIP example

=======

// 3D Jacobi HIP Code

### End of Output

// Prompt: 3D Jacobi HIP function
### Beginning of Output
iSynthesizing 5/10 solutions (Duplicates hidden)

=======

// 3D Jacobi HIP function
//

=======

// 3D Jacobi HIP function
// Function: Jacobi3D
// Description: 3D Jacobi HIP function

=======

// 3D Jacobi HIP functions
### End of Output
// Prompt: 3D Jacobi Thrust
### Beginning of Output
Synthesizing 9/10 solutions (Duplicates hidden)

=======

// 3D Jacobi Thrust example
// nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -o jacobi3D jacobi3D.cpp -lcudart -lthrust
// ./jacobi3D

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>

#define BLOCK_SIZE 32

#define EPSILON 0.001f
#define MAX_ITERATIONS 100

// CUDA kernel
__global__ void jacobi_kernel(float *A, float *B, int N, int M) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (i < N && j < M && k < M) {
    float sum = 0;
    if (i > 0) {
      sum += A[(i-1)*M*M + j*M + k];
    }
    if (i < N-1) {
      sum += A[(i+1)*M*M + j*M + k];
    }
    if (j > 0) {
      sum += A[i*M*M + (j-1)*M + k];
    }
    if (j < M-1) {
      sum += A[i*M*M + (j+1)*M + k];
    }
    if (k > 0) {
      sum += A[i*M*M + j*M + (k-1)];
    }
    if (k < M-1) {
      sum += A[i*M*M + j*M + (k+1)];
    }
    B[i*M*M + j*M + k] = sum / 6;
  }
}

// CPU kernel
void jacobi(float *A, float *B, int N, int M) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      for (int k = 0; k < M; k++) {
        float sum = 0;
        if (i > 0) {
          sum +=

=======

// 3D Jacobi Thrust
// 2015 (c) Vasily Volkov

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/tuple.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/tuple.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/tuple.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/tuple.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/tuple.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/tuple.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/tuple

=======

// 3D Jacobi Thrust code
// This code is a modification of the 2D Jacobi code
// by John D. McCalpin
// http://www.cs.virginia.edu/stream/ref.html
// Ported to CUDA Thrust by: Tom Deakin
// University of Bristol
// July 2012
// --------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#define MAXERROR 0.1
#define MAXITER 1000

// --------------------------------------------------------
// Kernel to compute the residual
// --------------------------------------------------------

struct residual : public thrust::binary_function<double, double, double>
{
    __host__ __device__
    double operator()(const double& x, const double& y) const
    {
        return fabs(x - y);
    }
};

// --------------------------------------------------------
// Kernel to compute the residual
// --------------------------------------------------------

struct update : public thrust::binary_function<double, double, double>
{
    __host__ __device__
    double operator()(const double& x, const double& y) const
    {
        return 0.125 * (x + y);
    }
};

// --------------------------------------------------------
// Main
// --------------------------------------------------------

int main(int argc, char* argv[])
{
    int i, j, k, iter;
    int N = atoi(argv[1]);
    double tol = MAXERROR * N * N;
    double error = 10 * tol;
    double *A, *B;
    double *d_A, *d_B;

    // Allocate memory on the host
    A = (double *)malloc(N * N * N * sizeof(double));
    B = (double *)malloc(N * N * N * sizeof(double));

    // Allocate memory on the device
    cudaMalloc((void **) &d_A, N * N * N * sizeof(double));
    cudaMalloc((void **) &d_B, N * N * N * sizeof(double));

    // Initialise the host memory
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (k = 0; k < N; k++)
            {
                A[i * N * N + j * N + k] =

=======

// 3D Jacobi Thrust Example
// This example demonstrates how to use Thrust to solve a 3D Jacobi iteration
// problem.
// The Jacobi iteration is a common way to solve a system of linear equations
// for a matrix A and a vector b. It is very similar to the Gauss-Seidel
// iteration, except that the Jacobi iteration updates each element of the
// solution vector x using the values of the previous iteration. The Gauss-
// Seidel iteration updates each element of the solution vector x using the
// values of the current iteration. The Jacobi iteration is more stable than
// the Gauss-Seidel iteration, but is slower.
// The Jacobi iteration is defined as follows:
// x = inv(D) * (b - (L + U) * x)
// where D is the diagonal of A, L is the lower triangular part of A, and U is
// the upper triangular part of A.
// The Jacobi iteration can be used to solve the following system of linear
// equations:
// A * x = b
// where A is a matrix, x is the solution vector, and b is the constant vector.
// The Jacobi iteration can also be used to solve the following system of
// linear equations:
// A * x = b
// where A is a matrix, x is the solution vector, and b is the constant vector.
// The Jacobi iteration can also be used to solve the following system of
// linear equations:
// A * x = b
// where A is a matrix, x is the solution vector, and b is the constant vector.
// The Jacobi iteration can also be used to solve the following system of
// linear equations:
// A * x = b
// where A is a matrix, x is the solution vector, and b is the constant vector.
// The Jacobi iteration can also be used to solve the following system of
// linear equations:
// A * x = b
// where A is a matrix, x is the solution vector, and b is the constant vector.
// The Jacobi iteration can also be used to solve the following system of
// linear equations:
// A * x = b
// where A is a matrix, x is the solution vector, and b is the constant vector.
// The Jacobi iteration can also be used to solve the following system of
// linear equations:
// A * x = b

=======

// 3D Jacobi Thrust implementation

=======

// 3D Jacobi Thrust code

=======

// 3D Jacobi Thrust Implementation

=======

// 3D Jacobi Thrust benchmark

### End of Output
// Prompt: 3D Jacobi Thrust function
### Beginning of Output
Synthesizing 2/10 solutions

=======

// 3D Jacobi Thrust function
// 2011-04-05

#include "jacobi3D.h"

void jacobi3D(float *d_u, float *d_unew, int dimx, int dimy, int dimz, int iter)
{
    int i;
    for (i = 0; i < iter; i++)
    {
        thrust::device_ptr<float> u(d_u);
        thrust::device_ptr<float> unew(d_unew);
        thrust::transform(u, u + dimx*dimy*dimz, unew, jacobi3D_functor(dimx, dimy, dimz));
        thrust::swap(u, unew);
    }
}

// Path: Jacobi3D/jacobi3D.cpp
// 3D Jacobi Thrust function
// 2011-04-05

#include "jacobi3D.h"

void jacobi3D(float *d_u, float *d_unew, int dimx, int dimy, int dimz, int iter)
{
    int i;
    for (i = 0; i < iter; i++)
    {
        thrust::device_ptr<float> u(d_u);
        thrust::device_ptr<float> unew(d_unew);
        thrust::transform(u, u + dimx*dimy*dimz, unew, jacobi3D_functor(dimx, dimy, dimz));
        thrust::swap(u, unew);
    }
}

// Path: Jacobi3D/jacobi3D.cpp
// 3D Jacobi Thrust function
// 2011-04-05

#include "jacobi3D.h"

void jacobi3D(float *d_u, float *d_unew, int dimx, int dimy, int dimz, int iter)
{
    int i;
    for (i = 0; i < iter; i++)
    {
        thrust::device_ptr<float> u(d_u);
        thrust::device_ptr<float> unew(d_unew);
        thrust::transform(u, u + dimx*dimy*dimz, unew, jacobi3D_functor(dimx, dimy, dimz));
        thrust::swap(u, unew);
    }
}

// Path: Jacobi3D/jacobi3D.cpp
// 3D Jacobi Thrust function
// 2011-04

=======

// 3D Jacobi Thrust function
// Version: 1.0

### End of Output
