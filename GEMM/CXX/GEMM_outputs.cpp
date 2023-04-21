
// Prompt: gemm OpenMP
Synthesizing 10 / 10 solutions

=======

    // gemm OpenMP

    void
    gemm_omp(double *A, double *B, double *C, int n, int m, int k) {

#pragma omp parallel for

  for (int i = 0; i < n; i++)

    for (int j = 0; j < m; j++)

      for (int l = 0; l < k; l++)

        C[i * m + j] += A[i * k + l] * B[l * m + j];
}

// gemm OpenCL

void gemm_ocl(double *A, double *B, double *C, int n, int m, int k) {

  // Create the buffers

  cl::Buffer bufA(context, CL_MEM_READ_ONLY, sizeof(double) * n * k);

  cl::Buffer bufB(context, CL_MEM_READ_ONLY, sizeof(double) * k * m);

  cl::Buffer bufC(context, CL_MEM_WRITE_ONLY, sizeof(double) * n * m);

  // Copy data to the buffers

  queue.enqueueWriteBuffer(bufA, CL_TRUE, 0, sizeof(double) * n * k, A);

  queue.enqueueWriteBuffer(bufB, CL_TRUE, 0, sizeof(double) * k * m, B);

  queue.enqueueWriteBuffer(bufC, CL_TRUE, 0, sizeof(double) * n * m, C);

  // Set the arguments

  kernel.setArg(0, bufA);

  kernel.setArg(1, bufB);

  kernel.setArg(2, bufC);

  kernel.setArg(3, n);

  kernel.setArg(4, m);

  kernel.setArg(5, k);

  // Run the kernel

  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n, m));

  // Read the result

  queue.enqueueReadBuffer(bufC, CL_TRUE, 0, sizeof(double) * n * m, C);

  // Release the buffers

  queue.finish();
}

// gemm CUBLAS

void gemm_cublas(double *A, double *B, double *C, int n, int m, int k) {

  // Copy data to the device

  cudaMemcpy(d_A, A, sizeof(double) * n * k, cudaMemcpyHostToDevice);

  cudaMemcpy(d_B, B, sizeof(double) * k * m, cudaMemcpyHostToDevice);

  cudaMemcpy(d_C, C, sizeof(double) * n * m, cudaMemcpyHostToDevice);

  // Perform the multiplication

  c

=======

      // gemm OpenMP version

      // Path: GEMM/CXX/GEMM_OMP.cpp

      // gemm OpenCL version

      // Path: GEMM/CXX/GEMM_CL.cpp

      // gemm cuBLAS version

      // Path: GEMM/CXX/GEMM_CUBLAS.cpp

      // gemm cuBLAS version with double precision

      // Path: GEMM/CXX/GEMM_CUBLAS_DP.cpp

      // gemm cuBLAS version with double precision and 64-bit integers

      // Path: GEMM/CXX/GEMM_CUBLAS_DP64.cpp

      // gemm cuBLAS version with double precision and 64-bit integers

      // Path: GEMM/CXX/GEMM_CUBLAS_DP64.cpp

      // gemm cuBLAS version with double precision and 64-bit integers

      // Path: GEMM/CXX/GEMM_CUBLAS_DP64.cpp

      // gemm cuBLAS version with double precision and 64-bit integers

      // Path: GEMM/CXX/GEMM_CUBLAS_DP64.cpp

      // gemm cuBLAS version with double precision and 64-bit integers

      // Path: GEMM/CXX/GEMM_CUBLAS_DP64.cpp

      // gemm cuBLAS version with double precision and 64-bit integers

      // Path: GEMM/CXX/GEMM_CUBLAS_DP64.cpp

      // gemm cuBLAS version with double precision and 64-bit integers

      // Path: GEMM/CXX/GEMM_CUBLAS_DP64.cpp

      // gemm cuBLAS version with double precision and 64-bit integers

      // Path: GEMM/CXX/GEMM_CUBLAS_DP64.cpp

      // gemm cuBLAS version with double precision and 64-bit integers

      // Path: GEMM/CXX/GEMM_CUBLAS_DP64.cpp

      // gemm cuBLAS version with double precision and 64-bit integers

      // Path: GEMM/CXX/GEMM_CUBLAS_DP64.cpp

      // gemm cuBLAS version with double precision and 64-bit integers

      // Path: GEMM/CXX/GEMM_CUBLAS_DP64.cpp

      // gemm cuBLAS version with double precision and 64-bit integers

      // Path: GEMM/CXX/GEMM_CUBLAS_DP64.cpp

      //

=======

      // gemm OpenMP

      void
      gemm_omp(const int N, const float *A, const float *B, float *C)

  {

#pragma omp parallel for

    for (int i = 0; i < N; ++i)

    {

      for (int j = 0; j < N; ++j)

      {

        float cij = C[i * N + j];

        for (int k = 0; k < N; ++k)

        {

          cij += A[i * N + k] * B[k * N + j];
        }

        C[i * N + j] = cij;
      }
    }
  }

=======

  // gemm OpenMP kernel

  void gemm_omp(int m, int n, int k, double *A, double *B, double *C) {

    int i, j, l;

    double sum;

#pragma omp parallel for private(i, j, l, sum)

    for (i = 0; i < m; i++) {

      for (j = 0; j < n; j++) {

        sum = 0.0;

        for (l = 0; l < k; l++) {

          sum += A[i * k + l] * B[l * n + j];
        }

        C[i * n + j] = sum;
      }
    }
  }

  // Path: GEMM/CXX/GEMM.cpp

=======

  // gemm OpenMP version
  void gemmOMP(int M, int N, int K, double *A, double *B, double *C) {
    int i, j, k;
    double sum;
#pragma omp parallel for private(i, j, k, sum)
    for (i = 0; i < M; i++) {
      for (j = 0; j < N; j++) {
        sum = 0;
        for (k = 0; k < K; k++) {
          sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
      }
    }
  }

  // gemm OpenMP SIMD version
  void gemmSIMD(int M, int N, int K, double *A, double *B, double *C) {
    int i, j, k;
    double sum;
#pragma omp parallel for private(i, j, k, sum)
    for (i = 0; i < M; i++) {
      for (j = 0; j < N; j++) {
        sum = 0;
        for (k = 0; k < K; k++) {
          sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
      }
    }
  }

  // gemm OpenMP SIMD version
  void gemmSIMD2(int M, int N, int K, double *A, double *B, double *C) {
    int i, j, k;
    double sum;
#pragma omp parallel for private(i, j, k, sum)
    for (i = 0; i < M; i++) {
      for (j = 0; j < N; j++) {
        sum = 0;
        for (k = 0; k < K; k++) {
          sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
      }
    }
  }

  // gemm OpenMP SIMD version
  void gemmSIMD3(int M, int N, int K, double *A, double *B, double *C) {
    int i, j, k;
    double sum;
#pragma omp parallel for private(i, j, k, sum)
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j

=======

// gemm OpenMP version

void gemm_omp(const double* A, const double* B, double* C, int size) {

=======

        // gemm OpenMP

#include "GEMM.h"

#include <omp.h>

#include <iostream>

#include <iomanip>

#include <cstdlib>

#include <ctime>

#include <cmath>

#include <string>

#include <fstream>
        using namespace std;

        int main(int argc, char *argv[])

        {

          // Check for correct number of arguments

          if (argc != 2)

          {

            cout << "Usage: " << argv[0] << " <matrix size>" << endl;

            return -1;
          }

          // Matrix size

          int N = atoi(argv[1]);

          // Number of threads

          int numThreads = omp_get_max_threads();

          cout << "Number of threads: " << numThreads << endl;

          // Allocate matrices

          double **A = new double *[N];

          double **B = new double *[N];

          double **C = new double *[N];

          for (int i = 0; i < N; i++)

          {

            A[i] = new double[N];

            B[i] = new double[N];

            C[i] = new double[N];
          }

          // Initialize matrices

          for (int i = 0; i < N; i++)

          {

            for (int j = 0; j < N; j++)

            {

              A[i][j] = i + j;

              B[i][j] = i - j;

              C[i][j] = 0.0;
            }
          }

          // Start timer

          clock_t start = clock();

            // Calculate C = A * B

#pragma omp parallel for

          for (int i = 0; i < N; i++)

          {

            for (int j = 0; j < N; j++)

            {

              for (int k = 0; k < N; k++)

              {

                C[i][j] += A[i][k] * B[k][j];
              }
            }
          }

          // Stop timer

          clock_t stop = clock();

          // Compute time

          double time = (stop - start) / (double)CLOCKS_PER_SEC;

          // Print time

          cout << "Time: " << time << " seconds" << endl;

          // Compute and print GFLOPS

          cout << "GFLOPS: " << 2.0 * N * N * N / time / 1e9 << endl;

          // Free memory

          for (int i = 0; i < N; i++)

          {

            delete[] A[i];

            delete[] B[i];

            delete[] C[i];
          }

          delete[] A;

          delete[] B;

=======

          // gemm OpenMP C++ version

          // Path: GEMM/CXX/GEMM_OpenMP.cpp

=======

          // gemm OpenMP: GEMM OpenMP implementation

          // Path: GEMM/OMP/GEMM.cpp

=======

          // gemm OpenMP version

          // NOTE: This is a very naive implementation of GEMM. It is not
          // optimized for

          // performance. It is intended to be simple and easy to understand.

          // This version of GEMM uses OpenMP to parallelize the outer loop of
          // the GEMM

          // algorithm. The code is written in a way that makes it easy to
          // parallelize

          // the inner loop as well.

          // Prompt: gemm OpenACC
          Synthesizing 10 / 10 solutions

=======

              // gemm OpenACC version

              void
              gemm(const int m, const int n, const int k, const double alpha,

                   const double *A, const int lda, const double *B,
                   const int ldb,

                   const double beta, double *C, const int ldc) {

            // A: m x k matrix

            // B: k x n matrix

            // C: m x n matrix

            // C = alpha * A * B + beta * C

            // A, B, C are column-major

            // m, n, k are the number of rows and columns of A, B, C
            // respectively

            // lda, ldb, ldc are the leading dimension of A, B, C respectively

            // lda, ldb, ldc are the number of rows of A, B, C respectively

            // A is m x k matrix

            // lda is the number of rows of A

=======

            // gemm OpenACC

            // Path: GEMM/OpenACC/gemm.c

            // Path: GEMM/OpenACC/gemm.h

            // Path: GEMM/OpenACC/gemm_kernel.c

            // Path: GEMM/OpenACC/gemm_kernel.h

            // Path: GEMM/OpenACC/gemm_kernel_acc.c

            // Path: GEMM/OpenACC/gemm_kernel_acc.h

            // Path: GEMM/OpenACC/gemm_kernel_acc2.c

            // Path: GEMM/OpenACC/gemm_kernel_acc2.h

            // Path: GEMM/OpenACC/gemm_kernel_acc3.c

            // Path: GEMM/OpenACC/gemm_kernel_acc3.h

            // Path: GEMM/OpenACC/gemm_kernel_acc4.c

            // Path: GEMM/OpenACC/gemm_kernel_acc4.h

            // Path: GEMM/OpenACC/gemm_kernel_acc5.c

            // Path: GEMM/OpenACC/gemm_kernel_acc5.h

            // Path: GEMM/OpenACC/gemm_kernel_acc6.c

            // Path: GEMM/OpenACC/gemm_kernel_acc6.h

            // Path: GEMM/OpenACC/gemm_kernel_acc7.c

            // Path: GEMM/OpenACC/gemm_kernel_acc7.h

            // Path: GEMM/OpenACC/gemm_kernel_acc8.c

            // Path: GEMM/OpenACC/gemm_kernel_acc8.h

            // Path: GEMM/OpenACC/gemm_kernel_acc9.c

            // Path: GEMM/OpenACC/gemm_kernel_acc9.h

            // Path: GEMM/OpenACC/gemm_kernel_acc10.c

            // Path: GEMM/OpenACC/gemm_kernel_acc10.h

            // Path: GEMM/OpenACC/gemm_kernel_acc11.c

            // Path: GEMM/OpenACC/gemm_kernel_acc11.h

            // Path: GEMM/OpenACC/gemm_kernel_acc12.c

            // Path: GEMM/OpenACC/gemm_kernel_acc12.h

            // Path: GEMM/OpenACC/gemm_kernel_acc13.c

            // Path: GEMM/OpenACC/gemm_kernel_acc13.h

            // Path: GEMM/OpenACC/gemm_kernel_acc14.c

            // Path: GEMM/OpenACC/gemm_kernel_acc14.h

            // Path: GEMM/OpenACC/gemm_kernel_acc15.c

            // Path: GEMM/OpenACC/gemm_kernel_acc15.h

=======

            // gemm OpenACC version

            void gemm_acc(const double *A, const double *B, double *C,
                          const int m, const int n, const int k) {

              const double alpha = 1.0;

              const double beta = 1.0;

#pragma acc data copyin(A [0:m * k], B [0:k * n]) copy(C [0:m * n])

#pragma acc kernels loop independent

              for (int i = 0; i < m; ++i) {

                for (int j = 0; j < n; ++j) {

                  double cij = C[i * n + j];

                  for (int l = 0; l < k; ++l) {

                    cij += A[i * k + l] * B[l * n + j];
                  }

                  C[i * n + j] = cij;
                }
              }
            }

            // Path: GEMM/CXX/GEMM.cpp

=======

            // gemm OpenACC version

            void gemm_openacc(double *A, double *B, double *C, int n) {

              int i, j, k;

#pragma acc kernels

              for (i = 0; i < n; i++) {

                for (j = 0; j < n; j++) {

                  for (k = 0; k < n; k++) {

                    C[i * n + j] += A[i * n + k] * B[k * n + j];
                  }
                }
              }
            }

            // Path: GEMM/CXX/GEMM.cpp

=======

            // gemm OpenACC version
            void gemm(int m, int n, int k, double alpha, double *A, double *B,
                      double beta, double *C) {
#pragma acc kernels loop independent
              for (int i = 0; i < m; i++) {
#pragma acc loop independent
                for (int j = 0; j < n; j++) {
                  double cij = C[i + j * m] * beta;
#pragma acc loop independent reduction(+ : cij)
                  for (int l = 0; l < k; l++) {
                    cij += alpha * A[i + l * m] * B[l + j * k];
                  }
                  C[i + j * m] = cij;
                }
              }
            }

            // gemm OpenMP version
            void gemm_omp(int m, int n, int k, double alpha, double *A,
                          double *B, double beta, double *C) {
#pragma omp parallel for
              for (int i = 0; i < m; i++) {
#pragma omp parallel for
                for (int j = 0; j < n; j++) {
                  double cij = C[i + j * m] * beta;
#pragma omp parallel for reduction(+ : cij)
                  for (int l = 0; l < k; l++) {
                    cij += alpha * A[i + l * m] * B[l + j * k];
                  }
                  C[i + j * m] = cij;
                }
              }
            }

            // gemm OpenMP+OpenACC version
            void gemm_omp_acc(int m, int n, int k, double alpha, double *A,
                              double *B, double beta, double *C) {
#pragma omp parallel for
              for (int i = 0; i < m; i++) {
#pragma omp parallel for
                for (int j = 0; j < n; j++) {
                  double cij = C[i + j * m] * beta;
#pragma omp parallel for reduction(+ : cij)
                  for (int l = 0; l < k; l++) {
                    cij += alpha * A[i + l * m] * B[l + j * k];
                  }
                  C[i + j * m] = cij;
                }
              }
            }

            // gemm CPU version

=======

            // gemm OpenACC version

            void gemm_openacc(float *C, float *A, float *B, int m, int n,
                              int k) {

=======

              // gemm OpenACC version

              void gemm_acc(double *C, double *A, double *B, int n) {

                int i, j, k;

                int block_size = 32;

#pragma acc kernels copyin(A [0:n * n], B [0:n * n]), copyout(C [0:n * n])

#pragma acc loop independent

                for (i = 0; i < n; i++) {

                  for (j = 0; j < n; j++) {

                    double tmp = 0;

                    for (k = 0; k < n; k++) {

                      tmp += A[i * n + k] * B[k * n + j];
                    }

                    C[i * n + j] = tmp;
                  }
                }
              }

              // Path: GEMM/CXX/GEMM.cpp

=======

              // gemm OpenACC version
              // Path: GEMM/CXX/GEMM_OPENACC.cpp

=======

              // gemm OpenACC version, with data partitioning in the m dimension

              void gemm_acc_m_partitioned(int m, int n, int k, const float *A,
                                          const float *B, float *C) {

#pragma acc data copyin(A [0:m * k], B [0:k * n]) copy(C [0:m * n])

                {

                  int m_block_size = 16;

                  int n_block_size = 16;

                  int k_block_size = 16;

#pragma acc kernels loop gang(m / m_block_size, n / n_block_size) vector(8)

                  for (int j = 0; j < n; j += n_block_size) {

                    for (int i = 0; i < m; i += m_block_size) {

                      for (int l = 0; l < k; l += k_block_size) {

#pragma acc loop seq

                        for (int jj = j; jj < j + n_block_size; ++jj) {

                          for (int ii = i; ii < i + m_block_size; ++ii) {

                            for (int ll = l; ll < l + k_block_size; ++ll) {

                              C[ii + jj * m] += A[ii + ll * m] * B[ll + jj * k];
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }

              // Path: GEMM/CXX/GEMM.cpp

=======

              // gemm OpenACC C++ version

              // Prompt: gemm Kokkos
              Synthesizing 9 / 10 solutions(Duplicates hidden)

=======

                  // gemm KokkosKernels interface

                  void

                  gemm(

                      const char *transA, const char *transB,

                      const int m, const int n, const int k,

                      const double alpha,

                      const double *A, const int lda,

                      const double *B, const int ldb,

                      const double beta,

                      double *C, const int ldc)

              {

                // Create Kokkos::Views from input data

                using Kokkos::View;

                using Kokkos::LayoutLeft;

                typedef View<double *, LayoutLeft> vector_type;

                typedef View<double **, LayoutLeft> matrix_type;

                vector_type A_view(A, lda * k);

                vector_type B_view(B, ldb * n);

                vector_type C_view(C, ldc * n);

                // Create KokkosKernels::Handle

                using KokkosKernels::Experimental::KokkosKernelsHandle;

                KokkosKernelsHandle<double, int, int> handle;

                // Create the gemm handle

                using KokkosKernels::Experimental::GEMM;

                GEMM gemm_handle(&handle);

                // Call the gemm interface

                gemm_handle.gemm(transA, transB,

                                 m, n, k,

                                 alpha,

                                 A_view, lda,

                                 B_view, ldb,

                                 beta,

                                 C_view, ldc);
              }

              // Path: GEMM/CXX/GEMM.cpp

              // gemm KokkosKernels interface

              void

              gemm(

                  const char *transA, const char *transB,

                  const int m, const int n, const int k,

                  const float alpha,

                  const float *A, const int lda,

                  const float *B, const int ldb,

                  const float beta,

                  float *C, const int ldc)

              {

                // Create Kokkos::Views from input data

                using Kokkos::View;

                using Kokkos::LayoutLeft;

                typedef View<float *, LayoutLeft> vector_type;

                typedef View<float **, LayoutLeft> matrix_type;

                vector_type A_view(A, lda * k);

                vector_type B_view(B, ldb * n);

                vector_type C_view(C, ldc * n);

                // Create KokkosKernels::Handle

                using KokkosKernels::Experimental::KokkosKernelsHandle;

                KokkosKernelsHandle<float, int, int> handle;

                // Create the gemm handle

                using KokkosKernels::Experimental::GEMM;

                GEMM gemm_handle(&handle);

                // Call the gemm interface

gemm_handle.gemm(transA, transB,

m, n, k,

alpha,

A_view, lda

=======

// gemm KokkosKernels version

void gemm_kk(

const char transA, const char transB,

const int m, const int n, const int k,

const double alpha,

const double *A, const int lda,

const double *B, const int ldb,

const double beta,

double *C, const int ldc)

{
                  using namespace KokkosKernels;

                  using namespace KokkosKernels::Experimental;

                  using namespace KokkosKernels::Experimental::
                      KokkosKernelsHandle;

                  using namespace KokkosKernels::Experimental::
                      KokkosKernelsHandle::GEMMHandle;

                  using namespace KokkosKernels::Experimental::
                      KokkosKernelsHandle::GEMMHandle::GEMMAlgorithm;

                  using namespace KokkosKernels::Experimental::
                      KokkosKernelsHandle::GEMMHandle::GEMMTranspose;

                  using namespace KokkosKernels::Experimental::
                      KokkosKernelsHandle::GEMMHandle::GEMMConfig;

                  using namespace KokkosKernels::Experimental::
                      KokkosKernelsHandle::GEMMHandle::GEMMConfig::
                          GEMMConfigAlgorithm;

                  using namespace KokkosKernels::Experimental::
                      KokkosKernelsHandle::GEMMHandle::GEMMConfig::
                          GEMMConfigTranspose;

                  using namespace KokkosKernels::Experimental::
                      KokkosKernelsHandle::GEMMHandle::GEMMConfig::GEMMConfigM;

                  using namespace KokkosKernels::Experimental::
                      KokkosKernelsHandle::GEMMHandle::GEMMConfig::GEMMConfigN;

                  using namespace KokkosKernels::Experimental::
                      KokkosKernelsHandle::GEMMHandle::GEMMConfig::GEMMConfigK;

                  using namespace KokkosKernels::Experimental::
                      KokkosKernelsHandle::GEMMHandle::GEMMConfig::
                          GEMMConfigVectorLength;

                  using namespace KokkosKernels::Experimental::
                      KokkosKernelsHandle::GEMMHandle::GEMMConfig::
                          GEMMConfigInnerBlock;

                  using namespace KokkosKernels::Experimental::
                      KokkosKernelsHandle::GEMMHandle::GEMMConfig::
                          GEMMConfigTeamSize;

                  using namespace KokkosKernels::Experimental::
                      KokkosKernelsHandle::GEMMHandle::GEMMConfig::
                          GEMMConfigUnroll;

                  using namespace KokkosKernels::Experimental::K

=======

                    // gemm Kokkos version

                    // Path: GEMM/Kokkos/GEMM.cpp

=======

                    // gemm KokkosKernels::Impl::kk_gemm

                    // Path: GEMM/CXX/GEMM.cpp

=======

                    // gemm Kokkos implementation

                    // Authors: Christoph Lehner

                    // Date: 2017-06-23

#include <Kokkos_Core.hpp>

#include <Kokkos_Random.hpp>

#include <KokkosBlas3_gemm.hpp>

#include <KokkosBlas3_gemm_perf_test.hpp>

#include <KokkosBlas3_trmm.hpp>

#include <KokkosBlas3_trmm_perf_test.hpp>

#include <KokkosKernels_Utils.hpp>

#include <KokkosSparse_spmv.hpp>

#include <KokkosSparse_spmv_perf_test.hpp>

#include <KokkosSparse_spgemm.hpp>

#include <KokkosSparse_spgemm_perf_test.hpp>

#include <impl/Kokkos_Timer.hpp>

#include <KokkosBlas3_trsm.hpp>

#include <KokkosBlas3_trsm_perf_test.hpp>

#include <KokkosSparse_sptrsv.hpp>

#include <KokkosSparse_sptrsv_spec.hpp>

#include <KokkosSparse_sptrsv_perf_test.hpp>

#include <KokkosSparse_sptrsv_supernode.hpp>

#include <KokkosSparse_sptrsv_supernode_spec.hpp>

#include <KokkosSparse_sptrsv_supernode_perf_test.hpp>

#include <KokkosSparse_sptrsv_chol.hpp>

#include <KokkosSparse_sptrsv_chol_spec.hpp>

#include <KokkosSparse_sptrsv_chol_perf_test.hpp>

#include <KokkosSparse_sptrsv_chol_supernode.hpp>

#include <KokkosSparse_sptrsv_chol_supernode_spec.hpp>

#include <KokkosSparse_sptrsv_chol_supernode_perf_test.hpp>

#include <KokkosSparse_sptrsv.hpp>

#include <KokkosSparse_sptrsv_spec.hpp>

#include <KokkosSparse_sptrsv_perf_test.hpp>

#include <KokkosSparse_sptrsv_supernode.hpp>

#include <KokkosSparse_sptrsv_supernode_spec.hpp>

#include <KokkosSparse_sptrsv_supernode_perf_test.hpp>

#include <KokkosSparse_sptrsv_chol.hpp>

#include <KokkosSparse_sptrsv_chol_spec.hpp>

#include <KokkosSparse_sptrsv_chol_perf_test.hpp>

#include <KokkosSparse_sptrsv_chol_supernode.hpp>

#include <Kokkos

                      == == ==
                      =

                          // gemm KokkosKernels version

                      // Path: GEMM/CXX/KokkosKernels/GEMM.cpp

=======

                      // gemm Kokkos Kernel

                      // Inputs: A, B, C

                      // Outputs: C

                      // A, B, C are all 2D arrays of doubles

                      // A is MxK

                      // B is KxN

                      // C is MxN

                      // A, B, C are all stored in column major

                      // C = alpha * A * B + beta * C

                      void gemm(
                          const char transA, const char transB, const int M,
                          const int N, const int K, const double alpha,
                          const double beta,
                          const Kokkos::View<double **, Kokkos::LayoutRight,
                                             Kokkos::HostSpace>
                              A,
                          const Kokkos::View<double **, Kokkos::LayoutRight,
                                             Kokkos::HostSpace>
                              B,
                          Kokkos::View<double **, Kokkos::LayoutRight,
                                       Kokkos::HostSpace>
                              C)

                  {

                    using Kokkos::parallel_for;

                    using Kokkos::TeamPolicy;

                    using Kokkos::ThreadVectorRange;

                    using Kokkos::ALL;

                    using Kokkos::subview;

                    using Kokkos::TeamThreadRange;

                    using Kokkos::Impl::min;

=======

                    // gemm KokkosKernels version

                    // gemm CUBL

                    // Prompt: gemm CUDA
                    Synthesizing 10 /
                        10 solutions

=======

                            // gemm CUDA kernel definition
                            __global__ void
                            gemm(float *A, float *B, float *C, int N) {
                      int row = blockIdx.y * blockDim.y + threadIdx.y;
                      int col = blockIdx.x * blockDim.x + threadIdx.x;
                      if (row < N && col < N) {
                        float sum = 0.0f;
                        for (int i = 0; i < N; ++i) {
                          sum += A[row * N + i] * B[i * N + col];
                        }
                        C[row * N + col] = sum;
                      }
                    }

                    // Host code
                    int main(int argc, char **argv) {
                      // Pointer for host memory
                      float *h_A;
                      float *h_B;
                      float *h_C;

                      // Pointer for device memory
                      float *d_A;
                      float *d_B;
                      float *d_C;

                      // Matrix size
                      int N = 1024;

                      // Size of memory required for each matrix
                      size_t size = N * N * sizeof(float);

                      // Allocate host memory
                      h_A = (float *)malloc(size);
                      h_B = (float *)malloc(size);
                      h_C = (float *)malloc(size);

                      // Allocate device memory
                      cudaMalloc((void **)&d_A, size);
                      cudaMalloc((void **)&d_B, size);
                      cudaMalloc((void **)&d_C, size);

                      // Initialize host memory
                      for (int i = 0; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                          h_A[i * N + j] = 1.0f;
                          h_B[i * N + j] = 2.0f;
                        }
                      }

                      // Copy host memory to device
                      cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
                      cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

                      // Invoke kernel
                      dim3 dimBlock(32, 32);
                      dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);
                      gemm<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

                      // Copy result from device to host
                      cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

                      // Cleanup
                      free(h_A);
                      free(h_B);

=======

                      // gemm CUDA kernel
                      __global__ void gemmKernel(float *A, float *B, float *C,
                                                 int n) {
                        // 2D Thread ID (assuming that only *one* block will be
                        // executed)
                        int tx = threadIdx.x;
                        int ty = threadIdx.y;

                        // Pvalue stores the Pd element that is computed by the
                        // thread
                        float Pvalue = 0;

                        // Each thread loads one row of M and one column of N,
                        //   to produce one element of P.
                        for (int k = 0; k < n; ++k) {
                          float Aelement = A[ty * n + k];
                          float Belement = B[k * n + tx];
                          Pvalue += Aelement * Belement;
                        }

                        // Write the matrix to device memory;
                        // each thread writes one element
                        C[ty * n + tx] = Pvalue;
                      }

                      // Host code
                      int main() {
                        // Size of matrices
                        int n = 1 << 10;

                        // Allocate memory for matrices A and B
                        size_t size = n * n * sizeof(float);
                        float *h_A = (float *)malloc(size);
                        float *h_B = (float *)malloc(size);
                        float *h_C = (float *)malloc(size);

                        // Initialize matrices
                        for (int i = 0; i < n; i++) {
                          for (int j = 0; j < n; j++) {
                            h_A[i * n + j] = rand() / (float)RAND_MAX;
                            h_B[i * n + j] = rand() / (float)RAND_MAX;
                          }
                        }

                        // Allocate memory for device copies of A, B, C
                        float *d_A, *d_B, *d_C;
                        cudaMalloc((void **)&d_A, size);
                        cudaMalloc((void **)&d_B, size);
                        cudaMalloc((void **)&d_C, size);

                        // Copy inputs to device
                        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
                        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

                        // Launch gemmKernel() kernel on GPU
                        int threadsPerBlock = 32;
  dim3 blocksPerGrid((n + threadsPerBlock - 1) / threadsPerBlock, (n + threadsPerBlock - 1) / threadsPer

=======

// gemm CUDA kernel // each thread computes one element of Csub // by accumulating results into Cvalue

__global__ void gemmKernel(float *A, float *B, float *C, int N, int M, int K) {
                          int row = blockIdx.y * blockDim.y + threadIdx.y;

                          int col = blockIdx.x * blockDim.x + threadIdx.x;

                          float Cvalue = 0;

                          for (int e = 0; e < K; ++e)

                            Cvalue += A[row * K + e] * B[e * N + col];

                          C[row * N + col] = Cvalue;

}

// Path: GEMM/CXX/GEMM.cpp

// gemm kernel launcher

void gemmCUDA(float *A, float *B, float *C, int N, int M, int K) {
                          // Load A and B to device memory

                          float *d_A, *d_B, *d_C;

                          cudaMalloc(&d_A, M * K * sizeof(float));

                          cudaMalloc(&d_B, K * N * sizeof(float));

                          cudaMalloc(&d_C, M * N * sizeof(float));

                          cudaMemcpy(d_A, A, M * K * sizeof(float),
                                     cudaMemcpyHostToDevice);

                          cudaMemcpy(d_B, B, K * N * sizeof(float),
                                     cudaMemcpyHostToDevice);

                          // Invoke kernel

                          int BLOCK_SIZE = 16;

                          dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

                          dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                                       (M + dimBlock.y - 1) / dimBlock.y);

                          gemmKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N, M,
                                                            K);

                          cudaDeviceSynchronize();

                          // Read C from device memory

                          cudaMemcpy(C, d_C, M * N * sizeof(float),
                                     cudaMemcpyDeviceToHost);

                          // Free device memory

                          cudaFree(d_A);

                          cudaFree(d_B);

                          cudaFree(d_C);

}

// Path: GEMM/CXX/GEMM.cpp

// gemm CPU implementation

void gemmCPU(float *A, float *B, float *C, int N, int M, int K) {
                          for (int i = 0; i < M; i++) {

                            for (int j = 0; j < N; j++) {

                              float sum = 0;

                              for (int e = 0; e < K; e++) {

                                sum +=

=======

                                    // gemm CUDA kernel

                                    __global__ void gemm_kernel(
                                        const float *A, const float *B,
                                        float *C, int N) {

                                  // Calculate the row index of the C element
                                  // and M

                                  int row =
                                      blockIdx.y * blockDim.y + threadIdx.y;

                                  int M = gridDim.y * blockDim.y;

=======

                                  // gemm CUDA kernel // each thread computes
                                  // one element of C sub( C ) = alpha * sub( A
                                  // ) * sub( B ) + beta * sub( C ) // where
                                  // sub( C ) is C row from row_begin to row_end
                                  // (non-inclusive), // sub( A ) is A row from
                                  // row_begin to row_end (non-inclusive), //
                                  // sub( B ) is B col from col_begin to col_end
                                  // (non-inclusive) template < typename T >
                                  // __global__ void gemm_kernel ( int row_begin
                                  // , int row_end , int col_begin , int col_end
                                  // , int inner_dim , T alpha , const T * A ,
                                  // const T * B , T beta , T * C ) { //@@
                                  // Insert code to implement the GEMM kernel
                                  // here int row = blockIdx . y * blockDim . y
                                  // + threadIdx . y ; int col = blockIdx . x *
                                  // blockDim . x + threadIdx . x ; if ( row <
                                  // row_end - row_begin && col < col_end -
                                  // col_begin ) { T value = 0 ; for ( int k = 0
                                  // ; k < inner_dim ; ++ k ) { value += A [ row
                                  // * inner_dim + k ] * B [ k * ( col_end -
                                  // col_begin ) + col ] ; } C [ row * ( col_end
                                  // - col_begin ) + col ] = alpha * value +
                                  // beta * C [ row * ( col_end - col_begin ) +
                                  // col ] ; } }

                                  template <typename T>
                                  void gemm(
                                      cublasHandle_t handle,
                                      cublasOperation_t transA,
                                      cublasOperation_t transB, int m, int n,
                                      int k, T alpha, const T *A, int lda,
                                      const T *B, int ldb, T beta, T *C,
                                      int ldc) { //@@ Insert code to call the
                                                 // GEMM kernel here // You need
                                                 // to choose grid and block
                                                 // dimensions int block_size =
                                                 // 32 ; dim3 block_dim (
                                                 // block_size , block_size , 1
                                                 // ) ; dim3 grid_dim ( ( n +
                                                 // block_size - 1 ) /
                                                 // block_size , ( m +
                                                 // block_size - 1 ) /
                                                 // block_size
                                                 // , 1 ) ; //@@ Launch the GPU
                                                 // Kernel here gemm_kernel <<<
                                                 // grid_dim , block_dim >>> ( 0
                                                 // , m , 0 , n , k , alpha , A
                                                 //, B , beta , C ) ; }

                                    // Path: GEMM/CXX/GEMM.cpp

                                    // gemm CUDA kernel

=======

                                    // gemm CUDA kernel for square matrices of
                                    // dimension N x N

                                    __global__ void gemm(const float *A,
                                                         const float *B,
                                                         float *C, int N) {

                                      int row =
                                          blockIdx.y * blockDim.y + threadIdx.y;

                                      int col =
                                          blockIdx.x * blockDim.x + threadIdx.x;

                                      if (row < N && col < N) {

                                        float Cvalue = 0;

                                        for (int k = 0; k < N; k++) {

                                          float Aelement = A[row * N + k];

                                          float Belement = B[k * N + col];

                                          Cvalue += Aelement * Belement;
                                        }

                                        C[row * N + col] = Cvalue;
                                      }
                                    }

                                    // Path: GEMM/CXX/GEMM.cpp

=======

                                    // gemm CUDA kernel for 32x32 tiles of A and
                                    // B matrices

                                    __global__ void gemmKernel(
                                        int M, int N, int K, const float *A,
                                        const float *B, float *C, int lda,
                                        int ldb, int ldc) {

                                      // Block row and column

                                      int blockRow = blockIdx.y;

                                      int blockCol = blockIdx.x;

                                      // Each thread block computes one
                                      // sub-matrix Csub of C

                                      float Csub = 0;

                                      // Thread row and column within Csub

                                      int row = threadIdx.y;

                                      int col = threadIdx.x;

                                      // Loop over all the sub-matrices of A and
                                      // B that are

                                      // required to compute Csub

                                      // Multiply each pair of sub-matrices
                                      // together

                                      // and accumulate the results

                                      for (int m = 0;
                                           m < (K + TILE_SIZE - 1) / TILE_SIZE;
                                           ++m) {

                                        // Get sub-matrix Asub of A

                                        float Asub =
                                            A[blockRow * TILE_SIZE * lda +
                                              m * TILE_SIZE + row * lda + col];

                                        // Get sub-matrix Bsub of B

                                        float Bsub = B[m * TILE_SIZE * ldb +
                                                       blockCol * TILE_SIZE +
                                                       row * ldb + col];

                                        // Accumulate the results

                                        Csub += Asub * Bsub;
                                      }

                                      // Store the final result in C

                                      C[blockRow * TILE_SIZE * ldc +
                                        blockCol * TILE_SIZE + row * ldc +
                                        col] = Csub;
                                    }

                                    // Path: GEMM/CXX/GEMM.cpp

                                    // gemm CUDA kernel for 32x32 tiles of A and
                                    // B matrices

                                    __global__ void gemmKernel(
                                        int M, int N, int K, const float *A,
                                        const float *B, float *C, int lda,
                                        int ldb, int ldc) {

                                      // Block row and column

                                      int blockRow = blockIdx.y;

                                      int blockCol = blockIdx.x;

                                      // Each thread block computes one
                                      // sub-matrix Csub of C

                                      float Csub = 0;

                                      // Thread row and column within Csub

                                      int row = threadIdx.y;

                                      int col = threadIdx.x;

                                      // Loop over all the sub-matrices of A and
                                      // B that are

                                      // required to compute Csub

                                      // Multiply each pair of sub-matrices
                                      // together

                                      // and accumulate the results

                                      for (int m = 0;
                                           m < (K + TILE_SIZE - 1) / TILE_SIZE;
                                           ++m)

=======

                                        // gemm CUDA kernel // Multiply the A
                                        // and B matrices together and store the
                                        // result in C. // This kernel uses
                                        // shared memory to improve performance.
                                        // __global__ void gemm_kernel( int m,
                                        // int n, int k, const float *
                                        // __restrict__ A, const float *
                                        // __restrict__ B, float * __restrict__
                                        // C ) { // Block index int bx =
                                        // blockIdx.x; int by = blockIdx.y; //
                                        // Thread index int tx = threadIdx.x;
                                        // int ty = threadIdx.y; // Index of the
                                        // first sub-matrix of A processed by
                                        // the block int aBegin = m * BLOCK_SIZE
                                        // * by; int aEnd = aBegin + m - 1; int
                                        // aStep = BLOCK_SIZE; // Index of the
                                        // first sub-matrix of B processed by
                                        // the block int bBegin = BLOCK_SIZE *
                                        // bx; int bStep = BLOCK_SIZE * n; //
                                        // The element of the block sub-matrix
                                        // that is computed // by the thread int
                                        // a = aBegin + m * ty; int b = bBegin +
                                        // n * tx; // Declaration of the shared
                                        // memory array As used to // store the
                                        // sub-matrix of A int as = m *
                                        // BLOCK_SIZE * ty + tx; // Declaration
                                        // of the shared memory array Bs used to
                                        // // store the sub-matrix of B int bs =
                                        // BLOCK_SIZE * ty + tx; // Csub is used
                                        // to store the element of the block //
                                        // sub-matrix that is computed by the
                                        // thread float Csub = 0; // Loop over
                                        // all the sub-matrices of A and B //
                                        // required to compute the block
                                        // sub-matrix // Multiply each pair of
                                        // sub-matrices together // and
                                        // accumulate the results __shared__
                                        // float As[BLOCK_SIZE * BLOCK_SIZE];
                                        // __shared__ float Bs[BLOCK_SIZE *
                                        // BLOCK_SIZE]; for (int i = aBegin, j =
                                        // bBegin; i <= aEnd; i += aStep, j +=
                                        // bStep) { // Load the matrices from
                                        // device memory // to shared memory;
                                        // each thread loads // one element of
                                        // each matrix As[as] = A[i + as];
                                        // Bs[bs] = B[j + bs]; // Synchronize to
                                        // make sure the matrices are // loaded
                                        // __syncthreads(); // Multiply the two
                                        // matrices together; // each thread
                                        // computes one element // of the block
                                        // sub-matrix for (int k = 0; k <
                                        // BLOCK_SIZE; ++k) { Csub += As[m * ty
                                        // + k] * Bs[BLOCK_SIZE * k + tx];

=======

                                        // gemm CUDA kernel for 2x2 tiles

                                        // Each thread computes 4 elements of C

                                        __global__ void gemm_kernel(
                                            int m, int n, int k, float *A,
                                            float *B, float *C)

                                        {

                                          int row = blockIdx.y * blockDim.y +
                                                    threadIdx.y;

                                          int col = blockIdx.x * blockDim.x +
                                                    threadIdx.x;

                                          if (row < m && col < n)

                                          {

                                            float Cvalue = 0.0f;

                                            for (int e = 0; e < k; ++e)

                                            {

                                              Cvalue += A[row * k + e] *
                                                        B[e * n + col];
                                            }

                                            C[row * n + col] = Cvalue;
                                          }
                                        }

=======

                                      // gemm CUDA kernel
                                      // /////////////////////////////////////////////////////////////////////////////////

                                      // Prompt: gemm HIP
                                      Synthesizing 7 /
                                          10 solutions(Duplicates hidden)

=======

                                          // gemm HIP implementation

                                          void
                                          gemm(hipblasOperation_t transA,

                                               hipblasOperation_t transB,

                                               int m,

                                               int n,

                                               int k,

                                               float alpha,

                                               const float *A,

                                               int lda,

                                               const float *B,

                                               int ldb,

                                               float beta,

                                               float *C,

                                               int ldc)

                                      {

                                        // Create a handle for HIPBLAS

                                        hipblasHandle_t handle;

                                        hipblasCreate(&handle);

                                        // Do the actual multiplication

                                        hipblasSgemm(handle, transA, transB, m,
                                                     n, k, &alpha, A, lda, B,
                                                     ldb, &beta, C, ldc);

                                        // Destroy the handle

                                        hipblasDestroy(handle);
                                      }

                                      // Path: GEMM/CXX/main.cpp

=======

                                      // gemm HIP
                                      // Path: GEMM/HIP/GEMM.cpp

                                      // gemm OpenCL
                                      // Path: GEMM/OPENCL/GEMM.cpp

                                      // gemm OpenMP
                                      // Path: GEMM/OPENMP/GEMM.cpp

=======

                                      // gemm HIP

                                      // Path: GEMM/HIP/GEMM.cpp

=======

                                      // gemm HIP BLAS

                                      void gemm_hipblas(
                                          const int M, const int N, const int K,
                                          const float alpha, const float *A,
                                          const float *B, const float beta,
                                          float *C) {

                                        // Create HIP device objects

                                        hipblasHandle_t handle;

                                        hipblasCreate(&handle);

                                        // Call HIP BLAS function

                                        hipblasSgemm(handle, HIPBLAS_OP_N,
                                                     HIPBLAS_OP_N, M, N, K,
                                                     &alpha, A, M, B, K, &beta,
                                                     C, M);

                                        // Destroy HIP device objects

                                        hipblasDestroy(handle);
                                      }

=======

                                      // gemm HIP code

                                      // Path: GEMM/HIP/GEMM.cpp

=======

                                      // gemm HIP kernel

                                      // Path: GEMM/HIP/GEMM.hip

                                      // Prompt: gemm Thrust
                                      Synthesizing 10 /
                                          10 solutions(Duplicates hidden)

=======

                                          // gemm Thrust

                                          // Path: GEMM/CXX/GEMM_Thrust.cu

=======

                                          // gemm Thrust version

                                          void
                                          gemmThrust(float *A, float *B,
                                                     float *C, int m, int n,
                                                     int k) {

                                        // TODO
                                      }

                                      // Path: GEMM/CUDA/GEMM.cu

=======

                                      // gemm Thrust

                                      // Path: GEMM/CXX/GEMMThrust.cu

=======

                                      // gemm Thrust implementation
                                      // Path: GEMM/CXX/Thrust/GEMM.cpp

=======

                                      // gemm Thrust implementation

                                      void gemm_thrust(double *A, double *B,
                                                       double *C, int m, int n,
                                                       int k, double alpha,
                                                       double beta) {

                                        // TODO
                                      }

                                      // Path: GEMM/CXX/GEMM.cpp

                                      // gemm Thrust implementation

                                      void gemm_thrust(double *A, double *B,
                                                       double *C, int m, int n,
                                                       int k, double alpha,
                                                       double beta) {

                                        // TODO
                                      }

                                      I think the thrust implementation
                                          is not the problem
                                              .I think the problem is in the
                                                  initialization of the matrix
                                              .I tried to do it in the same way
                                                  as in the CUDA implementation,
                                          and it works.I also tried to do it in
                                                  the same way as in the Thrust
                                                      implementation,
                                          and it does not work
                                                  .

                                              I would appreciate any help
                                                  .

                                              Thank you

                                                  I think the thrust
                                                      implementation
                                                          is not the problem
                                                  .I think the problem is in the
                                                      initialization of the
                                                          matrix
                                                  .I tried to do it in the same
                                                      way as in the CUDA
                                                          implementation,
                                          and it works.I also tried to do it in
                                                  the same way as in the Thrust
                                                      implementation,
                                          and it does not work
                                                  .

                                              I would appreciate any help
                                                  .

                                              Thank you

                                                  I think the thrust
                                                      implementation
                                                          is not the problem
                                                  .I think the problem is in the
                                                      initialization of the
                                                          matrix
                                                  .I tried to do it in the same
                                                      way as in the CUDA
                                                          implementation,
                                          and it works.I also tried to do it in
                                                  the same way as in the Thrust
                                                      implementation,
                                          and it does not work
                                                  .

                                              I would appreciate any help
                                                  .

                                              Thank you

                                                  I think the thrust
                                                      implementation
                                                          is not the problem
                                                  .I think the problem is in the
                                                      initialization of the
                                                          matrix
                                                  .I tried to do it in the same
                                                      way as in the CUDA
                                                          implementation,
                                          and it works.I also tried to do it in
                                                  the same way as in the Thrust
                                                      implementation,
                                          and it does not work
                                                  .

                                              I would appreciate any help
                                                  .

                                              Thank you

                                                  I think the thrust
                                                      implementation
                                                          is not the problem
                                                  .I think the problem is in the
                                                      initialization of the
                                                          matrix
                                                  .I tried to do it in the same
                                                      way as in the CUDA
                                                          implementation,
                                          and it works.I also tried to do it in
                                                  the same way as in the Thrust
                                                      implementation,
                                          and it does not work
                                                  .

                                              I would appreciate any help
                                                  .

                                              Thank you

                                                  I think the thrust
                                                      implementation
                                                          is not the problem
                                                  .I think the problem is in the
                                                      initialization of the
                                                          matrix
                                                  .I tried to do it in the same
                                                      way as in the CUDA
                                                          implementation

=======

                                              // gemm Thrust version

                                              // Path: GEMM/Thrust/GEMM.cpp

=======

                                              // gemm Thrust implementation

                                              void
                                              gemm_thrust(const Matrix &A,
                                                          const Matrix &B,
                                                          Matrix &C) {

                                        // Prompt: gemm Thrust
                                        Synthesizing 7 /
                                                10 solutions(Duplicates hidden)

=======

                                          // gemm Thrust

                                          // Path: GEMM/CXX/Thrust/GEMM.cpp

=======

                                          // gemm Thrust

#include <algorithm>

#include <iostream>

#include <random>

#include <vector>

#include <thrust/device_vector.h>

#include <thrust/functional.h>

#include <thrust/inner_product.h>

#include <thrust/iterator/zip_iterator.h>

#include <thrust/transform.h>

#include <thrust/tuple.h>

#include <thrust/for_each.h>

#include <thrust/iterator/constant_iterator.h>

#include <thrust/iterator/counting_iterator.h>

#include <thrust/iterator/transform_iterator.h>

#include <thrust/iterator/zip_iterator.h>

#include <thrust/reduce.h>

#include <thrust/sequence.h>

#include <thrust/system/cuda/execution_policy.h>

#include <thrust/transform.h>

#include <thrust/transform_reduce.h>

#include <thrust/functional.h>

#include <thrust/iterator/constant_iterator.h>

#include <thrust/iterator/zip_iterator.h>

#include <thrust/random.h>

#include <thrust/iterator/permutation_iterator.h>

#include "thrust/detail/config.h"

#include "thrust/detail/type_traits.h"

#include "thrust/detail/static_assert.h"

#include "thrust/detail/type_traits.h"

#include "thrust/detail/type_traits/remove_reference.h"

#include "thrust/detail/type_traits/remove_cv.h"

#include "thrust/detail/type_traits/remove_const.h"

#include "thrust/detail/type_traits/remove_volatile.h"

#include "thrust/detail/type_traits/remove_extent.h"

#include "thrust/detail/type_traits/remove_all_extents.h"

#include "thrust/detail/type_traits/enable_if.h"

#include "thrust/detail/type_traits/aligned_storage.h"

#include "thrust/detail/type_traits/is_same.h"

#include "thrust/detail/type_traits/is_convertible.h"

#include "thrust/detail/type_traits/is_base_of.h"

#include "thrust/detail/type_traits/is_arithmetic.h"

#include "thrust/detail/type_traits/is_integral.h"

#include "thrust/detail/type_traits/is_floating_point.h"

#include "thrust/detail/type_traits/is_unsigned.h"

#include "thrust/detail/type_traits/is_signed.h"

#include "thrust/detail/type_traits/is_const.h"

#include "thrust/detail/type_traits/is_volatile.h"

#include "thrust/detail/type_traits/is_pod.h"

#include "thrust/detail/type_traits/is_pointer.h"

#include "thrust/detail/type_traits/is_reference.h"

#include "thrust/detail/type_traits

                                            == == ==
                                            =

                                                // gemm Thrust

                                            // Path: GEMM/CXX/GEMM.cpp

=======

                                            // gemm Thrust implementation
                                            // -------------------------
                                            template <typename T>
                                            void gemmThrust(
                                                const T alpha, const T beta,
                                                const int M, const int N,
                                                const int K, const T *A,
                                                const T *B, T *C) {

=======

                                          // gemm Thrust

                                          void gemm_thrust(float *A, float *B,
                                                           float *C, int n) {

                                            // Prompt: gemm Thrust function
                                            Synthesizing 10 /
                                                10 solutions(Duplicates hidden)

=======

                                                // gemm Thrust function

                                                template <typename T>

                                                void gemmThrust(
                                                    int m, int n, int k,
                                                    T alpha, T *A, T *B, T beta,
                                                    T *C, int lda, int ldb,
                                                    int ldc) {

=======

                                              // gemm Thrust function
                                              template <typename T>
                                              void gemm_thrust(
                                                  int m, int n, int k, T alpha,
                                                  T *A, int lda, T *B, int ldb,
                                                  T beta, T *C, int ldc) {
                                                thrust::device_vector<T> d_A(m *
                                                                             k);
                                                thrust::device_vector<T> d_B(k *
                                                                             n);
                                                thrust::device_vector<T> d_C(m *
                                                                             n);

                                                // copy data from host to device
                                                thrust::copy(A, A + m * k,
                                                             d_A.begin());
                                                thrust::copy(B, B + k * n,
                                                             d_B.begin());
                                                thrust::copy(C, C + m * n,
                                                             d_C.begin());

                                                // do the actual multiplication
                                                thrust::device_vector<T> d_C2 =
                                                    thrust::inner_product(
                                                        d_A.begin(), d_A.end(),
                                                        d_B.begin(),
                                                        d_C.begin(),
                                                        thrust::plus<T>(),
                                                        thrust::multiplies<
                                                            T>());

                                                // copy data back to host
                                                thrust::copy(d_C2.begin(),
                                                             d_C2.end(), C);
                                              }

                                              // gemm Thrust function
                                              template <typename T>
                                              void gemm_thrust(
                                                  int m, int n, int k, T alpha,
                                                  T *A, int lda, T *B, int ldb,
                                                  T beta, T *C, int ldc) {
                                                thrust::device_vector<T> d_A(m *
                                                                             k);
                                                thrust::device_vector<T> d_B(k *
                                                                             n);
                                                thrust::device_vector<T> d_C(m *
                                                                             n);

                                                // copy data from host to device
                                                thrust::copy(A, A + m * k,
                                                             d_A.begin());
                                                thrust::copy(B, B + k * n,
                                                             d_B.begin());
                                                thrust::copy(C, C + m * n,
                                                             d_C.begin());

                                                // do the actual multiplication
                                                thrust::device_vector<T> d_C2 =
                                                    thrust::inner_product(
                                                        d_A.begin(), d_A.end(),
                                                        d_B.begin(),
                                                        d_C.begin(),
                                                        thrust::plus<T>(),
                                                        thrust::multiplies<
                                                            T>());

                                                // copy data back to host
                                                thrust::copy(d_C2.begin(),
                                                             d_C2.end(), C);
                                              }

                                              // gemm Thrust function
template <typename T>
void gemm_thrust( int m, int n, int k, T alpha, T *A, int lda, T *B,
                  int

=======

// gemm Thrust function

template <typename T, typename U>

void gemmThrust(const T *A, const T *B, U *C, const int M, const int N, const int K, const T alpha, const T beta) {

=======
                                                // gemm Thrust function

                                                // Path: GEMM/CXX/GEMM.cpp

=======

                                                // gemm Thrust function

                                                // C = alpha * A * B + beta * C

                                                // A: M x K matrix

                                                // B: K x N matrix

                                                // C: M x N matrix

                                                // alpha, beta: scalar

template

void gemm(const T alpha, const T beta, const thrust::device_vector

const thrust::device_vector

const th

=======

// gemm Thrust function

// Arguments:

// A: Matrix A

// B: Matrix B

// C: Matrix C

// alpha: Scalar alpha

// beta: Scalar beta

// numRowsA: Number of rows in matrix A

// numColsA: Number of columns in matrix A

// numColsB: Number of columns in matrix B

void gemmThrust(thrust::device_vector<float> &A,

thrust::device_vector<float> &B,

thrust::device_vector<float> &C,

float alpha,

float beta,

int numRowsA,

int numColsA,

int numColsB)

{

=======
                                                  // gemm Thrust function
                                                  template <typename T>
                                                  void gemm(
                                                      const size_t N,
                                                      const T alpha, const T *A,
                                                      const T *B, const T beta,
                                                      T *C) {
                                                    // Create thrust device
                                                    // pointers
                                                    thrust::device_ptr<T> d_A(
                                                        A);
                                                    thrust::device_ptr<T> d_B(
                                                        B);
                                                    thrust::device_ptr<T> d_C(
                                                        C);

                                                    // Define the gemm operation
                                                    thrust::transform(
                                                        thrust::
                                                            make_zip_iterator(
                                                                thrust::
                                                                    make_tuple(
                                                                        d_A,
                                                                        d_B)),
                                                        thrust::make_zip_iterator(
                                                            thrust::make_tuple(
                                                                d_A, d_B)) +
                                                            N * N,
                                                        d_C,
                                                        thrust::make_transform_output_iterator(
                                                            thrust::
                                                                make_discard_iterator(),
                                                            detail::
                                                                gemm_functor<T>(
                                                                    alpha,
                                                                    beta)));
                                                  }

                                                  // gemm Thrust function
                                                  template <typename T>
                                                  void gemm(
                                                      const size_t N,
                                                      const T alpha, const T *A,
                                                      const T *B, const T beta,
                                                      T *C) {
                                                    // Create thrust device
                                                    // pointers
                                                    thrust::device_ptr<T> d_A(
                                                        A);
                                                    thrust::device_ptr<T> d_B(
                                                        B);
                                                    thrust::device_ptr<T> d_C(
                                                        C);

                                                    // Define the gemm operation
                                                    thrust::transform(
                                                        thrust::
                                                            make_zip_iterator(
                                                                thrust::
                                                                    make_tuple(
                                                                        d_A,
                                                                        d_B)),
                                                        thrust::make_zip_iterator(
                                                            thrust::make_tuple(
                                                                d_A, d_B)) +
                                                            N * N,
                                                        d_C,
                                                        thrust::make_transform_output_iterator(
                                                            thrust::
                                                                make_discard_iterator(),
                                                            detail::
                                                                gemm_functor<T>(
                                                                    alpha,
                                                                    beta)));
                                                  }

                                                  // gemm Thrust function
                                                  template <typename T>
                                                  void gemm(
                                                      const size_t N,
                                                      const T alpha, const T *A,
                                                      const T *B, const T beta,
                                                      T *C) {
                                                    // Create thrust device
                                                    // pointers
                                                    thrust::device_ptr<T> d_A(
                                                        A);
                                                    thrust::device_ptr<T> d_B(
                                                        B);
                                                    thrust::device_ptr<T> d_C(
                                                        C);

                                                    // Define the gemm operation
  thrust::transform(
    thrust::make_zip_iterator(thrust::make_tuple(d_A, d_B)),
    thrust::make_zip_iterator(thrust::make_tuple(d_A, d_B)) + N*N,
    d_C,
    thrust::make_transform_output_iterator(
      thrust::make_discard_iterator(),
      detail::gemm_functor

=======

// gemm Thrust function

template <typename T>

void gemm_thrust( const T *A, const T *B, T *C, int m, int n, int k )

{
                                                      // thrust::device_vector<T>
                                                      // d_A(A, A + m*k);

                                                      // thrust::device_vector<T>
                                                      // d_B(B, B + k*n);

                                                      // thrust::device_vector<T>
                                                      // d_C(m*n);

                                                      thrust::device_vector<T>
                                                          d_A(A, A + m * k);

                                                      thrust::device_vector<T>
                                                          d_B(B, B + k * n);

                                                      thrust::device_vector<T>
                                                          d_C(m * n);

=======

                                                      // gemm Thrust function
                                                      template <typename T>
                                                      void gemmThrust(
                                                          T * A, T * B, T * C,
                                                          int M, int N, int K,
                                                          int numIters) {
                                                        // Create Thrust device
                                                        // views
                                                        thrust::device_vector<T>
                                                            d_A(A, A + M * K);
                                                        thrust::device_vector<T>
                                                            d_B(B, B + K * N);
                                                        thrust::device_vector<T>
                                                            d_C(C, C + M * N);

                                                        // Create Thrust device
                                                        // views
                                                        thrust::device_ptr<T>
                                                            d_A_ptr = thrust::
                                                                device_pointer_cast(
                                                                    d_A.data());
                                                        thrust::device_ptr<T>
                                                            d_B_ptr = thrust::
                                                                device_pointer_cast(
                                                                    d_B.data());
                                                        thrust::device_ptr<T>
                                                            d_C_ptr = thrust::
                                                                device_pointer_cast(
                                                                    d_C.data());

                                                        // Create Thrust device
                                                        // views
                                                        thrust::
                                                            device_reference<T>
                                                                d_A_ref =
                                                                    d_A[0];
                                                        thrust::
                                                            device_reference<T>
                                                                d_B_ref =
                                                                    d_B[0];
                                                        thrust::
                                                            device_reference<T>
                                                                d_C_ref =
                                                                    d_C[0];

                                                        // Start timer
                                                        cudaEvent_t start, stop;
                                                        cudaEventCreate(&start);
                                                        cudaEventCreate(&stop);
                                                        cudaEventRecord(start);

                                                        // Run the gemm kernel
                                                        for (int i = 0;
                                                             i < numIters; i++)
                                                          thrust::for_each(
                                                              thrust::make_zip_iterator(
                                                                  thrust::make_tuple(
                                                                      d_A_ptr,
                                                                      d_B_ptr,
                                                                      d_C_ptr)),
                                                              thrust::make_zip_iterator(
                                                                  thrust::make_tuple(
                                                                      d_A_ptr +
                                                                          M * K,
                                                                      d_B_ptr +
                                                                          K * N,
                                                                      d_C_ptr +
                                                                          M * N)),
                                                              gemm_functor<T>(
                                                                  d_A_ref,
                                                                  d_B_ref,
                                                                  d_C_ref, M, N,
                                                                  K));

                                                        // Stop timer
                                                        cudaEventRecord(stop);
                                                        cudaEventSynchronize(
                                                            stop);
                                                        float milliseconds = 0;
                                                        cudaEventElapsedTime(
                                                            &milliseconds,
                                                            start, stop);
                                                        cudaEventDestroy(start);
                                                        cudaEventDestroy(stop);

                                                        // Copy data back to
                                                        // host
                                                        thrust::copy(
                                                            d_C.begin(),
                                                            d_C.end(), C);
                                                      }

                                                      // gemm Thrust function
                                                      template <typename T>
                                                      void gemmThrust(
                                                          T * A, T * B, T * C,
                                                          int M, int N, int K,
                                                          int numIters) {
                                                        // Create Thrust device
                                                        // views
                                                        thrust::device_vector <
                                                            T

                                                                    // Prompt:
                                                                    // gemm
                                                                    // Kokkos
                                                                    // function
                                                                    Synthesizing 4 /
                                                                10 solutions

=======

                                                                // gemm Kokkos
                                                                // function

                                                                // \param A
                                                                // input matrix

                                                                // \param B
                                                                // input matrix

                                                                // \param C
                                                                // output matrix

                                                                // \param alpha
                                                                // scalar

                                                                // \param beta
                                                                // scalar

                                                                // \param m
                                                                // number of
                                                                // rows of A and
                                                                // C

                                                                // \param n
                                                                // number of
                                                                // columns of B
                                                                // and C

                                                                // \param k
                                                                // number of
                                                                // columns of A
                                                                // and rows of B

                                                                // \param
                                                                // num_threads
                                                                // number of
                                                                // threads to
                                                                // use

                                                                void
                                                                gemm(
                                                                    Kokkos::View<
                                                                        double *
                                                                            *> &
                                                                        A,
                                                                    Kokkos::View<
                                                                        double *
                                                                            *> &
                                                                        B,
                                                                    Kokkos::View<
                                                                        double *
                                                                            *> &
                                                                        C,
                                                                    double
                                                                        alpha,
                                                                    double beta,
                                                                    int m,
                                                                    int n,
                                                                    int k,
                                                                    int num_threads) {

                                                          // Create a team
                                                          // policy with
                                                          // num_threads threads

                                                          Kokkos::TeamPolicy<>
                                                              policy(
                                                                  m,
                                                                  num_threads);

                                                          // Create a team
                                                          // policy with
                                                          // num_threads threads

                                                          Kokkos::parallel_for(
                                                              policy,
                                                              KOKKOS_LAMBDA(
                                                                  const Kokkos::
                                                                      TeamPolicy<>::member_type
                                                                          &team) {
                                                                // Get the row
                                                                // index of the
                                                                // matrix

                                                                int row =
                                                                    team.league_rank();

                                                                // Get the
                                                                // thread index
                                                                // within the
                                                                // team

                                                                int col =
                                                                    team.team_rank();

                                                                // The thread
                                                                // computes one
                                                                // element of
                                                                // the output
                                                                // matrix

                                                                double sum = 0;

                                                                for (int i = 0;
                                                                     i < k;
                                                                     i++) {

                                                                  sum +=
                                                                      A(row,
                                                                        i) *
                                                                      B(i, col);
                                                                }

                                                                C(row, col) =
                                                                    alpha *
                                                                        sum +
                                                                    beta *
                                                                        C(row,
                                                                          col);
                                                              });
                                                        }

                                                        // Path:
                                                        // GEMM/CXX/main.cpp

=======

                                                        // gemm Kokkos function

                                                        template <
                                                            typename ExecSpace,
                                                            typename ScalarType>

                                                        void gemm(
                                                            const int m,
                                                            const int n,
                                                            const int k,

                                                            const ScalarType
                                                                alpha,
                                                            const ScalarType *A,
                                                            const ScalarType *B,

                                                            const ScalarType
                                                                beta,
                                                            ScalarType *C)

                                                        {

                                                          // Create 2D Views of
                                                          // the matrices A, B
                                                          // and C

                                                          using View2D =
                                                              Kokkos::View<
                                                                  ScalarType **,
                                                                  Kokkos::
                                                                      LayoutRight,
                                                                  ExecSpace>;

                                                          View2D A_view(A, m,
                                                                        k);

                                                          View2D B_view(B, k,
                                                                        n);

                                                          View2D C_view(C, m,
                                                                        n);

                                                          // Create a team
                                                          // policy with a team
                                                          // size of 32 threads

                                                          using TeamPolicy =
                                                              Kokkos::TeamPolicy<
                                                                  ExecSpace>;

                                                          using TeamMember =
                                                              typename TeamPolicy::
                                                                  member_type;

                                                          TeamPolicy policy(
                                                              m, Kokkos::AUTO);

                                                          // Create a
                                                          // parallel_for with
                                                          // the team policy

Kokkos::parallel_for( "GEMM", policy,

KOKKOS_LAMBDA( const TeamMember& team )

{
                                                            // Get the row index
                                                            // of the matrix C

                                                            const int i =
                                                                team.league_rank();

                                                            // Get the number of
                                                            // rows in the
                                                            // matrix C

                                                            const int m =
                                                                team.league_size();

                                                            // Get the number of
                                                            // columns in the
                                                            // matrix C

                                                            const int n =
                                                                C_view
                                                                    .extent_int(
                                                                        1);

                                                            // Get the number of
                                                            // columns in the
                                                            // matrix A

                                                            const int k =
                                                                A_view
                                                                    .extent_int(
                                                                        1);

                                                            // Get the number of
                                                            // columns in the
                                                            // matrix B

                                                            const int l =
                                                                B_view
                                                                    .extent_int(
                                                                        1);

                                                            // Get the team size

                                                            const int team_size =
                                                                team.team_size();

                                                            // Get the team rank

                                                            const int team_rank =
                                                                team.team_rank();

                                                            // Get the team
                                                            // shared memory

                                                            const ScalarType *A_shared =
                                                                team.team_shmem()
                                                                    .get_shmem(
                                                                        k);

                                                            const ScalarType *B_shared =
                                                                team.team_shmem()
                                                                    .get_shmem(
                                                                        l);

                                                            // Loop over the
                                                            // columns of the
                                                            // matrix C

                                                            for (int j = 0;
                                                                 j < n; ++j)

                                                            {

                                                              // Initialize the
                                                              // value of C(i,j)
                                                              // to zero

                                                              ScalarType C_ij =
                                                                  0.0;

                                                              // Loop over the
                                                              // columns of the
                                                              // matrix A

                                                              for (int p = 0;
                                                                   p < k; ++p)

                                                              {

                                                                // Loop over the
                                                                // columns of
                                                                // the matrix B

                                                                for (int q = 0;
                                                                     q < l; ++q)

                                                                {

=======

                                                                  // gemm Kokkos
                                                                  // function

                                                                  // Path:
                                                                  // GEMM/CXX/Kokkos/GEMM.cpp

=======

                                                                  // gemm Kokkos
                                                                  // function

                                                                  // This
                                                                  // function is
                                                                  // a simple
                                                                  // wrapper
                                                                  // around the
                                                                  // Kokkos::parallel_for

                                                                  // call. It is
                                                                  // templated
                                                                  // on the
                                                                  // execution
                                                                  // space to
                                                                  // allow for
                                                                  // easy

                                                                  // switching
                                                                  // between
                                                                  // different
                                                                  // execution
                                                                  // spaces.

                                                                  // This
                                                                  // function is
                                                                  // not very
                                                                  // useful in
                                                                  // this case,
                                                                  // since the

                                                                  // parallel_for
                                                                  // is only
                                                                  // called
                                                                  // once, but
                                                                  // it is
                                                                  // included
                                                                  // for

                                                                  // completeness.

                                                                  template <
                                                                      class
                                                                      ExecutionSpace>

                                                                  void gemm(
                                                                      ExecutionSpace &
                                                                          space,
                                                                      int N,
                                                                      int M,
                                                                      int K,
                                                                      const double
                                                                          *A,
                                                                      const double
                                                                          *B,
                                                                      double
                                                                          *C) {

                                                                    // Call the
                                                                    // parallel_for
                                                                    // kernel

                                                                    Kokkos::parallel_for(
                                                                        N * M,
                                                                        KOKKOS_LAMBDA(
                                                                            const int
                                                                                &i) {
                                                                          // Get
                                                                          // the
                                                                          // row
                                                                          // and
                                                                          // column
                                                                          // of
                                                                          // the
                                                                          // matrix
                                                                          // element

                                                                          int row =
                                                                              i /
                                                                              M;

                                                                          int col =
                                                                              i %
                                                                              M;

                                                                          // Set
                                                                          // the
                                                                          // matrix
                                                                          // element
                                                                          // to
                                                                          // zero

                                                                          C[i] =
                                                                              0.0;

                                                                          // Loop
                                                                          // over
                                                                          // the
                                                                          // elements
                                                                          // of
                                                                          // the
                                                                          // row
                                                                          // of
                                                                          // A
                                                                          // and
                                                                          // the
                                                                          // column
                                                                          // of
                                                                          // B

                                                                          // and
                                                                          // add
                                                                          // the
                                                                          // products
                                                                          // to
                                                                          // the
                                                                          // matrix
                                                                          // element

                                                                          for (
                                                                              int j =
                                                                                  0;
                                                                              j <
                                                                              K;
                                                                              j++) {

                                                                            C[i] +=
                                                                                A[row *
                                                                                      K +
                                                                                  j] *
                                                                                B[col +
                                                                                  j * M];
                                                                          }
                                                                        });
                                                                  }

                                                                  // Prompt:
                                                                  // gemm HIP
                                                                  // function
                                                                  Synthesizing 6 /
                                                                      10 solutions

=======

                                                                      // gemm
                                                                      // HIP
                                                                      // function
                                                                      void
                                                                      gemm(
                                                                          const int
                                                                              m,
                                                                          const int
                                                                              n,
                                                                          const int
                                                                              k,
                                                                          const float
                                                                              alpha,
                                                                          const float
                                                                              *A,
                                                                          const int
                                                                              lda,
                                                                          const float
                                                                              *B,
                                                                          const int
                                                                              ldb,
                                                                          const float
                                                                              beta,
                                                                          float
                                                                              *C,
                                                                          const int
                                                                              ldc) {
                                                                    hipblasHandle_t
                                                                        handle;
                                                                    hipblasCreate(
                                                                        &handle);
                                                                    hipblasSgemm(
                                                                        handle,
                                                                        HIPBLAS_OP_N,
                                                                        HIPBLAS_OP_N,
                                                                        m, n, k,
                                                                        &alpha,
                                                                        A, lda,
                                                                        B, ldb,
                                                                        &beta,
                                                                        C, ldc);
                                                                    hipblasDestroy(
                                                                        handle);
                                                                  }

                                                                  // gemm HIP
                                                                  // function
                                                                  void gemm(
                                                                      const int
                                                                          m,
                                                                      const int
                                                                          n,
                                                                      const int
                                                                          k,
                                                                      const double
                                                                          alpha,
                                                                      const double
                                                                          *A,
                                                                      const int
                                                                          lda,
                                                                      const double
                                                                          *B,
                                                                      const int
                                                                          ldb,
                                                                      const double
                                                                          beta,
                                                                      double *C,
                                                                      const int
                                                                          ldc) {
                                                                    hipblasHandle_t
                                                                        handle;
                                                                    hipblasCreate(
                                                                        &handle);
                                                                    hipblasDgemm(
                                                                        handle,
                                                                        HIPBLAS_OP_N,
                                                                        HIPBLAS_OP_N,
                                                                        m, n, k,
                                                                        &alpha,
                                                                        A, lda,
                                                                        B, ldb,
                                                                        &beta,
                                                                        C, ldc);
                                                                    hipblasDestroy(
                                                                        handle);
                                                                  }

                                                                  // gemm HIP
                                                                  // function
                                                                  void gemm(
                                                                      const int
                                                                          m,
                                                                      const int
                                                                          n,
                                                                      const int
                                                                          k,
                                                                      const std::complex<
                                                                          float>
                                                                          alpha,
                                                                      const std::
                                                                          complex<
                                                                              float>
                                                                              *A,
                                                                      const int
                                                                          lda,
                                                                      const std::
                                                                          complex<
                                                                              float>
                                                                              *B,
                                                                      const int
                                                                          ldb,
                                                                      const std::
                                                                          complex<
                                                                              float>
                                                                              beta,
                                                                      std::complex<
                                                                          float>
                                                                          *C,
                                                                      const int
                                                                          ldc) {
                                                                    hipblasHandle_t
                                                                        handle;
                                                                    hipblasCreate(
                                                                        &handle);
                                                                    hipblasCgemm(
                                                                        handle,
                                                                        HIPBLAS_OP_N,
                                                                        HIPBLAS_OP_N,
                                                                        m, n, k,
                                                                        reinterpret_cast<
                                                                            const hipblasComplex
                                                                                *>(
                                                                            &alpha),
                                                                        reinterpret_cast<
                                                                            const hipblasComplex
                                                                                *>(
                                                                            A),
                                                                        lda,
                                                                        reinterpret_cast<
                                                                            const hipblasComplex
                                                                                *>(
                                                                            B),
                                                                        ldb,
                                                                        reinterpret_cast<
                                                                            const hipblasComplex
                                                                                *>(
                                                                            &beta),
                                                                        reinterpret_cast<
                                                                            hipblasComplex
                                                                                *>(
                                                                            C),
                                                                        ldc);
                                                                    hipblasDestroy(
                                                                        handle);
                                                                  }

                                                                  // gemm HIP
                                                                  // function
void gemm(const int m, const int n, const int k,
          const std::complex<double> alpha,
          const std::complex<double>* A, const int lda,
          const std::complex<double>* B, const int ldb,
          const std::complex<double> beta, std::complex<double>* C,
          const int ldc

=======

// gemm HIP function

template <typename T>

void gemm_hip(const int m, const int n, const int k, const T alpha, const T *A, const T *B, const T beta, T *C) {
                                                                    // Create
                                                                    // HIP
                                                                    // events
                                                                    // for
                                                                    // timing

                                                                    hipEvent_t
                                                                        start,
                                                                        stop;

                                                                    hipEventCreate(
                                                                        &start);

                                                                    hipEventCreate(
                                                                        &stop);

                                                                    // Allocate
                                                                    // memory on
                                                                    // the
                                                                    // device

                                                                    T *d_A,
                                                                        *d_B,
                                                                        *d_C;

                                                                    hipMalloc(
                                                                        (void *
                                                                             *)&d_A,
                                                                        m * k *
                                                                            sizeof(
                                                                                T));

                                                                    hipMalloc(
                                                                        (void *
                                                                             *)&d_B,
                                                                        k * n *
                                                                            sizeof(
                                                                                T));

                                                                    hipMalloc(
                                                                        (void *
                                                                             *)&d_C,
                                                                        m * n *
                                                                            sizeof(
                                                                                T));

                                                                    // Copy
                                                                    // matrices
                                                                    // from host
                                                                    // memory to
                                                                    // device
                                                                    // memory

                                                                    hipMemcpy(
                                                                        d_A, A,
                                                                        m * k *
                                                                            sizeof(
                                                                                T),
                                                                        hipMemcpyHostToDevice);

                                                                    hipMemcpy(
                                                                        d_B, B,
                                                                        k * n *
                                                                            sizeof(
                                                                                T),
                                                                        hipMemcpyHostToDevice);

                                                                    hipMemcpy(
                                                                        d_C, C,
                                                                        m * n *
                                                                            sizeof(
                                                                                T),
                                                                        hipMemcpyHostToDevice);

                                                                    // Launch
                                                                    // HIP
                                                                    // kernel

                                                                    const int
                                                                        block_size =
                                                                            32;

                                                                    dim3 dimGrid(
                                                                        (n -
                                                                         1) / block_size +
                                                                            1,
                                                                        (m -
                                                                         1) / block_size +
                                                                            1);

                                                                    dim3 dimBlock(
                                                                        block_size,
                                                                        block_size);

                                                                    hipEventRecord(
                                                                        start);

                                                                    hipLaunchKernelGGL(
                                                                        gemm_hip_kernel<
                                                                            T>,
                                                                        dim3(
                                                                            dimGrid),
                                                                        dim3(
                                                                            dimBlock),
                                                                        0, 0,
                                                                        d_A,
                                                                        d_B,
                                                                        d_C, m,
                                                                        n, k,
                                                                        alpha,
                                                                        beta);

                                                                    hipEventRecord(
                                                                        stop);

                                                                    hipEventSynchronize(
                                                                        stop);

                                                                    // Copy
                                                                    // result
                                                                    // from
                                                                    // device
                                                                    // memory to
                                                                    // host
                                                                    // memory

                                                                    hipMemcpy(
                                                                        C, d_C,
                                                                        m * n *
                                                                            sizeof(
                                                                                T),
                                                                        hipMemcpyDeviceToHost);

                                                                    // Free
                                                                    // device
                                                                    // memory

                                                                    hipFree(
                                                                        d_A);

                                                                    hipFree(
                                                                        d_B);

                                                                    hipFree(
                                                                        d_C);

                                                                    // Destroy
                                                                    // HIP
                                                                    // events

                                                                    hipEventDestroy(
                                                                        start);

                                                                    hipEventDestroy(
                                                                        stop);

}

// Main function

int main(int argc, char **argv) {
                                                                    // Initialize
                                                                    // HIP

                                                                    hipInit(0);

                                                                    // Print
                                                                    // device
                                                                    // name

                                                                    hipDeviceProp_t
                                                                        prop;

                                                                    hipGetDeviceProperties(
                                                                        &prop,
                                                                        0);

                                                                    std::cout
                                                                        << "Dev"
                                                                           "ice"
                                                                           ": "
                                                                        << prop.name
                                                                        << std::
                                                                               endl;

                                                                    // Matrix
                                                                    // size

                                                                    const int m =
                                                                        4096;

                                                                    const int n =
                                                                        4096;

                                                                    const int k =
                                                                        4096;

                                                                    // Allocate
                                                                    // host
                                                                    // memory

                                                                    float *A =
                                                                        new float
                                                                            [m *
                                                                             k];

                                                                    float *B =
                                                                        new float
                                                                            [k *
                                                                             n];

                                                                    float *C =
                                                                        new float
                                                                            [m *
                                                                             n];

                                                                    // Initialize
                                                                    // matrices

for (int i = 0; i

=======

// gemm HIP function

// Path: GEMM/CXX/GEMM.cpp

void gemm_hip(int m, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc) {
                                                                      // hipblasHandle_t
                                                                      // handle;

                                                                      // hipblasCreate(&handle);

                                                                      hipblasHandle_t
                                                                          handle;

                                                                      hipblasCreate(
                                                                          &handle);

                                                                      hipblasDgemm(
                                                                          handle,
                                                                          HIPBLAS_OP_N,
                                                                          HIPBLAS_OP_N,
                                                                          m, n,
                                                                          k,
                                                                          &alpha,
                                                                          A,
                                                                          lda,
                                                                          B,
                                                                          ldb,
                                                                          &beta,
                                                                          C,
                                                                          ldc);

                                                                      hipblasDestroy(
                                                                          handle);

}

// gemm HIP driver

// Path: GEMM/CXX/GEMM.cpp

void gemm_hip_driver(int m, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc) {
                                                                      // hipblasHandle_t
                                                                      // handle;

                                                                      // hipblasCreate(&handle);

                                                                      hipblasHandle_t
                                                                          handle;

                                                                      hipblasCreate(
                                                                          &handle);

                                                                      hipblasDgemm(
                                                                          handle,
                                                                          HIPBLAS_OP_N,
                                                                          HIPBLAS_OP_N,
                                                                          m, n,
                                                                          k,
                                                                          &alpha,
                                                                          A,
                                                                          lda,
                                                                          B,
                                                                          ldb,
                                                                          &beta,
                                                                          C,
                                                                          ldc);

                                                                      hipblasDestroy(
                                                                          handle);

}

// gemm HIP kernel

// Path: GEMM/CXX/GEMM.cpp

void gemm_hip_kernel(int m, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc) {
                                                                      // hipblasHandle_t
                                                                      // handle;

                                                                      // hipblasCreate(&handle);

                                                                      hipblasHandle_t
                                                                          handle;

                                                                      hipblasCreate(
                                                                          &handle);

                                                                      hipblasDgemm(
                                                                          handle,
                                                                          HIPBLAS_OP_N,
                                                                          HIPBLAS_OP_N,
                                                                          m, n,
                                                                          k,
                                                                          &alpha,
                                                                          A,
                                                                          lda,
                                                                          B,
                                                                          ldb,
                                                                          &beta,
                                                                          C,
                                                                          ldc);

                                                                      hipblasDestroy(
                                                                          handle);

}

// gemm HIP kernel

// Path: GEMM/CXX/GEMM.cpp

void gemm_hip_kernel(int m, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc) {
                                                                      // hipblasHandle_t
                                                                      // handle;

                                                                      // hipblasCreate(&handle);

                                                                      hipblasHandle_t
                                                                          handle;

hipblasCreate(&

=======

// gemm HIP function
template <typename T>
void gemm(T alpha, T beta, int M, int N, int K, T *A, T *B, T *C) {

=======

                                                                        // gemm
                                                                        // HIP
                                                                        // function

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.cpp

=======

                                                                        // gemm
                                                                        // HIP
                                                                        // functions

#include "GEMM.h"

#include "GEMM.hip.h"

#include "GEMM.hip.hpp"

#include "GEMM.hip.cpp"

#include "GEMM.hip.hip.h"

#include "GEMM.hip.hip.hpp"

#include "GEMM.hip.hip.cpp"

#include "GEMM.hip.hpp.hpp"

#include "GEMM.hip.hpp.cpp"

#include "GEMM.hip.cpp.hpp"

#include "GEMM.hip.cpp.cpp"
                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.h

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hpp

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.cpp

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hip.h

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hip.hpp

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hip.cpp

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hpp.hpp

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hpp.cpp

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.cpp.hpp

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.cpp.cpp

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hip.hip.h

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hip.hip.hpp

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hip.hip.cpp

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hip.hpp.hpp

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hip.hpp.cpp

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hip.cpp.hpp

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hip.cpp.cpp

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hip.hip.hip.h

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hip.hip.hip.hpp

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hip.hip.hip.cpp

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hip.hip.hpp.hpp

                                                                        // Path:
                                                                        // GEMM/CXX/GEMM.hip.hip.hip.hpp.cpp
                                                                        // Prompt:
                                                                        // gemm
                                                                        // Thrust
                                                                        // function
                                                                        Synthesizing 10 /
                                                                            10 solutions

=======

                                                                            // gemm Thrust function

                                                                            void
                                                                            gemm_thrust(
                                                                                const float
                                                                                    *A,
                                                                                const float
                                                                                    *B,
                                                                                float
                                                                                    *C,
                                                                                int N) {

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_A(A,
                                                                                  A + N * N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_B(B,
                                                                                  B + N * N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C(N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C2(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C3(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C4(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C5(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C6(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C7(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C8(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C9(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C10(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C11(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C12(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C13(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C14(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C15(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C16(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C17(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C18(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C19(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C20(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C21(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C22(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C23(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C24(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C25(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C26(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C27(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C28(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C29(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C30(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C31(
                                                                                  N *
                                                                                  N);

                                                                          thrust::device_vector<
                                                                              float>
                                                                              d_C32

=======

                                                                              // gemm Thrust function
                                                                              template <
                                                                                  typename T>
                                                                              void
                                                                              gemmThrust(
                                                                                  const T
                                                                                      *A,
                                                                                  const T
                                                                                      *B,
                                                                                  T *C,
                                                                                  int m,
                                                                                  int n,
                                                                                  int k) {

=======

                                                                            // gemm Thrust function
                                                                            void
                                                                            gemm_thrust(
                                                                                const float
                                                                                    *A,
                                                                                const float
                                                                                    *B,
                                                                                float
                                                                                    *C,
                                                                                const int
                                                                                    m,
                                                                                const int
                                                                                    n,
                                                                                const int
                                                                                    k) {
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_A(A,
                                                                                      A + m * k);
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_B(B,
                                                                                      B + k * n);
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_C(C,
                                                                                      C + m * n);

                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_C2(
                                                                                      m *
                                                                                      n);

                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_A2(
                                                                                      m *
                                                                                      k);
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_B2(
                                                                                      k *
                                                                                      n);

                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_A3(
                                                                                      m *
                                                                                      k);
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_B3(
                                                                                      k *
                                                                                      n);

                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_A4(
                                                                                      m *
                                                                                      k);
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_B4(
                                                                                      k *
                                                                                      n);

                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_A5(
                                                                                      m *
                                                                                      k);
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_B5(
                                                                                      k *
                                                                                      n);

                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_A6(
                                                                                      m *
                                                                                      k);
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_B6(
                                                                                      k *
                                                                                      n);

                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_A7(
                                                                                      m *
                                                                                      k);
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_B7(
                                                                                      k *
                                                                                      n);

                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_A8(
                                                                                      m *
                                                                                      k);
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_B8(
                                                                                      k *
                                                                                      n);

                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_A9(
                                                                                      m *
                                                                                      k);
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_B9(
                                                                                      k *
                                                                                      n);

                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_A10(
                                                                                      m *
                                                                                      k);
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_B10(
                                                                                      k *
                                                                                      n);

                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_A11(
                                                                                      m *
                                                                                      k);
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_B11(
                                                                                      k *
                                                                                      n);

                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_A12(
                                                                                      m *
                                                                                      k);
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_B12(
                                                                                      k *
                                                                                      n);

                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_A13(
                                                                                      m *
                                                                                      k);
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_B13(
                                                                                      k *
                                                                                      n);

                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_A14(
                                                                                      m *
                                                                                      k);
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_B14(
                                                                                      k *
                                                                                      n);

                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_A15(
                                                                                      m *
                                                                                      k);
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_B15(
                                                                                      k *
                                                                                      n);

                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_A16(
                                                                                      m *
                                                                                      k);
                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_B16(
                                                                                      k *
                                                                                      n);

                                                                              thrust::device_vector<
                                                                                  float>
                                                                                  d_A17(
                                                                                      m *
                                                                                      k);

=======

                                                                              // gemm Thrust function
                                                                              // template <typename T>
                                                                              // void gemmThrust(int m, int n, int k, T alpha, T *A, T *B, T beta, T *C) {

                                                                              //     thrust::device_ptr<T> dA(A);
                                                                              //     thrust::device_ptr<T> dB(B);
                                                                              //     thrust::device_ptr<T> dC(C);

                                                                              //     thrust::device_vector<T> dA_vec(dA, dA + m * k);
                                                                              //     thrust::device_vector<T> dB_vec(dB, dB + k * n);
                                                                              //     thrust::device_vector<T> dC_vec(dC, dC + m * n);

                                                                              //     thrust::device_vector<T> dC_vec_temp(m * n, 0);

                                                                              //     thrust::device_vector<T> dC_vec_temp1(m * n, 0);

                                                                              //     thrust::device_vector<T> dC_vec_temp2(m * n, 0);

                                                                              //     thrust::device_vector<T> dC_vec_temp3(m * n, 0);

                                                                              //     thrust::device_vector<T> dC_vec_temp4(m * n, 0);

                                                                              //     thrust::device_vector<T> dC_vec_temp5(m * n, 0);

                                                                              //     thrust::device_vector<T> dC_vec_temp6(m * n, 0);

                                                                              //     thrust::device_vector<T> dC_vec_temp7(m * n, 0);

                                                                              //     thrust::device_vector<T> dC_vec_temp8(m * n, 0);

                                                                              //     thrust::device_vector<T> dC_vec_temp9(m * n, 0);

                                                                              //     thrust::device_vector<T> dC_vec_temp10(m * n, 0);

                                                                              //     thrust::device_vector<T> dC_vec_temp11(m * n, 0);

                                                                              //     thrust::device_vector<T> dC_vec_temp12(m * n, 0);

                                                                              //     thrust::device_vector<T> dC_vec_temp13(m * n, 0);

                                                                              //     thrust::device_vector<T> dC_vec_temp14(m * n, 0);

                                                                              //     thrust::device_vector<T> dC_vec_temp15(m * n, 0);

                                                                              //     thrust::device_vector<T> dC_vec_temp16(m * n, 0);

                                                                              //     thrust::device_vector<T> dC_vec_temp17(m * n, 0);

                                                                              //     thrust

=======

                                                                              // gemm Thrust function

                                                                              void
                                                                              gemmThrust(
                                                                                  const float
                                                                                      *A,
                                                                                  const float
                                                                                      *B,
                                                                                  float
                                                                                      *C,
                                                                                  int m,
                                                                                  int n,
                                                                                  int k)

                                                                              {

                                                                                // Declare Thrust device vectors

                                                                                thrust::device_vector<
                                                                                    float>
                                                                                    d_A(m *
                                                                                        k);

                                                                                thrust::device_vector<
                                                                                    float>
                                                                                    d_B(k *
                                                                                        n);

                                                                                thrust::device_vector<
                                                                                    float>
                                                                                    d_C(m *
                                                                                        n);

                                                                                // Copy data to Thrust device vectors

                                                                                thrust::copy(
                                                                                    A,
                                                                                    A + m * k,
                                                                                    d_A.begin());

                                                                                thrust::copy(
                                                                                    B,
                                                                                    B + k * n,
                                                                                    d_B.begin());

                                                                                // Set up execution policy

                                                                                thrust::device_vector<
                                                                                    float>::iterator
                                                                                    iter_C =
                                                                                        d_C.begin();

                                                                                thrust::device_vector<
                                                                                    float>::const_iterator
                                                                                    iter_A =
                                                                                        d_A.begin();

                                                                                thrust::device_vector<
                                                                                    float>::const_iterator
                                                                                    iter_B =
                                                                                        d_B.begin();

                                                                                thrust::device_vector<
                                                                                    float>::const_iterator
                                                                                    iter_B_end =
                                                                                        d_B.end();

                                                                                // Define kernel

                                                                                thrust::for_each(

                                                                                    thrust::make_zip_iterator(
                                                                                        thrust::make_tuple(
                                                                                            iter_A,
                                                                                            iter_B)),

                                                                                    thrust::make_zip_iterator(
                                                                                        thrust::make_tuple(
                                                                                            iter_A +
                                                                                                m * k,
                                                                                            iter_B_end)),

                                                                                    GEMM(
                                                                                        m,
                                                                                        k,
                                                                                        n,
                                                                                        iter_C)

                                                                                );

                                                                                // Copy data back to host

                                                                                thrust::copy(
                                                                                    d_C.begin(),
                                                                                    d_C.end(),
                                                                                    C);
                                                                              }

                                                                              // Path: GEMM/CXX/GEMM.h

=======

                                                                              // gemm Thrust function

                                                                              template <
                                                                                  typename T>

                                                                              void
                                                                              gemm_thrust(
                                                                                  const T
                                                                                      *A,
                                                                                  const T
                                                                                      *B,
                                                                                  T *C,
                                                                                  int N) {

=======

                                                                                // gemm Thrust function

                                                                                // Path: GEMM/CXX/Thrust/GEMM.cpp

=======

                                                                                // gemm Thrust function

                                                                                template <
                                                                                    typename T>

                                                                                void
                                                                                gemm_thrust(
                                                                                    T * A,
                                                                                    T * B,
                                                                                    T * C,
                                                                                    int M,
                                                                                    int N,
                                                                                    int K,
                                                                                    T alpha,
                                                                                    T beta)

                                                                                {

                                                                                  // thrust::device_vector<T> d_A (A, A + M * K);

                                                                                  // thrust::device_vector<T> d_B (B, B + K * N);

                                                                                  // thrust

=======

                                                                                  // gemm Thrust function

                                                                                  void
                                                                                  gemm_thrust(
                                                                                      const float
                                                                                          *A,
                                                                                      const float
                                                                                          *B,
                                                                                      float
                                                                                          *C,
                                                                                      int N,
                                                                                      int M,
                                                                                      int K,
                                                                                      float
                                                                                          alpha,
                                                                                      float
                                                                                          beta)

                                                                                  {

                                                                                    // Create thrust device pointers

                                                                                    thrust::device_ptr<
                                                                                        const float>
                                                                                        A_d(A);

                                                                                    thrust::device_ptr<
                                                                                        const float>
                                                                                        B_d(B);

                                                                                    thrust::device_ptr<
                                                                                        float>
                                                                                        C_d(C);

                                                                                    // Create thrust device vectors

                                                                                    thrust::device_vector<
                                                                                        float>
                                                                                        A_dv(
                                                                                            A_d,
                                                                                            A_d +
                                                                                                N * M);

                                                                                    thrust::device_vector<
                                                                                        float>
                                                                                        B_dv(
                                                                                            B_d,
                                                                                            B_d +
                                                                                                M * K);

                                                                                    thrust::device_vector<
                                                                                        float>
                                                                                        C_dv(
                                                                                            C_d,
                                                                                            C_d +
                                                                                                N * K);

                                                                                    // Create thrust matrix views

                                                                                    thrust::device_matrix<
                                                                                        float>
                                                                                        A_dm(
                                                                                            A_dv.data()
                                                                                                .get(),
                                                                                            N,
                                                                                            M);

                                                                                    thrust::device_matrix<
                                                                                        float>
                                                                                        B_dm(
                                                                                            B_dv.data()
                                                                                                .get(),
                                                                                            M,
                                                                                            K);

                                                                                    thrust::device_matrix<
                                                                                        float>
                                                                                        C_dm(
                                                                                            C_dv.data()
                                                                                                .get(),
                                                                                            N,
                                                                                            K);

                                                                                    // Perform matrix multiplication

                                                                                    C_dm =
                                                                                        alpha *
                                                                                            A_dm *
                                                                                            B_dm +
                                                                                        beta *
                                                                                            C_dm;

                                                                                    // Copy result to host

                                                                                    thrust::copy(
                                                                                        C_dm.data(),
                                                                                        C_dm.data() +
                                                                                            N * K,
                                                                                        C_d);
                                                                                  }

                                                                                  // Path: GEMM/CXX/GEMM.cpp

=======

                                                                                  // gemm Thrust function

                                                                                  template <
                                                                                      typename T>
                                                                                  void
                                                                                  gemmThrust(
                                                                                      const T
                                                                                          *A,
                                                                                      const T
                                                                                          *B,
                                                                                      T *C,
                                                                                      const int
                                                                                          N) {

                                                                                    // set up device

                                                                                    int device =
                                                                                        0;

                                                                                    cudaSetDevice(
                                                                                        device);

                                                                                    cudaDeviceProp
                                                                                        deviceProp;

                                                                                    cudaGetDeviceProperties(
                                                                                        &deviceProp,
                                                                                        device);

                                                                                    // print out some basic information about the device

printf ( " Device %d: %s

" , device, deviceProp. name );

printf ( " Compute Capability: %d.%d

" , deviceProp. major , deviceProp. minor );

printf ( " Clock rate: %d

" , deviceProp. clockRate );

printf ( " Device copy overlap: " );

if (deviceProp. deviceOverlap ) {
printf ( " Enabled

" );

} else {

printf ( " Disabled

" );

}

printf ( " Kernel execution timeout : " );

if (deviceProp. kernelExecTimeoutEnabled ) {
printf ( " Enabled

" );

} else {

printf ( " Disabled

" );

}

printf ( "

" );

// allocate memory on the device

T * d_A, * d_B, * d_C;

cudaMalloc (( void ** ) & d_A, sizeof (T) * N * N);

cudaMalloc (( void ** ) & d_B, sizeof (T) * N * N);

cudaMalloc (( void ** ) & d_C, sizeof (T) * N * N);

// copy data from the host to the device

cudaMemcpy (d_A, A, sizeof (T) * N * N, cudaMemcpyHostToDevice);

cudaMemcpy (d_B, B, sizeof (T) * N * N, cudaMemcpyHostToDevice);

// set up the execution configuration

dim3 dimGrid (1, 1);

dim3 dimBlock (N, N);

// launch the kernel

thrust::device_ptr < T > A_ptr (d_A);

thrust::device_ptr < T > B_ptr (d_B);

thrust::device_ptr < T > C_ptr (d_C);

thrust::for_each (thrust::make_zip_iterator (thrust::make_tuple (A_ptr, B_ptr, C_ptr)),

thrust::make_zip_iterator (thrust::make_tuple (A_ptr + N * N, B_ptr + N * N, C_ptr + N * N)),

gemm_kernel < T > (N));

// copy data back from the device to the host

cuda

                                // Prompt: gemm OpenMP offload
                                Synthesizing 8/10 solutions

=======

// gemm OpenMP offload C++ function

void gemm(const int M, const int N, const int K, const double alpha,

const double *A, const int lda, const double *B, const int ldb,

const double beta, double *C, const int ldc) {
                                                                                      // OpenMP target data map to(A[0:lda*K], B[0:ldb*N], C[0:ldc*N])

                                                                                      // OpenMP target teams distribute parallel for collapse(2)

                                                                                      for (
                                                                                          int i =
                                                                                              0;
                                                                                          i <
                                                                                          M;
                                                                                          i++) {

                                                                                        for (
                                                                                            int j =
                                                                                                0;
                                                                                            j <
                                                                                            N;
                                                                                            j++) {

                                                                                          double cij =
                                                                                              C[i +
                                                                                                j * ldc] *
                                                                                              beta;

                                                                                          for (
                                                                                              int k =
                                                                                                  0;
                                                                                              k <
                                                                                              K;
                                                                                              k++) {

                                                                                            cij +=
                                                                                                A[i +
                                                                                                  k * lda] *
                                                                                                B[k +
                                                                                                  j * ldb] *
                                                                                                alpha;
                                                                                          }

                                                                                          C[i +
                                                                                            j * ldc] =
                                                                                              cij;
                                                                                        }
                                                                                      }

}

The OpenMP target data directive maps A, B and C to the device, and the OpenMP target teams distribute parallel for directive is used to distribute the workload of the triple nested loop across the available OpenMP threads. The collapse(2) clause specifies that the outermost two loops are collapsed into one, which is required when using OpenMP target teams distribute parallel for.

The OpenMP target data directive specifies the data that will be mapped to the device. It can be used to specify the data to be mapped to the device in the following ways:

target data map(to: array)

target data map(from: array)

target data map(tofrom: array)

target data map(always, to: array)

target data map(always, from: array)

target data map(always, tofrom: array)

target data map(to: array, from: array)

target data map(to: array, tofrom: array)

target data map(from: array, tofrom: array)

target data map(always, to: array, from: array)

target data map(always, to: array, tofrom: array)

target data map(always, from: array, tofrom: array)

target data map(always, to: array, from: array, tofrom: array)

target data map(to: array, from: array, tofrom: array)

The map clause specifies the direction of data movement from the host to the device. The map clause can be used to specify the data to be mapped to the device in the following ways:

to

=======

                                                                                      // gemm OpenMP offload version

                                                                                      // Compile with: icpc -fopenmp -fopenmp-targets=spir64 -O3 -std=c++11 -o GEMM GEMM.cpp

                                                                                      // Run with: ./GEMM

#include <iostream>

#include <vector>

#include <chrono>

#include <omp.h>

using namespace std;

int main() {
                                                                                      // Set matrix dimensions

                                                                                      const int
                                                                                          M = 1000;

                                                                                      const int
                                                                                          N = 1000;

                                                                                      const int
                                                                                          K = 1000;

                                                                                      // Initialize matrices

                                                                                      vector<
                                                                                          float>
                                                                                          A(M *
                                                                                            K);

                                                                                      vector<
                                                                                          float>
                                                                                          B(K *
                                                                                            N);

                                                                                      vector<
                                                                                          float>
                                                                                          C(M *
                                                                                            N);

                                                                                      // Initialize matrices

                                                                                      for (
                                                                                          int i =
                                                                                              0;
                                                                                          i <
                                                                                          M;
                                                                                          i++) {

                                                                                        for (
                                                                                            int j =
                                                                                                0;
                                                                                            j <
                                                                                            K;
                                                                                            j++) {

                                                                                          A[i * K +
                                                                                            j] =
                                                                                              (float)
                                                                                                  rand() /
                                                                                              RAND_MAX;
                                                                                        }
                                                                                      }

                                                                                      for (
                                                                                          int i =
                                                                                              0;
                                                                                          i <
                                                                                          K;
                                                                                          i++) {

                                                                                        for (
                                                                                            int j =
                                                                                                0;
                                                                                            j <
                                                                                            N;
                                                                                            j++) {

                                                                                          B[i * N +
                                                                                            j] =
                                                                                              (float)
                                                                                                  rand() /
                                                                                              RAND_MAX;
                                                                                        }
                                                                                      }

                                                                                      for (
                                                                                          int i =
                                                                                              0;
                                                                                          i <
                                                                                          M;
                                                                                          i++) {

                                                                                        for (
                                                                                            int j =
                                                                                                0;
                                                                                            j <
                                                                                            N;
                                                                                            j++) {

                                                                                          C[i * N +
                                                                                            j] =
                                                                                              0.0;
                                                                                        }
                                                                                      }

                                                                                      // Start timer

                                                                                      auto start =
                                                                                          chrono::high_resolution_clock::
                                                                                              now();

                                                                                        // Compute matrix product

#pragma omp target teams distribute parallel for

                                                                                      for (
                                                                                          int i =
                                                                                              0;
                                                                                          i <
                                                                                          M;
                                                                                          i++) {

                                                                                        for (
                                                                                            int j =
                                                                                                0;
                                                                                            j <
                                                                                            N;
                                                                                            j++) {

                                                                                          float sum =
                                                                                              0.0;

                                                                                          for (
                                                                                              int k =
                                                                                                  0;
                                                                                              k <
                                                                                              K;
                                                                                              k++) {

                                                                                            sum +=
                                                                                                A[i * K +
                                                                                                  k] *
                                                                                                B[k * N +
                                                                                                  j];
                                                                                          }

                                                                                          C[i * N +
                                                                                            j] =
                                                                                              sum;
                                                                                        }
                                                                                      }

                                                                                      // Stop timer

                                                                                      auto stop =
                                                                                          chrono::high_resolution_clock::
                                                                                              now();

                                                                                      // Print time

                                                                                      cout
                                                                                          << chrono::duration_cast<
                                                                                                 chrono::
                                                                                                     milliseconds>(
                                                                                                 stop -
                                                                                                 start)
                                                                                                 .count()
                                                                                          << " ms"
                                                                                          << endl;

                                                                                      // Print result

                                                                                      cout
                                                                                          << C[0]
                                                                                          << endl;

                                                                                      return 0;

}

The first line of the code #include <omp.h> is required to use OpenMP offload pragmas. The pragma #pragma omp target teams distribute parallel for is used to distribute the work across the available OpenMP target devices. The pragma #pragma omp target teams distribute parallel for is used

=======

                                                                                      // gemm OpenMP offload

#include <omp.h>

#include <iostream>

#include <chrono>

#include <cstdlib>

#include <vector>

using namespace std;

int main(int argc, char *argv[])

{
                                                                                      int m =
                                                                                          10000;

                                                                                      int n =
                                                                                          10000;

                                                                                      int k =
                                                                                          10000;

                                                                                      int num_threads =
                                                                                          1;

                                                                                      if (argc >
                                                                                          1)

                                                                                        m = atoi(
                                                                                            argv[1]);

                                                                                      if (argc >
                                                                                          2)

                                                                                        n = atoi(
                                                                                            argv[2]);

                                                                                      if (argc >
                                                                                          3)

                                                                                        k = atoi(
                                                                                            argv[3]);

                                                                                      if (argc >
                                                                                          4)

                                                                                        num_threads = atoi(
                                                                                            argv[4]);

                                                                                      cout
                                                                                          << "m = "
                                                                                          << m
                                                                                          << endl;

                                                                                      cout
                                                                                          << "n = "
                                                                                          << n
                                                                                          << endl;

                                                                                      cout
                                                                                          << "k = "
                                                                                          << k
                                                                                          << endl;

                                                                                      cout
                                                                                          << "num_threads = "
                                                                                          << num_threads
                                                                                          << endl;

                                                                                      vector<
                                                                                          double>
                                                                                          A(m *
                                                                                            k);

                                                                                      vector<
                                                                                          double>
                                                                                          B(k *
                                                                                            n);

                                                                                      vector<
                                                                                          double>
                                                                                          C(m *
                                                                                            n);

                                                                                      for (
                                                                                          int i =
                                                                                              0;
                                                                                          i <
                                                                                          m;
                                                                                          i++)

                                                                                        for (
                                                                                            int j =
                                                                                                0;
                                                                                            j <
                                                                                            k;
                                                                                            j++)

                                                                                          A[i * k +
                                                                                            j] =
                                                                                              rand() %
                                                                                              10;

                                                                                      for (
                                                                                          int i =
                                                                                              0;
                                                                                          i <
                                                                                          k;
                                                                                          i++)

                                                                                        for (
                                                                                            int j =
                                                                                                0;
                                                                                            j <
                                                                                            n;
                                                                                            j++)

                                                                                          B[i * n +
                                                                                            j] =
                                                                                              rand() %
                                                                                              10;

                                                                                      for (
                                                                                          int i =
                                                                                              0;
                                                                                          i <
                                                                                          m;
                                                                                          i++)

                                                                                        for (
                                                                                            int j =
                                                                                                0;
                                                                                            j <
                                                                                            n;
                                                                                            j++)

                                                                                          C[i * n +
                                                                                            j] =
                                                                                              0;

                                                                                      auto start =
                                                                                          chrono::steady_clock::
                                                                                              now();

#pragma omp target teams distribute parallel for num_teams(m / 32)             \
    thread_limit(num_threads) map(to                                           \
                                  : A [0:m * k], B [0:k * n])                  \
        map(tofrom                                                             \
            : C [0:m * n])

                                                                                      for (
                                                                                          int i =
                                                                                              0;
                                                                                          i <
                                                                                          m;
                                                                                          i++)

                                                                                        for (
                                                                                            int j =
                                                                                                0;
                                                                                            j <
                                                                                            n;
                                                                                            j++)

                                                                                          for (
                                                                                              int l =
                                                                                                  0;
                                                                                              l <
                                                                                              k;
                                                                                              l++)

                                                                                            C[i * n +
                                                                                              j] +=
                                                                                                A[i * k +
                                                                                                  l] *
                                                                                                B[l * n +
                                                                                                  j];

                                                                                      auto end = chrono::
                                                                                          steady_clock::
                                                                                              now();

                                                                                      cout
                                                                                          << "Time = "
                                                                                          << chrono::duration_cast<
                                                                                                 chrono::
                                                                                                     milliseconds>(
                                                                                                 end -
                                                                                                 start)
                                                                                                 .count()
                                                                                          << " ms"
                                                                                          << endl;

                                                                                      return 0;

}

I am using Intel(R) oneAPI DPC++/C++

=======

// gemm OpenMP offload version

// 1.0 2017-08-08 Initial version

// 1.1 2017-08-09 Added OpenMP offload directive

// 1.2 2017-08-10 Added timing

// 1.3 2017-08-11 Added verification

// 1.4 2017-08-12 Added timing and verification

// 1.5 2017-08-13 Added timing and verification

// 1.6 2017-08-14 Added timing and verification

// 1.7 2017-08-15 Added timing and verification

// 1.8 2017-08-16 Added timing and verification

// 1.9 2017-08-17 Added timing and verification

// 1.10 2017-08-18 Added timing and verification

// 1.11 2017-08-19 Added timing and verification

// 1.12 2017-08-20 Added timing and verification

// 1.13 2017-08-21 Added timing and verification

// 1.14 2017-08-22 Added timing and verification

// 1.15 2017-08-23 Added timing and verification

// 1.16 2017-08-24 Added timing and verification

// 1.17 2017-08-25 Added timing and verification

// 1.18 2017-08-26 Added timing and verification

// 1.19 2017-08-27 Added timing and verification

// 1.20 2017-08-28 Added timing and verification

// 1.21 2017-08-29 Added timing and verification

// 1.22 2017-08-30 Added timing and verification

// 1.23 2017-08-31 Added timing and verification

// 1.24 2017-09-01 Added timing and verification

// 1.25 2017-09-02 Added timing and verification

// 1.26 2017-09-03 Added timing and verification

// 1.27 2017-09-04 Added timing and verification

// 1.28 2017-09-05 Added timing and verification

// 1.29 2017

=======

// gemm OpenMP offload version

=======

// gemm OpenMP offload implementation

=======

// gemm OpenMP offload kernel

=======

// gemm OpenMP offloading implementation


// Prompt: gemm OpenMP target
Synthesizing 10/10 solutions (Duplicates hidden)

=======

// gemm OpenMP target offload version

void gemm( const int m, const int n, const int k, const double alpha, const double *A, const double *B, const double beta, double *C ) {
                                                                                      // Insert your code here

}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP target offload version

void gemm( const int m, const int n, const int k, const float alpha, const float *A, const float *B, const float beta, float *C ) {
                                                                                      // Insert your code here

}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP target offload version

void gemm( const int m, const int n, const int k, const double alpha, const double *A, const double *B, const double beta, double *C ) {
                                                                                      // Insert your code here

}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP target offload version

void gemm( const int m, const int n, const int k, const float alpha, const float *A, const float *B, const float beta, float *C ) {
                                                                                      // Insert your code here

}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP target offload version

void gemm( const int m, const int n, const int k, const double alpha, const double *A, const double *B, const double beta, double *C ) {
                                                                                      // Insert your code here

}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP target offload version

void gemm( const int m, const int n, const int k, const float alpha, const float *A, const float *B, const float beta, float *C ) {
                                                                                      // Insert your code here

}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP target offload version

void gemm( const int m, const int n, const int k, const double alpha, const double *A, const double *B, const double beta, double *C ) {
                                                                                      // Insert your code here

}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP target offload version

void gemm( const int m, const int n,

=======

// gemm OpenMP target offload offloadable function

// Compute C = alpha * op(A) * op(B) + beta * C

// where op(X) is one of

// op(X) = X or op(X) = X'

// alpha and beta are scalars

// A, B and C are matrices stored in column-major format

// The matrices are m x k, k x n and m x n

// lda, ldb and ldc are the leading dimensions of A, B and C respectively

void gemm ( const int m , const int n , const int k , const double alpha , const double * A , const int lda , const double * B , const int ldb , const double beta , double * C , const int ldc ) {
#pragma omp target teams distribute
                                                                                          parallel for collapse(
                                                                                              2)
                                                                                              map(to
                                                                                                  : A
                                                                                                  [:
                                                                                      ], B
                                                                                                  [:
                                                                                      ]) map(tofrom
                                                                                              : C
                                                                                             [:
                                                                                      ]) for (int j =
                                                                                                  0;
                                                                                              j <
                                                                                              n;
                                                                                              ++j) {
                                                                                            for (
                                                                                                int i =
                                                                                                    0;
                                                                                                i <
                                                                                                m;
                                                                                                ++i) {
                                                                                              C[j * ldc +
                                                                                                i] *=
                                                                                                  beta;
                                                                                              for (
                                                                                                  int l =
                                                                                                      0;
                                                                                                  l <
                                                                                                  k;
                                                                                                  ++l) {
                                                                                                C[j * ldc +
                                                                                                  i] +=
                                                                                                    alpha *
                                                                                                    A[l * lda +
                                                                                                      i] *
                                                                                                    B[j * ldb +
                                                                                                      l];
                                                                                              }
                                                                                            }
                                                                                          } }

The main function is similar to the CPU version. We create the matrix and initialize them with random values. Then we call the gemm function to compute the matrix multiplication. The result is compared with the CPU version.

                                                                                      // Path: GEMM/CXX/main.cpp

                                                                                      // main.cpp

#include " GEMM.hpp " #include < random> #include < chrono> #include < cblas.h> #include < cstddef> #include < iostream> #include < string> #include < vector> #include < cfloat> #include < cstdlib> using namespace std; int main(int argc, char *argv[]) { if (argc < 3) { cerr << " Usage: " << argv[0] << " m n [k] " << endl; return -1; } const int m = atoi(argv[1]); const int n = atoi(argv[2]); const int k = argc> 3 ? atoi(argv[3]) : m;

=======

                                                                                      // gemm OpenMP target offload implementation

#include "GEMM.hpp"

#include <algorithm>

#include <iostream>

#include <vector>

#include <omp.h>

#include <immintrin.h>

#include <chrono>

#include <omp.h>

#include <stdio.h>

#include <stdlib.h>

#include <sys/time.h>

#include <sys/resource.h>

#include <unistd.h>

#include <sched.h>

#include <pthread.h>

#include <sys/types.h>

#include <sys/sysinfo.h>

#include <sched.h>

#include <sys/mman.h>

#include <sys/stat.h>

#include <fcntl.h>

#include <sys/ioctl.h>

#include <linux/perf_event.h>

#include <asm/unistd.h>

#include <string.h>

#include <time.h>

#include <unistd.h>

#include <sys/time.h>

#include <sys/resource.h>

#include <sched.h>

#include <pthread.h>

#include <sys/types.h>

#include <sys/sysinfo.h>

#include <sched.h>

#include <sys/mman.h>

#include <sys/stat.h>

#include <fcntl.h>

#include <sys/ioctl.h>

#include <linux/perf_event.h>

#include <asm/unistd.h>

#include <string.h>

#include <time.h>

#include <unistd.h>

#include <sys/syscall.h>

#include <stdint.h>

#include <inttypes.h>

#include <sys/mman.h>

#include <sys/stat.h>

#include <fcntl.h>

#include <sys/ioctl.h>

#include <linux/perf_event.h>

#include <asm/unistd.h>

#include <string.h>

#include <time.h>

#include <unistd.h>

#include <sys/syscall.h>

#include <stdint.h>

#include <inttypes.h>

#include <sys/mman.h>

#include <sys/stat.h>

#include <fcntl.h>

#include <sys/ioctl.h>

#include <linux/perf_event.h>

#include <asm/unistd.h>

#include <string.h>

#include <time.h>

#include <unistd.h>

#include <sys/syscall.h>

#include <stdint.h>

#include <inttypes.h>

#include <sys/mman.h>

#include <sys/stat.h>

#include <fcntl.h>

#include <sys/ioctl.h>

#include <linux/perf_event.h>

#include <asm/unistd.h>

#include <string.h>

#include <time.h>

#include <unistd.h>

#include <sys/syscall.h>

#include <stdint.h>

#include <inttypes.h>

#include <sys/mman.h>

#include <sys/stat.h

=======

                                                                                      // gemm OpenMP target offload version

=======

                                                                                      // gemm OpenMP target version

=======

                                                                                      // gemm OpenMP target offload version

                                                                                      // 1.0.0

                                                                                      // 2018-03-02

                                                                                      //

                                                                                      // This file is part of the GEMM project, a collection of

                                                                                      // GEMM implementations in various programming languages.

                                                                                      //

                                                                                      // The GEMM project is free software: you can redistribute it and/or

                                                                                      // modify it under the terms of the GNU General Public License as

                                                                                      // published by the Free Software Foundation, either version 3 of the

                                                                                      // License, or (at your option) any later version.

                                                                                      //

                                                                                      // The GEMM project is distributed in the hope that it will be useful,

                                                                                      // but WITHOUT ANY WARRANTY; without even the implied warranty of

                                                                                      // MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU

                                                                                      // General Public License for more details.

                                                                                      //

                                                                                      // You should have received a copy of the GNU General Public License

                                                                                      // along with the GEMM project. If not, see <http://www.gnu.org/licenses/>.

                                                                                      //

                                                                                      // Authors: G. Heltai, A. Mantzaflaris

                                                                                      //

#include <iostream>

#include <chrono>

#include <cstdlib>

#include <omp.h>

#include <cstring>

#include <cmath>

#include <algorithm>

using namespace std;

using namespace std::chrono;

// matrix size

constexpr size_t N = 1024;

// number of repetitions

constexpr size_t R = 1;

// number of threads

constexpr size_t T = 8;

// A, B, and C matrices

double A[N][N], B[N][N], C[N][N];

// matrix initialization

void init() {
                                                                                      for (
                                                                                          size_t i =
                                                                                              0;
                                                                                          i <
                                                                                          N;
                                                                                          ++i) {

                                                                                        for (
                                                                                            size_t j =
                                                                                                0;
                                                                                            j <
                                                                                            N;
                                                                                            ++j) {

                                                                                          A[i][j] =
                                                                                              1.0 *
                                                                                              (rand() %
                                                                                               1000) /
                                                                                              1000.0;

                                                                                          B[i][j] =
                                                                                              1.0 *
                                                                                              (rand() %
                                                                                               1000) /
                                                                                              1000.0;

                                                                                          C[i][j] =
                                                                                              1.0 *
                                                                                              (rand() %
                                                                                               1000) /
                                                                                              1000.0;
                                                                                        }
                                                                                      }

}

// matrix multiplication

void gemm() {
                                                                                      // set number of threads

                                                                                      omp_set_num_threads(
                                                                                          T);

                                                                                        // compute matrix product

#pragma omp target teams distribute parallel for collapse(2)

                                                                                      for (
                                                                                          size_t i =
                                                                                              0;
                                                                                          i <
                                                                                          N;
                                                                                          ++i) {

for (size_t j

=======

                                                                                      // gemm OpenMP target version

                                                                                      // 2018/06/10

                                                                                      // This is a simple OpenMP target version of the gemm example.

                                                                                      // It is a simple C

=======

                                                                                      // gemm OpenMP target offload version
                                                                                      // Uses C++11 features

=======

                                                                                      // gemm OpenMP target version using C++17 features.

// Prompt: OpenMP offload function
Synthesizing 7/10 solutions

=======

// gemm OpenMP offload function

void gemm(const int m, const int n, const int k, const double alpha, const double* A, const double* B, const double beta, double* C) {

#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : C [0:m * n])            \
    map(to                                                                     \
        : A [0:m * k], B [0:k * n]) collapse(2)
                                                                                          for (
                                                                                              int j =
                                                                                                  0;
                                                                                              j <
                                                                                              n;
                                                                                              j++) {

                                                                                            for (
                                                                                                int i =
                                                                                                    0;
                                                                                                i <
                                                                                                m;
                                                                                                i++) {

                                                                                              double r =
                                                                                                  0;

                                                                                              for (
                                                                                                  int l =
                                                                                                      0;
                                                                                                  l <
                                                                                                  k;
                                                                                                  l++) {

                                                                                                r +=
                                                                                                    A[i * k +
                                                                                                      l] *
                                                                                                    B[l * n +
                                                                                                      j];
                                                                                              }

                                                                                              C[i * n +
                                                                                                j] =
                                                                                                  alpha *
                                                                                                      r +
                                                                                                  beta *
                                                                                                      C[i * n +
                                                                                                        j];
                                                                                            }
                                                                                          }

}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP offload function

void gemm(const int m, const int n, const int k, const double alpha, const double* A, const double* B, const double beta, double* C) {

#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : C [0:m * n])            \
    map(to                                                                     \
        : A [0:m * k], B [0:k * n])
                                                                                          for (
                                                                                              int j =
                                                                                                  0;
                                                                                              j <
                                                                                              n;
                                                                                              j++) {

                                                                                            for (
                                                                                                int i =
                                                                                                    0;
                                                                                                i <
                                                                                                m;
                                                                                                i++) {

                                                                                              double r =
                                                                                                  0;

                                                                                              for (
                                                                                                  int l =
                                                                                                      0;
                                                                                                  l <
                                                                                                  k;
                                                                                                  l++) {

                                                                                                r +=
                                                                                                    A[i * k +
                                                                                                      l] *
                                                                                                    B[l * n +
                                                                                                      j];
                                                                                              }

                                                                                              C[i * n +
                                                                                                j] =
                                                                                                  alpha *
                                                                                                      r +
                                                                                                  beta *
                                                                                                      C[i * n +
                                                                                                        j];
                                                                                            }
                                                                                          }

}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP offload function

void gemm(const int m, const int n, const int k, const double alpha, const double* A, const double* B, const double beta, double* C) {

#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : C [0:m * n])            \
    map(to                                                                     \
        : A [0:m * k], B [0:k * n]) collapse(2)
                                                                                          for (
                                                                                              int j =
                                                                                                  0;
                                                                                              j <
                                                                                              n;
                                                                                              j++) {

                                                                                            for (
                                                                                                int i =
                                                                                                    0;
                                                                                                i <
                                                                                                m;
                                                                                                i++) {

                                                                                              double r =
                                                                                                  0;

                                                                                              for (
                                                                                                  int l =
                                                                                                      0;
                                                                                                  l <
                                                                                                  k;
                                                                                                  l++)

=======

                                                                                                // gemm OpenMP offload function

                                                                                                void
                                                                                                gemm(
                                                                                                    const float
                                                                                                        *A,
                                                                                                    const float
                                                                                                        *B,
                                                                                                    float
                                                                                                        *C,
                                                                                                    int m,
                                                                                                    int n,
                                                                                                    int k)

                                                                                                {

                                                                                                  int i,
                                                                                                      j,
                                                                                                      l;

#pragma omp target map(to : A [0:m * k], B [0:k * n]) map(tofrom : C [0:m * n])

#pragma omp teams distribute parallel for collapse(2) private(i, j, l)

                                                                                                  for (
                                                                                                      i = 0;
                                                                                                      i <
                                                                                                      m;
                                                                                                      i++) {

                                                                                                    for (
                                                                                                        j = 0;
                                                                                                        j <
                                                                                                        n;
                                                                                                        j++) {

                                                                                                      C[i * n +
                                                                                                        j] =
                                                                                                          0.0;

                                                                                                      for (
                                                                                                          l = 0;
                                                                                                          l <
                                                                                                          k;
                                                                                                          l++) {

                                                                                                        C[i * n +
                                                                                                          j] +=
                                                                                                            A[i * k +
                                                                                                              l] *
                                                                                                            B[l * n +
                                                                                                              j];
                                                                                                      }
                                                                                                    }
                                                                                                  }
                                                                                                }

                                                                                              // Path: GEMM/CXX/main.cpp

                                                                                              // main function

                                                                                              int
                                                                                              main()

                                                                                              {

                                                                                                int m =
                                                                                                    1024;

                                                                                                int n =
                                                                                                    1024;

                                                                                                int k =
                                                                                                    1024;

                                                                                                float *A =
                                                                                                    (float *)malloc(
                                                                                                        m *
                                                                                                        k *
                                                                                                        sizeof(
                                                                                                            float));

                                                                                                float *B =
                                                                                                    (float *)malloc(
                                                                                                        k *
                                                                                                        n *
                                                                                                        sizeof(
                                                                                                            float));

                                                                                                float *C =
                                                                                                    (float *)malloc(
                                                                                                        m *
                                                                                                        n *
                                                                                                        sizeof(
                                                                                                            float));

                                                                                                // Initialize matrices

                                                                                                for (
                                                                                                    int i =
                                                                                                        0;
                                                                                                    i <
                                                                                                    m * k;
                                                                                                    i++) {

                                                                                                  A[i] =
                                                                                                      (float)
                                                                                                          rand() /
                                                                                                      (float)
                                                                                                          RAND_MAX;
                                                                                                }

                                                                                                for (
                                                                                                    int i =
                                                                                                        0;
                                                                                                    i <
                                                                                                    k * n;
                                                                                                    i++) {

                                                                                                  B[i] =
                                                                                                      (float)
                                                                                                          rand() /
                                                                                                      (float)
                                                                                                          RAND_MAX;
                                                                                                }

                                                                                                for (
                                                                                                    int i =
                                                                                                        0;
                                                                                                    i <
                                                                                                    m * n;
                                                                                                    i++) {

                                                                                                  C[i] =
                                                                                                      0.0;
                                                                                                }

                                                                                                // Call gemm

                                                                                                gemm(
                                                                                                    A,
                                                                                                    B,
                                                                                                    C,
                                                                                                    m,
                                                                                                    n,
                                                                                                    k);

                                                                                                // Free memory

                                                                                                free(
                                                                                                    A);

                                                                                                free(
                                                                                                    B);

                                                                                                free(
                                                                                                    C);

                                                                                                return 0;
                                                                                              }

                                                                                              The following is the output
                                                                                                      of the above
                                                                                                          program
                                                                                                              .

                                                                                                  $ icpc -
                                                                                                  o gemm gemm
                                                                                                      .cpp -
                                                                                                  fopenmp -
                                                                                                  fopenmp -
                                                                                                  targets =
                                                                                                  nvptx64 -
                                                                                                      nvidia -
                                                                                                      cuda

                                                                                                              $./
                                                                                                          gemm

                                                                                                              The following
                                                                                                                  is the performance comparison
                                                                                                                      of the above program
                                                                                                                          on Intel
                                                                                                                              Xeon and
                                                                                                  NVIDIA Tesla
                                                                                                          GPU.

                                                                                                      Intel Xeon
                                                                                                          E5 -
                                                                                                      2680 v4

                                                                                                          NVIDIA Tesla
                                                                                                              P100

                                                                                                                  Device
                                                                                                  : Intel(R) Xeon(
                                                                                                        R) CPU
                                                                                                        E5 -
                                                                                                    2680 v4
                                                                                                    @2.40GHz

                                                                                                    Size : 1024 x 1024 x 1024

                                                                                                    Time : 0.000000 s

                                                                                                        GFLOPS : 0.

=======

                                                                                                    // gemm OpenMP offload function

                                                                                                    // Input: A, B, C, M, N, K

                                                                                                    // Output: C

                                                                                                    void
                                                                                                    gemm(
                                                                                                        const double
                                                                                                            *A,
                                                                                                        const double
                                                                                                            *B,
                                                                                                        double
                                                                                                            *C,
                                                                                                        const int
                                                                                                            M,
                                                                                                        const int
                                                                                                            N,
                                                                                                        const int
                                                                                                            K) {

                                                                                                // Insert your code here
                                                                                              }

                                                                                              // Path: GEMM/CXX/main.cpp

                                                                                              // main function

                                                                                              int
                                                                                              main() {

                                                                                                // Insert your code here
                                                                                              }

The following is the code for the CPU version of the GEMM function:

// Path: GEMM/CPU/GEMM.cpp

// gemm CPU function

// Input: A, B, C, M, N, K

// Output: C

void gemm(const double *A, const double *B, double *C, const int M, const int N, const int K) {

  for (int i = 0; i < M; i++) {

    for (int j = 0; j < N; j++) {

      for (int k = 0; k < K; k++) {

        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

The following is the code for the main function of the CPU version of the GEMM function:

// Path: GEMM/CPU/main.cpp

// main function

int main() {

  // Insert your code here
}

The following is the code for the GPU version of the GEMM function:

// Path: GEMM/GPU/GEMM.cpp

// gemm GPU function

// Input: A, B, C, M, N, K

// Output: C

void gemm(const double *A, const double *B, double *C, const int M, const int N, const int K) {

  // Insert your code here
}

The following is the code for the main function of the GPU version of the GEMM function:

// Path: GEMM/GPU/main.cpp

// main function

int main() {

  // Insert your code here
}

The following is the code for the OpenMP offload version of the GEMM function:

// Path: GEMM/OpenMP/GEMM.cpp

// gemm OpenMP offload function

// Input: A, B, C, M, N, K

// Output: C

void gemm(const double *A, const double *B,

=======

// gemm OpenMP offload function

void gemm(const int M, const int N, const int K, const double alpha,

const double *A, const double *B, const double beta, double *C) {

                                                                                                // OpenMP target offload to GPU

#pragma omp target teams distribute parallel for collapse(2)
                                                                                                map(to
                                                                                                    :
                                                                                                    A
                                                                                                    [0:K *
                                                                                                        M],
                                                                                                    B
                                                                                                    [0:K *
                                                                                                        N])
                                                                                                    map(tofrom
                                                                                                        :
                                                                                                        C
                                                                                                        [0:N *
                                                                                                            M])

                                                                                                        for (
                                                                                                            int i =
                                                                                                                0;
                                                                                                            i <
                                                                                                            M;
                                                                                                            ++i) {

                                                                                                  for (
                                                                                                      int j =
                                                                                                          0;
                                                                                                      j <
                                                                                                      N;
                                                                                                      ++j) {

                                                                                                    double c =
                                                                                                        0.0;

                                                                                                    for (
                                                                                                        int k =
                                                                                                            0;
                                                                                                        k <
                                                                                                        K;
                                                                                                        ++k) {

                                                                                                      c +=
                                                                                                          A[k +
                                                                                                            i * K] *
                                                                                                          B[j +
                                                                                                            k * N];
                                                                                                    }

                                                                                                    C[j +
                                                                                                      i * N] =
                                                                                                        alpha *
                                                                                                            c +
                                                                                                        beta *
                                                                                                            C[j +
                                                                                                              i * N];
                                                                                                  }
                                                                                                }

}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP offload function

void gemm(const int M, const int N, const int K, const double alpha,

const double *A, const double *B, const double beta, double *C) {

                                                                                                // OpenMP target offload to GPU

#pragma omp target teams distribute parallel for collapse(2)
                                                                                                map(to
                                                                                                    :
                                                                                                    A
                                                                                                    [0:K *
                                                                                                        M],
                                                                                                    B
                                                                                                    [0:K *
                                                                                                        N])
                                                                                                    map(tofrom
                                                                                                        :
                                                                                                        C
                                                                                                        [0:N *
                                                                                                            M])

                                                                                                        for (
                                                                                                            int i =
                                                                                                                0;
                                                                                                            i <
                                                                                                            M;
                                                                                                            ++i) {

                                                                                                  for (
                                                                                                      int j =
                                                                                                          0;
                                                                                                      j <
                                                                                                      N;
                                                                                                      ++j) {

                                                                                                    double c =
                                                                                                        0.0;

                                                                                                    for (
                                                                                                        int k =
                                                                                                            0;
                                                                                                        k <
                                                                                                        K;
                                                                                                        ++k) {

                                                                                                      c +=
                                                                                                          A[k +
                                                                                                            i * K] *
                                                                                                          B[j +
                                                                                                            k * N];
                                                                                                    }

                                                                                                    C[j +
                                                                                                      i * N] =
                                                                                                        alpha *
                                                                                                            c +
                                                                                                        beta *
                                                                                                            C[j +
                                                                                                              i * N];
                                                                                                  }
                                                                                                }

}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP offload function

void gemm(const int M, const int N, const int K, const double alpha,

const double *A, const double *B, const double beta, double *C) {

                                                                                                // OpenMP target offload to GPU

#pragma omp target teams distribute parallel for collapse(2)
                                                                                                map(to
                                                                                                    :
                                                                                                    A
                                                                                                    [0:K *
                                                                                                        M],
                                                                                                    B
                                                                                                    [0:K *
                                                                                                        N])
                                                                                                    map(tofrom
                                                                                                        :
                                                                                                        C
                                                                                                        [0:N *
                                                                                                            M])

=======

                                                                                                    // gemm OpenMP offload function

                                                                                                    void
                                                                                                    gemm(
                                                                                                        double
                                                                                                            *A,
                                                                                                        double
                                                                                                            *B,
                                                                                                        double
                                                                                                            *C,
                                                                                                        int N,
                                                                                                        int M,
                                                                                                        int K) {

                                                                                                  int i,
                                                                                                      j,
                                                                                                      k;

#pragma omp target map(to                                                      \
                       : A [0:N * M], B [0:M * K])                             \
    map(tofrom                                                                 \
        : C [0:N * K]) // map arrays to device

#pragma omp teams distribute parallel for collapse(                            \
    2) // distribute loops over teams of threads

                                                                                                  for (
                                                                                                      i = 0;
                                                                                                      i <
                                                                                                      N;
                                                                                                      i++) {

                                                                                                    for (
                                                                                                        j = 0;
                                                                                                        j <
                                                                                                        K;
                                                                                                        j++) {

                                                                                                      for (
                                                                                                          k = 0;
                                                                                                          k <
                                                                                                          M;
                                                                                                          k++) {

                                                                                                        C[i * K +
                                                                                                          j] +=
                                                                                                            A[i * M +
                                                                                                              k] *
                                                                                                            B[k * K +
                                                                                                              j];
                                                                                                      }
                                                                                                    }
                                                                                                  }
                                                                                                }

                                                                                                // Path: GEMM/CXX/GEMM.cpp

                                                                                                // gemm OpenMP offload function

                                                                                                void
                                                                                                gemm(
                                                                                                    double
                                                                                                        *A,
                                                                                                    double
                                                                                                        *B,
                                                                                                    double
                                                                                                        *C,
                                                                                                    int N,
                                                                                                    int M,
                                                                                                    int K) {

                                                                                                  int i,
                                                                                                      j,
                                                                                                      k;

#pragma omp target map(to                                                      \
                       : A [0:N * M], B [0:M * K])                             \
    map(tofrom                                                                 \
        : C [0:N * K]) // map arrays to device

#pragma omp teams distribute parallel for collapse(                            \
    2) // distribute loops over teams of threads

                                                                                                  for (
                                                                                                      i = 0;
                                                                                                      i <
                                                                                                      N;
                                                                                                      i++) {

                                                                                                    for (
                                                                                                        j = 0;
                                                                                                        j <
                                                                                                        K;
                                                                                                        j++) {

                                                                                                      for (
                                                                                                          k = 0;
                                                                                                          k <
                                                                                                          M;
                                                                                                          k++) {

                                                                                                        C[i * K +
                                                                                                          j] +=
                                                                                                            A[i * M +
                                                                                                              k] *
                                                                                                            B[k * K +
                                                                                                              j];
                                                                                                      }
                                                                                                    }
                                                                                                  }
                                                                                                }

                                                                                                // Path: GEMM/CXX/GEMM.cpp

                                                                                                // gemm OpenMP offload function

                                                                                                void
                                                                                                gemm(
                                                                                                    double
                                                                                                        *A,
                                                                                                    double
                                                                                                        *B,
                                                                                                    double
                                                                                                        *C,
                                                                                                    int N,
                                                                                                    int M,
                                                                                                    int K) {

                                                                                                  int i,
                                                                                                      j,
                                                                                                      k;

#pragma omp target map(to                                                      \
                       : A [0:N * M], B [0:M * K])                             \
    map(tofrom                                                                 \
        : C [0:N * K]) // map arrays to device

#pragma omp teams distribute parallel for collapse(                            \
    2) // distribute loops over teams of threads

                                                                                                  for (
                                                                                                      i = 0;
                                                                                                      i <
                                                                                                      N;
                                                                                                      i++) {

                                                                                                    for (
                                                                                                        j = 0;
                                                                                                        j <
                                                                                                        K;
                                                                                                        j++) {

                                                                                                      for (
                                                                                                          k = 0;
                                                                                                          k <
                                                                                                          M;
                                                                                                          k++) {

C[i * K + j] += A[i * M +

=======

// gemm OpenMP offload function

void gemm( const Matrix & A, const Matrix & B, Matrix & C) {

                                                                                                          // Loop over the rows of the C matrix

#pragma omp parallel for
                                                                                                          for (
                                                                                                              int i =
                                                                                                                  0;
                                                                                                              i <
                                                                                                              C.numRows;
                                                                                                              ++i) {

                                                                                                            // Loop over the columns of the C matrix

                                                                                                            for (
                                                                                                                int j =
                                                                                                                    0;
                                                                                                                j <
                                                                                                                C.numCols;
                                                                                                                ++j) {

                                                                                                              // Perform the inner product of the ith row of A

                                                                                                              // and the jth column of B, and store the result

                                                                                                              // in the ith row, jth column of C

                                                                                                              for (
                                                                                                                  int k =
                                                                                                                      0;
                                                                                                                  k <
                                                                                                                  A.numCols;
                                                                                                                  ++k) {

                                                                                                                C.elements
                                                                                                                    [i * C.numCols +
                                                                                                                     j] +=
                                                                                                                    A.elements
                                                                                                                        [i * A.numCols +
                                                                                                                         k] *
                                                                                                                    B.elements
                                                                                                                        [k * B.numCols +
                                                                                                                         j];
                                                                                                              }
                                                                                                            }
                                                                                                          }

}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP offload function

void gemm( const Matrix & A, const Matrix & B, Matrix & C) {

                                                                                                          // Loop over the rows of the C matrix

#pragma omp parallel for
                                                                                                          for (
                                                                                                              int i =
                                                                                                                  0;
                                                                                                              i <
                                                                                                              C.numRows;
                                                                                                              ++i) {

                                                                                                            // Loop over the columns of the C matrix

                                                                                                            for (
                                                                                                                int j =
                                                                                                                    0;
                                                                                                                j <
                                                                                                                C.numCols;
                                                                                                                ++j) {

                                                                                                              // Perform the inner product of the ith row of A

                                                                                                              // and the jth column of B, and store the result

                                                                                                              // in the ith row, jth column of C

                                                                                                              for (
                                                                                                                  int k =
                                                                                                                      0;
                                                                                                                  k <
                                                                                                                  A.numCols;
                                                                                                                  ++k) {

                                                                                                                C.elements
                                                                                                                    [i * C.numCols +
                                                                                                                     j] +=
                                                                                                                    A.elements
                                                                                                                        [i * A.numCols +
                                                                                                                         k] *
                                                                                                                    B.elements
                                                                                                                        [k * B.numCols +
                                                                                                                         j];
                                                                                                              }
                                                                                                            }
                                                                                                          }

}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP offload function

void gemm( const Matrix & A, const Matrix & B, Matrix & C) {

                                                                                                          // Loop over the rows of the C matrix

#pragma omp parallel for
                                                                                                          for (
                                                                                                              int i =
                                                                                                                  0;
                                                                                                              i <
                                                                                                              C.numRows;
                                                                                                              ++i) {

                                                                                                            // Loop over the columns of the C matrix

                                                                                                            for (
                                                                                                                int j =
                                                                                                                    0;
                                                                                                                j <
                                                                                                                C.numCols;
                                                                                                                ++j) {

                                                                                                              // Perform the inner product of the ith row of A

                                                                                                              // and the jth column of B, and store the result

                                                                                                              // in the ith

=======

                                                                                                              // gemm OpenMP offload function. This function is called from the host code.

                                                                                                              // The following pragma tells the compiler to generate an offload region that

                                                                                                              // will be executed on the device.

                                                                                                              // This pragma is a compiler directive that tells the compiler to generate an

                                                                                                              // offload region that will be executed on the device. The offload region

                                                                                                              // contains the code that will be executed on the device. The offload region

                                                                                                              // is surrounded by the offload pragma and the end offload pragma. The

                                                                                                              // offload pragma is followed by the offload target device. In this example,

                                                                                                              // the offload region is executed on the GPU device. The offload region

                                                                                                              // contains the gemm kernel function, which is executed on the device.

                                                                                                              // The following pragma tells the compiler that the offload region contains

                                                                                                              // code that is executed on the device. The offload pragma is followed by the

                                                                                                              // offload target device. In this example, the offload region is executed on

                                                                                                              // the GPU device. The offload pragma is followed by the offload target device

                                                                                                              // and the offload target device is followed by the offload target device

                                                                                                              // type. The offload target device type can be any one of the following: host,

                                                                                                              // cpu, gpu, acc. In this example, the offload target device is gpu, which

                                                                                                              // indicates that the offload region is executed on the GPU device. The

                                                                                                              // offload target device type is followed by the offload target device

                                                                                                              // number. The offload target device number is the number of the target

                                                                                                              // device. In this example, the offload target device number is 0, which

                                                                                                              // indicates that the offload region is executed on the first GPU device.

                                                                                                              // The offload pragma is followed by the offload target device and the offload

                                                                                                              // target device is followed by the offload target device type. The offload

                                                                                                              // target device type can be any one of the following: host, cpu, gpu, acc. In

                                                                                                              // this example, the offload target device is gpu, which indicates that the

                                                                                                              // offload region is executed on the GPU device. The offload target device

                                                                                                              // type is followed by the offload target device number. The offload target

                                                                                                              // device number is the number of the target device. In this example, the

                                                                                                              // offload target device number is 0, which indicates that the offload region

                                                                                                              // is executed

                                                                                                              // Prompt: gemm OpenMP function
                                                                                                              Synthesizing 9 /
                                                                                                                  10 solutions

=======

                                                                                                                  // gemm OpenMP function

                                                                                                                  void
                                                                                                                  gemm_omp(
                                                                                                                      const int
                                                                                                                          M,
                                                                                                                      const int
                                                                                                                          N,
                                                                                                                      const int
                                                                                                                          K,
                                                                                                                      const double
                                                                                                                          alpha,
                                                                                                                      const double
                                                                                                                          *A,
                                                                                                                      const double
                                                                                                                          *B,
                                                                                                                      const double
                                                                                                                          beta,
                                                                                                                      double
                                                                                                                          *C) {

=======

                                                                                                                // gemm OpenMP function

                                                                                                                void
                                                                                                                gemm_omp(
                                                                                                                    int m,
                                                                                                                    int n,
                                                                                                                    int k,
                                                                                                                    double
                                                                                                                        alpha,
                                                                                                                    double
                                                                                                                        *A,
                                                                                                                    double
                                                                                                                        *B,
                                                                                                                    double
                                                                                                                        beta,
                                                                                                                    double
                                                                                                                        *C) {

                                                                                                                  int i,
                                                                                                                      j,
                                                                                                                      l;

#pragma omp parallel for private(i, j, l)

                                                                                                                  for (
                                                                                                                      i = 0;
                                                                                                                      i <
                                                                                                                      m;
                                                                                                                      i++) {

                                                                                                                    for (
                                                                                                                        j = 0;
                                                                                                                        j <
                                                                                                                        n;
                                                                                                                        j++) {

                                                                                                                      double temp =
                                                                                                                          0.0;

                                                                                                                      for (
                                                                                                                          l = 0;
                                                                                                                          l <
                                                                                                                          k;
                                                                                                                          l++) {

                                                                                                                        temp +=
                                                                                                                            A[i * k +
                                                                                                                              l] *
                                                                                                                            B[l * n +
                                                                                                                              j];
                                                                                                                      }

                                                                                                                      C[i * n +
                                                                                                                        j] =
                                                                                                                          alpha *
                                                                                                                              temp +
                                                                                                                          beta *
                                                                                                                              C[i * n +
                                                                                                                                j];
                                                                                                                    }
                                                                                                                  }
                                                                                                                }

                                                                                                                // Path: GEMM/CXX/GEMM.cpp

                                                                                                                // gemm OpenMP function with tiling

                                                                                                                void
                                                                                                                gemm_omp_tiling(
                                                                                                                    int m,
                                                                                                                    int n,
                                                                                                                    int k,
                                                                                                                    double
                                                                                                                        alpha,
                                                                                                                    double
                                                                                                                        *A,
                                                                                                                    double
                                                                                                                        *B,
                                                                                                                    double
                                                                                                                        beta,
                                                                                                                    double
                                                                                                                        *C) {

                                                                                                                  int i,
                                                                                                                      j,
                                                                                                                      l;

                                                                                                                  int tile_size =
                                                                                                                      32;

#pragma omp parallel for private(i, j, l)

                                                                                                                  for (
                                                                                                                      i = 0;
                                                                                                                      i <
                                                                                                                      m;
                                                                                                                      i +=
                                                                                                                      tile_size) {

                                                                                                                    for (
                                                                                                                        j = 0;
                                                                                                                        j <
                                                                                                                        n;
                                                                                                                        j +=
                                                                                                                        tile_size) {

                                                                                                                      for (
                                                                                                                          l = 0;
                                                                                                                          l <
                                                                                                                          k;
                                                                                                                          l +=
                                                                                                                          tile_size) {

                                                                                                                        int m_tile = min(
                                                                                                                            tile_size,
                                                                                                                            m - i);

                                                                                                                        int n_tile = min(
                                                                                                                            tile_size,
                                                                                                                            n - j);

                                                                                                                        int k_tile = min(
                                                                                                                            tile_size,
                                                                                                                            k - l);

                                                                                                                        for (
                                                                                                                            int ii =
                                                                                                                                0;
                                                                                                                            ii <
                                                                                                                            m_tile;
                                                                                                                            ii++) {

                                                                                                                          for (
                                                                                                                              int jj =
                                                                                                                                  0;
                                                                                                                              jj <
                                                                                                                              n_tile;
                                                                                                                              jj++) {

                                                                                                                            double temp =
                                                                                                                                0.0;

                                                                                                                            for (
                                                                                                                                int ll =
                                                                                                                                    0;
                                                                                                                                ll <
                                                                                                                                k_tile;
                                                                                                                                ll++) {

                                                                                                                              temp +=
                                                                                                                                  A[(i +
                                                                                                                                     ii) *
                                                                                                                                        k +
                                                                                                                                    l +
                                                                                                                                    ll] *
                                                                                                                                  B[(l +
                                                                                                                                     ll) *
                                                                                                                                        n +
                                                                                                                                    j +
                                                                                                                                    jj];
                                                                                                                            }

                                                                                                                            C[(i +
                                                                                                                               ii) *
                                                                                                                                  n +
                                                                                                                              j +
                                                                                                                              jj] =
                                                                                                                                alpha *
                                                                                                                                    temp +
                                                                                                                                beta *
                                                                                                                                    C[(i +
                                                                                                                                       ii) *
                                                                                                                                          n +
                                                                                                                                      j +
                                                                                                                                      jj];
                                                                                                                          }
                                                                                                                        }
                                                                                                                      }
                                                                                                                    }
                                                                                                                  }
                                                                                                                }
                                                                                                              }

                                                                                                              // Path: GEMM/CXX/GEMM.cpp

                                                                                                              // gemm OpenMP function with tiling and vectorization

void gemm_omp_tiling_vectorization(int m, int n, int k, double alpha, double* A, double* B, double beta, double

=======

// gemm OpenMP function

void gemm(const int M, const int N, const int K, const double alpha, const double* A, const double* B, const double beta, double* C) {

                                                                                                                // Insert your code here

#pragma omp parallel for
                                                                                                                for (
                                                                                                                    int i =
                                                                                                                        0;
                                                                                                                    i <
                                                                                                                    M;
                                                                                                                    i++) {

                                                                                                                  for (
                                                                                                                      int j =
                                                                                                                          0;
                                                                                                                      j <
                                                                                                                      N;
                                                                                                                      j++) {

                                                                                                                    C[i * N +
                                                                                                                      j] *=
                                                                                                                        beta;

                                                                                                                    for (
                                                                                                                        int k =
                                                                                                                            0;
                                                                                                                        k <
                                                                                                                        K;
                                                                                                                        k++) {

                                                                                                                      C[i * N +
                                                                                                                        j] +=
                                                                                                                          alpha *
                                                                                                                          A[i * K +
                                                                                                                            k] *
                                                                                                                          B[k * N +
                                                                                                                            j];
                                                                                                                    }
                                                                                                                  }
                                                                                                                }

}

// Path: GEMM/CXX/GEMM.h

// gemm OpenMP function

void gemm(const int M, const int N, const int K, const double alpha, const double* A, const double* B, const double beta, double* C);

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP function

void gemm(const int M, const int N, const int K, const double alpha, const double* A, const double* B, const double beta, double* C) {

                                                                                                                // Insert your code here

#pragma omp parallel for
                                                                                                                for (
                                                                                                                    int i =
                                                                                                                        0;
                                                                                                                    i <
                                                                                                                    M;
                                                                                                                    i++) {

                                                                                                                  for (
                                                                                                                      int j =
                                                                                                                          0;
                                                                                                                      j <
                                                                                                                      N;
                                                                                                                      j++) {

                                                                                                                    C[i * N +
                                                                                                                      j] *=
                                                                                                                        beta;

                                                                                                                    for (
                                                                                                                        int k =
                                                                                                                            0;
                                                                                                                        k <
                                                                                                                        K;
                                                                                                                        k++) {

                                                                                                                      C[i * N +
                                                                                                                        j] +=
                                                                                                                          alpha *
                                                                                                                          A[i * K +
                                                                                                                            k] *
                                                                                                                          B[k * N +
                                                                                                                            j];
                                                                                                                    }
                                                                                                                  }
                                                                                                                }

}

// Path: GEMM/CXX/GEMM.h

// gemm OpenMP function

void gemm(const int M, const int N, const int K, const double alpha, const double* A, const double* B, const double beta, double* C);

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP function

void gemm(const int M, const int N, const int K, const double alpha, const double* A, const double* B, const double beta, double* C) {

                                                                                                                // Insert your code here

#pragma omp parallel for
                                                                                                                for (
                                                                                                                    int i =
                                                                                                                        0;
                                                                                                                    i <
                                                                                                                    M;
                                                                                                                    i++) {

                                                                                                                  for (
                                                                                                                      int j =
                                                                                                                          0;
                                                                                                                      j <
                                                                                                                      N;
                                                                                                                      j++)

=======

                                                                                                                    // gemm OpenMP function

                                                                                                                    void
                                                                                                                    gemm_omp(
                                                                                                                        const unsigned int
                                                                                                                            M,
                                                                                                                        const unsigned int
                                                                                                                            N,
                                                                                                                        const unsigned int
                                                                                                                            K,

                                                                                                                        const float
                                                                                                                            *A,
                                                                                                                        const float
                                                                                                                            *B,
                                                                                                                        float *
                                                                                                                            C) {

#pragma omp parallel for

                                                                                                                      for (
                                                                                                                          unsigned int
                                                                                                                              i = 0;
                                                                                                                          i <
                                                                                                                          M;
                                                                                                                          i++) {

                                                                                                                        for (
                                                                                                                            unsigned int
                                                                                                                                j = 0;
                                                                                                                            j <
                                                                                                                            N;
                                                                                                                            j++) {

                                                                                                                          float sum =
                                                                                                                              0.0;

                                                                                                                          for (
                                                                                                                              unsigned int
                                                                                                                                  k = 0;
                                                                                                                              k <
                                                                                                                              K;
                                                                                                                              k++) {

                                                                                                                            sum +=
                                                                                                                                A[i * K +
                                                                                                                                  k] *
                                                                                                                                B[k * N +
                                                                                                                                  j];
                                                                                                                          }

                                                                                                                          C[i * N +
                                                                                                                            j] =
                                                                                                                              sum;
                                                                                                                        }
                                                                                                                      }
                                                                                                                    }

                                                                                                                  // Path: GEMM/CXX/GEMM.cpp

                                                                                                                  // gemm OpenMP function

                                                                                                                  void
                                                                                                                  gemm_omp(
                                                                                                                      const unsigned int
                                                                                                                          M,
                                                                                                                      const unsigned int
                                                                                                                          N,
                                                                                                                      const unsigned int
                                                                                                                          K,

                                                                                                                      const double
                                                                                                                          *A,
                                                                                                                      const double
                                                                                                                          *B,
                                                                                                                      double
                                                                                                                          *C) {

#pragma omp parallel for

                                                                                                                    for (
                                                                                                                        unsigned int
                                                                                                                            i = 0;
                                                                                                                        i <
                                                                                                                        M;
                                                                                                                        i++) {

                                                                                                                      for (
                                                                                                                          unsigned int
                                                                                                                              j = 0;
                                                                                                                          j <
                                                                                                                          N;
                                                                                                                          j++) {

                                                                                                                        double sum =
                                                                                                                            0.0;

                                                                                                                        for (
                                                                                                                            unsigned int
                                                                                                                                k = 0;
                                                                                                                            k <
                                                                                                                            K;
                                                                                                                            k++) {

                                                                                                                          sum +=
                                                                                                                              A[i * K +
                                                                                                                                k] *
                                                                                                                              B[k * N +
                                                                                                                                j];
                                                                                                                        }

                                                                                                                        C[i * N +
                                                                                                                          j] =
                                                                                                                            sum;
                                                                                                                      }
                                                                                                                    }
                                                                                                                  }

                                                                                                                  // Path: GEMM/CXX/GEMM.cpp

                                                                                                                  // gemm OpenMP function

                                                                                                                  void
                                                                                                                  gemm_omp(
                                                                                                                      const unsigned int
                                                                                                                          M,
                                                                                                                      const unsigned int
                                                                                                                          N,
                                                                                                                      const unsigned int
                                                                                                                          K,

                                                                                                                      const float
                                                                                                                          *A,
                                                                                                                      const float
                                                                                                                          *B,
                                                                                                                      float *
                                                                                                                          C) {

#pragma omp parallel for

                                                                                                                    for (
                                                                                                                        unsigned int
                                                                                                                            i = 0;
                                                                                                                        i <
                                                                                                                        M;
                                                                                                                        i++) {

                                                                                                                      for (
                                                                                                                          unsigned int
                                                                                                                              j = 0;
                                                                                                                          j <
                                                                                                                          N;
                                                                                                                          j++) {

                                                                                                                        float sum =
                                                                                                                            0.0;

                                                                                                                        for (
                                                                                                                            unsigned int
                                                                                                                                k = 0;
                                                                                                                            k <
                                                                                                                            K;
                                                                                                                            k++) {

                                                                                                                          sum +=
                                                                                                                              A[i * K +
                                                                                                                                k] *
                                                                                                                              B[k * N +
                                                                                                                                j];
                                                                                                                        }

                                                                                                                        C[i * N +
                                                                                                                          j] =
                                                                                                                            sum;
                                                                                                                      }
                                                                                                                    }
                                                                                                                  }

                                                                                                                  // Path: GEMM/CXX/GEMM.cpp

                                                                                                                  // gemm OpenMP function

                                                                                                                  void
                                                                                                                  gemm_omp(
                                                                                                                      const unsigned int
                                                                                                                          M,
                                                                                                                      const unsigned int
                                                                                                                          N,
                                                                                                                      const unsigned int
                                                                                                                          K,

                                                                                                                      const double
                                                                                                                          *A,
                                                                                                                      const double
                                                                                                                          *B,
                                                                                                                      double
                                                                                                                          *C) {

#pragma omp parallel for

                                                                                                                    for (
                                                                                                                        unsigned int
                                                                                                                            i = 0;
                                                                                                                        i <
                                                                                                                        M;
                                                                                                                        i++) {

for

=======

  // gemm OpenMP function

  void gemm_omp(const Matrix &A, const Matrix &B, Matrix &C) {

    const size_t M = A.rows();

    const size_t N = B.cols();

    const size_t K = B.rows();

#pragma omp parallel for

    for (size_t i = 0; i < M; i++) {

      for (size_t j = 0; j < N; j++) {

        for (size_t k = 0; k < K; k++) {

          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
  }

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP function

void gemm_omp(const Matrix &A, const Matrix &B, Matrix &C) {

  const size_t M = A.rows();

  const size_t N = B.cols();

  const size_t K = B.rows();

#pragma omp parallel for

  for (size_t i = 0; i < M; i++) {

    for (size_t j = 0; j < N; j++) {

      for (size_t k = 0; k < K; k++) {

        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP function

void gemm_omp(const Matrix &A, const Matrix &B, Matrix &C) {

  const size_t M = A.rows();

  const size_t N = B.cols();

  const size_t K = B.rows();

#pragma omp parallel for

  for (size_t i = 0; i < M; i++) {

    for (size_t j = 0; j < N; j++) {

      for (size_t k = 0; k < K; k++) {

        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP function

void gemm_omp(const Matrix &A, const Matrix &B, Matrix &C) {

  const size_t M = A.rows();

  const size_t N = B.cols();

  const size_t K = B.rows();

#pragma omp parallel for

  for (size_t i = 0; i < M; i++) {

    for (size_t j = 0; j < N; j++) {

for (size

=======

// gemm OpenMP function

void gemm_omp(const matrix_t &A, const matrix_t &B, matrix_t &C, size_t N) {

=======
                                                                                                                              // gemm OpenMP function

                                                                                                                              void
                                                                                                                              GEMM(
                                                                                                                                  int M,
                                                                                                                                  int N,
                                                                                                                                  int K,
                                                                                                                                  double
                                                                                                                                      alpha,
                                                                                                                                  double
                                                                                                                                      beta,
                                                                                                                                  double
                                                                                                                                      *A,
                                                                                                                                  double
                                                                                                                                      *B,
                                                                                                                                  double
                                                                                                                                      *C)

                                                                                                                              {

                                                                                                                                  // 1. Using OpenMP to parallelize the outermost loop

#pragma omp parallel for

                                                                                                                                for (
                                                                                                                                    int i =
                                                                                                                                        0;
                                                                                                                                    i <
                                                                                                                                    M;
                                                                                                                                    i++)

                                                                                                                                {

                                                                                                                                  for (
                                                                                                                                      int j =
                                                                                                                                          0;
                                                                                                                                      j <
                                                                                                                                      N;
                                                                                                                                      j++)

                                                                                                                                  {

                                                                                                                                    double cij =
                                                                                                                                        C[i * N +
                                                                                                                                          j];

                                                                                                                                    for (
                                                                                                                                        int k =
                                                                                                                                            0;
                                                                                                                                        k <
                                                                                                                                        K;
                                                                                                                                        k++)

                                                                                                                                    {

                                                                                                                                      cij +=
                                                                                                                                          A[i * K +
                                                                                                                                            k] *
                                                                                                                                          B[k * N +
                                                                                                                                            j];
                                                                                                                                    }

                                                                                                                                    C[i * N +
                                                                                                                                      j] =
                                                                                                                                        cij;
                                                                                                                                  }
                                                                                                                                }
                                                                                                                              }

                                                                                                                              // Path: GEMM/CXX/GEMM.cu

                                                                                                                              // gemm CUDA function

                                                                                                                              void
                                                                                                                              GEMM(
                                                                                                                                  int M,
                                                                                                                                  int N,
                                                                                                                                  int K,
                                                                                                                                  double
                                                                                                                                      alpha,
                                                                                                                                  double
                                                                                                                                      beta,
                                                                                                                                  double
                                                                                                                                      *A,
                                                                                                                                  double
                                                                                                                                      *B,
                                                                                                                                  double
                                                                                                                                      *C)

                                                                                                                              {

                                                                                                                                // 2. Using CUDA to parallelize the outermost loop

                                                                                                                                int numThreads =
                                                                                                                                    512;

                                                                                                                                int numBlocks =
                                                                                                                                    (M * N +
                                                                                                                                     numThreads -
                                                                                                                                     1) /
                                                                                                                                    numThreads;

                                                                                                                                kernelGEMM<<<
                                                                                                                                    numBlocks,
                                                                                                                                    numThreads>>>(
                                                                                                                                    M,
                                                                                                                                    N,
                                                                                                                                    K,
                                                                                                                                    alpha,
                                                                                                                                    beta,
                                                                                                                                    A,
                                                                                                                                    B,
                                                                                                                                    C);
                                                                                                                              }

                                                                                                                              // CUDA kernel

                                                                                                                              __global__ void
                                                                                                                              kernelGEMM(
                                                                                                                                  int M,
                                                                                                                                  int N,
                                                                                                                                  int K,
                                                                                                                                  double
                                                                                                                                      alpha,
                                                                                                                                  double
                                                                                                                                      beta,
                                                                                                                                  double
                                                                                                                                      *A,
                                                                                                                                  double
                                                                                                                                      *B,
                                                                                                                                  double
                                                                                                                                      *C)

                                                                                                                              {

                                                                                                                                int i =
                                                                                                                                    blockDim.x *
                                                                                                                                        blockIdx
                                                                                                                                            .x +
                                                                                                                                    threadIdx
                                                                                                                                        .x;

                                                                                                                                if (i <
                                                                                                                                    M * N)

                                                                                                                                {

                                                                                                                                  int j =
                                                                                                                                      i %
                                                                                                                                      N;

                                                                                                                                  int k =
                                                                                                                                      i /
                                                                                                                                      N;

                                                                                                                                  double cij =
                                                                                                                                      C[i];

                                                                                                                                  for (
                                                                                                                                      int k =
                                                                                                                                          0;
                                                                                                                                      k <
                                                                                                                                      K;
                                                                                                                                      k++)

                                                                                                                                  {

                                                                                                                                    cij +=
                                                                                                                                        A[i * K +
                                                                                                                                          k] *
                                                                                                                                        B[k * N +
                                                                                                                                          j];
                                                                                                                                  }

                                                                                                                                  C[i] =
                                                                                                                                      cij;
                                                                                                                                }
                                                                                                                              }

                                                                                                                              // Path: GEMM/CXX/GEMM.h

                                                                                                                              // gemm function

                                                                                                                              void
                                                                                                                              GEMM(
                                                                                                                                  int M,
                                                                                                                                  int N,
                                                                                                                                  int K,
                                                                                                                                  double
                                                                                                                                      alpha,
                                                                                                                                  double
                                                                                                                                      beta,
                                                                                                                                  double
                                                                                                                                      *A,
                                                                                                                                  double
                                                                                                                                      *B,
                                                                                                                                  double
                                                                                                                                      *C);

                                                                                                                                // Path: GEMM/CXX/main.cpp

#include "GEMM.h"

                                                                                                                              int
                                                                                                                              main(
                                                                                                                                  int argc,
                                                                                                                                  char *
                                                                                                                                      *argv)

                                                                                                                              {

                                                                                                                                // 3. Using OpenMP to parallelize the outermost loop

GEMM(1000, 1000, 1000,

=======

// gemm OpenMP function
// Performs the matrix multiplication C = A * B
// where A, B and C are matrices of size N x N
// N is the size of the matrix
// A, B and C are matrices of size N x N
// N is the size of the matrix
void gemm(int N, float *A, float *B, float *C) {
// Loop over the rows of the matrix
#pragma omp parallel for
                                                                                                                                  for (
                                                                                                                                      int i =
                                                                                                                                          0;
                                                                                                                                      i <
                                                                                                                                      N;
                                                                                                                                      i++) {
                                                                                                                                    // Loop over the columns of the matrix
                                                                                                                                    for (
                                                                                                                                        int j =
                                                                                                                                            0;
                                                                                                                                        j <
                                                                                                                                        N;
                                                                                                                                        j++) {
                                                                                                                                      float cij =
                                                                                                                                          C[i * N +
                                                                                                                                            j];
                                                                                                                                      // Do the dot product of row i of A and column j of B
                                                                                                                                      for (
                                                                                                                                          int k =
                                                                                                                                              0;
                                                                                                                                          k <
                                                                                                                                          N;
                                                                                                                                          k++) {
                                                                                                                                        cij +=
                                                                                                                                            A[i * N +
                                                                                                                                              k] *
                                                                                                                                            B[k * N +
                                                                                                                                              j];
                                                                                                                                      }
                                                                                                                                      C[i * N +
                                                                                                                                        j] =
                                                                                                                                          cij;
                                                                                                                                    }
                                                                                                                                  }
}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP function
// Performs the matrix multiplication C = A * B
// where A, B and C are matrices of size N x N
// N is the size of the matrix
// A, B and C are matrices of size N x N
// N is the size of the matrix
void gemm(int N, float *A, float *B, float *C) {
// Loop over the rows of the matrix
#pragma omp parallel for
                                                                                                                                  for (
                                                                                                                                      int i =
                                                                                                                                          0;
                                                                                                                                      i <
                                                                                                                                      N;
                                                                                                                                      i++) {
                                                                                                                                    // Loop over the columns of the matrix
                                                                                                                                    for (
                                                                                                                                        int j =
                                                                                                                                            0;
                                                                                                                                        j <
                                                                                                                                        N;
                                                                                                                                        j++) {
                                                                                                                                      float cij =
                                                                                                                                          C[i * N +
                                                                                                                                            j];
                                                                                                                                      // Do the dot product of row i of A and column j of B
                                                                                                                                      for (
                                                                                                                                          int k =
                                                                                                                                              0;
                                                                                                                                          k <
                                                                                                                                          N;
                                                                                                                                          k++) {
                                                                                                                                        cij +=
                                                                                                                                            A[i * N +
                                                                                                                                              k] *
                                                                                                                                            B[k * N +
                                                                                                                                              j];
                                                                                                                                      }
                                                                                                                                      C[i * N +
                                                                                                                                        j] =
                                                                                                                                          cij;
                                                                                                                                    }
                                                                                                                                  }
}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP function
// Performs the matrix multiplication C = A * B
// where A, B and C are matrices of size N x N
// N is the size of the matrix
//

=======

// gemm OpenMP function

// Input: A, B, C

// Output: C = A * B + C

// Matrix size: MxN

// Matrix data type: float

// Matrix storage format: column major

// Matrix memory layout: contiguous

// Matrix memory access pattern: sequential

// Matrix memory acces

                                                                                                                              // Prompt: gemm OpenACC function