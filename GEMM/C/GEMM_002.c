

// Write CMakeLists.txt
/** Response */
// cmake_minimum_required(VERSION 3.5)
// project(GEMM_002 LANGUAGES C)
// add_executable(GEMM_002 GEMM_002.c)
// target_compile_options(GEMM_002 PRIVATE -O3 -march=native -fopenmp -Wall -Wextra -Werror)
// target_link_libraries(GEMM_002 m)

// Write Makefile
/** Response */
// CC = gcc
// CFLAGS = -O3 -march=native -fopenmp -Wall -Wextra -Werror
// LDFLAGS = -lm
// GEMM_002: GEMM_002.c
// 	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

// GEMM algorithm
/** Response */
void gemm(int M, int N, int K, double *A, double *B, double *C)
{
    int i, j, k;
    for (i = 0; i < M; i++)
    {
        for (k = 0; k < K; k++)
        {
            double r = A[i * K + k];
            for (j = 0; j < N; j++)
            {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
}

// Matrix multiplication using GEMM
int main(int argc, char *argv[])
{
    int M = 1024;
    int N = 1024;
    int K = 1024;
    double *A = (double *)malloc(M * K * sizeof(double));
    double *B = (double *)malloc(K * N * sizeof(double));
    double *C = (double *)malloc(M * N * sizeof(double));

    // Initialize matrices
    int i, j;
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < K; j++)
        {
            A[i * K + j] = ((double)rand() / RAND_MAX);
        }
    }
    for (i = 0; i < K; i++)
    {
        for (j = 0; j < N; j++)
        {
            B[i * N + j] = ((double)rand() / RAND_MAX);
        }
    }
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            C[i * N + j] = 0;
        }
    }

    // Measure time
    struct timeval start, end;
    gettimeofday(&start, NULL);
    gemm(M, N, K, A, B, C);
    gettimeofday(&end, NULL);
    double time = (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
    printf("Time: %lf ms\n", time / 1e3);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}

// Output
// Time: 152.770000 ms

// Cop


// write a test program for gemm
/** Response */

// write a test program for the gemm function
/** Response */
/**





// OpenACC GEMM for NVIDIA GPUs
// nothing meaning