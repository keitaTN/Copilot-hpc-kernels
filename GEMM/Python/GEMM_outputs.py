

## Prompt: Write Numba matrix multiplication algorithm function
@numba.jit(nopython=True)
    def matrixmultiply(A,B):
        C = np.zeros((A.shape[0],B.shape[1]))
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                for k in range(A.shape[1]):
                    C[i,j] += A[i,k] * B[k,j]
        return C

## Prompt: Write matrix multiplication algorithm function
def matrixmultiply(A,B):
        C = np.zeros((A.shape[0],B.shape[1]))
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                for k in range(A.shape[1]):
                    C[i,j] += A[i,k] * B[k,j]
        return C


# cuPY GEMM implementation

import numpy as np

import cupy as cp

import time

# call cuBLAS GEMM function

A = cp.array(np.random.rand(N, N), dtype=np.float64)

B = cp.array(np.random.rand(N, N), dtype=np.float64)

C = cp.zeros((N, N), dtype=np.float64)

start = time.time()

cp.linalg.matmul(A, B, out=C)

end = time.time()

print('cuPY GEMM time:', end - start)


# Prompt: gemm Numba
Synthesizing 7/10 solutions (Duplicates hidden)

=======

# gemm Numba implementation

=======

# gemm Numba version

=======

# gemm Numba version

import numpy as np

from numba import cuda

import time

import math

=======

# gemm Numba CUDA Python

import numpy as np

from numba import cuda

import time

=======

# gemm Numba CUDA kernel

@cuda.jit

=======

# gemm Numba CUDA Kernel

@cuda.jit


# Prompt: gemm numpy
Synthesizing 9/10 solutions

=======

# gemm numpy implementation

import numpy as np

import time

N = 1024

A = np.random.rand(N, N)

B = np.random.rand(N, N)

C = np.zeros((N, N))

start = time.time()

C = np.matmul(A, B)

end = time.time()

print('numpy GEMM time:', end - start)

=======

# gemm numpy version

import numpy as np

import time

N = 1024

A = np.random.rand(N, N)

B = np.random.rand(N, N)

C = np.zeros((N, N))

start = time.time()

C = np.matmul(A, B)

end = time.time()

print('numpy GEMM time:', end - start)

=======

# gemm numpy version

import numpy as np

import time

start = time.time()

A = np.random.rand(N, N)

B = np.random.rand(N, N)

C = np.dot(A, B)

end = time.time()

print('gemm numpy time:', end - start)

=======

# gemm numpy version

import numpy as np

import time

start = time.time()

A = np.random.rand(N, N)

B = np.random.rand(N, N)

C = np.zeros((N, N))

C = np.matmul(A, B)

end = time.time()

print('Numpy GEMM time:', end - start)

=======

# gemm numpy implementation

import numpy as np

import time

start = time.time()

N = 4096

A = np.random.rand(N, N)

B = np.random.rand(N, N)

C = np.zeros((N, N))

for i in range(N):

for j in range(N):

for k in range(N):

C[i][j] += A[i][k] * B[k][j]

end = time.time()

print('numpy GEMM time:', end - start)

=======

# gemm numpy implementation

import numpy as np

import time

=======

# gemm numpy version

import numpy as np

import time

=======

# gemm numpy CUDA Python

import numpy as np

from numba import cuda

import time

import math

=======

# gemm numpy implementation

import numpy as np

import time

import math


# Prompt: matrix multiplication cuPy

Synthesizing 10/10 solutions (Duplicates hidden)

=======

# matrix multiplication cuPy vs numpy

import cupy as cp

import numpy as np

import time

=======

# matrix multiplication cuPy vs numpy

import cupy as cp

import numpy as np

import time

import matplotlib.pyplot as plt

import sys

=======

# matrix multiplication cuPy vs numpy

import time

import cupy as cp

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

=======

# matrix multiplication cuPy vs. NumPy

import cupy as cp

import numpy as np

import time

import sys

import csv

import os

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import matplotlib.lines as mlines

import matplotlib

matplotlib.use('Agg')

from matplotlib import style

style.use('ggplot')

=======

# matrix multiplication cuPy vs. Numpy

import cupy as cp

import numpy as np

import time

import sys

import os

import matplotlib.pyplot as plt

import pandas as pd

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--size', type=int, default=1000,

help='size of the matrix')

parser.add_argument('--n', type=int, default=100,

help='number of times to run the test')

parser.add_argument('--csv', type=str, default=None,

help='csv file to save the results')

parser.add_argument('--plot', type=str, default=None,

help='png file to save the plot')

args = parser.parse_args()

=======

# matrix multiplication cuPy vs numpy

import cupy as cp

import numpy as np

import time

import matplotlib.pyplot as plt

import pandas as pd

from pandas import DataFrame

import os

import sys

import csv

import seaborn as sns

import math

import datetime

import argparse

import json

=======

# matrix multiplication cuPy vs. NumPy
# matrix multiplication using cuPy
# matrix multiplication using NumPy
# matrix multiplication using cuBLAS

import cupy as cp
import numpy as np
import time

=======

# matrix multiplication cuPy vs NumPy

=======

# matrix multiplication cuPy vs numpy


# gemm pyCUDA
def
Synthesizing 7/10 solutions

=======

def gemm_pyCUDA(A, B, C, alpha, beta, m, n, k, lda, ldb, ldc):
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    C = C.astype(np.float32)
    A_gpu = gpuarray.to_gpu(A)
    B_gpu = gpuarray.to_gpu(B)
    C_gpu = gpuarray.to_gpu(C)
    func = mod.get_function("gemm")
    func(A_gpu, B_gpu, C_gpu, np.float32(alpha), np.float32(beta), np.int32(m), np.int32(n), np.int32(k), np.int32(lda), np.int32(ldb), np.int32(ldc), block=(16, 16, 1), grid=(1, 1))
    return C_gpu.get()

=======

def gemmPyCUDA(M, N, K, alpha, A, B, beta, C):
    # Allocate device memory
    A_d = cuda.mem_alloc(A.nbytes)
    B_d = cuda.mem_alloc(B.nbytes)
    C_d = cuda.mem_alloc(C.nbytes)
    
    # Copy to device memory
    cuda.memcpy_htod(A_d, A)
    cuda.memcpy_htod(B_d, B)
    cuda.memcpy_htod(C_d, C)
    
    # Create a kernel
    mod = SourceModule("""
    #define M %(M)d
    #define N %(N)d
    #define K %(K)d
    #define ALPHA %(alpha)f
    #define BETA %(beta)f
    
    __global__ void gemm(float *A, float *B, float *C)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        float sum = 0;
        for (int k = 0; k < K; k++)
        {
            sum += ALPHA * A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = BETA * C[i * N + j] + sum;
    }
    """ % {
        'M': M,
        'N': N,
        'K': K,
        'alpha': alpha,
        'beta': beta,
    })
    
    # Get the kernel function
    gemm = mod.get_function("gemm")
    
    # Set the kernel dimensions
    threadsPerBlock = (16, 16)
    blocksPerGrid_x = int(math.ceil(M / threadsPerBlock[0]))
    blocksPerGrid_y = int(math.ceil(N / threadsPerBlock[1]))
    blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)
    
    # Call the kernel
    gemm(A_d, B_d, C_d, block=threadsPerBlock, grid=blocksPerGrid)
    
    # Copy the results
    cuda.memcpy_dtoh(C, C_d)
    
    # Free the device memory
    A_d.free()
    B_d.free()
    C_d.free()
    
    # Return the result
    return C

=======

def gemmPyCUDA(A, B, C, alpha, beta):
    n = A.shape[0]
    m = B.shape[1]
    k = A.shape[1]
    A_gpu = cuda.mem_alloc(A.nbytes)
    B_gpu = cuda.mem_alloc(B.nbytes)
    C_gpu = cuda.mem_alloc(C.nbytes)
    cuda.memcpy_htod(A_gpu, A)
    cuda.memcpy_htod(B_gpu, B)
    cuda.memcpy_htod(C_gpu, C)
    mod = SourceModule("""
    __global__ void gemm(float *A, float *B, float *C, float alpha, float beta, int n, int m, int k)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i < n && j < m)
        {
            float temp = 0;
            for (int p = 0; p < k; p++)
            {
                temp += A[i * k + p] * B[p * m + j];
            }
            C[i * m + j] = alpha * temp + beta * C[i * m + j];
        }
    }
    """)
    func = mod.get_function("gemm")
    func(A_gpu, B_gpu, C_gpu, np.float32(alpha), np.float32(beta), np.int32(n), np.int32(m), np.int32(k), block=(16, 16, 1), grid=(int(np.ceil(n / 16)), int(np.ceil(m / 16)), 1))
    C = np.empty_like(C)
    cuda.memcpy_dtoh(C, C_gpu)
    return C

=======

def gemmPyCuda(A, B, C, alpha, beta):
    # A: MxK, B: KxN, C: MxN
    # C = alpha * A * B + beta * C

    # init
    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]

    # A, B, C
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    C = C.astype(np.float32)

    # allocate memory
    A_gpu = gpuarray.empty((M, K), np.float32)
    B_gpu = gpuarray.empty((K, N), np.float32)
    C_gpu = gpuarray.empty((M, N), np.float32)

    # copy
    A_gpu.set(A)
    B_gpu.set(B)
    C_gpu.set(C)

    # kernel
    kernel = '''
    __global__ void gemm(float *A, float *B, float *C, float alpha, float beta, int M, int K, int N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k;

        if (i < M && j < N) {
            float temp = 0.0;
            for (k = 0; k < K; k++) {
                temp += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * temp + beta * C[i * N + j];
        }
    }
    '''

    # compile
    mod = SourceModule(kernel)

    # prepare
    gemm = mod.get_function("gemm")
    block = (32, 32, 1)
    grid = (int((M + block[0] - 1) / block[0]), int((N + block[1] - 1) / block[1]), 1)

    # call
    gemm(A_gpu, B_gpu, C_gpu, np.float32(alpha), np.float32(beta), np.int32(M), np.int32(K), np.int32(N), block=block, grid=grid)

    # copy
    C = C_gpu.get()

    # return
    return C

=======

def gemm(A, B, C, alpha, beta, m, n, k, lda, ldb, ldc):

    # block size
    block = (16, 16, 1)

    # grid size
    grid = (int(math.ceil(float(m)/block[0])), int(math.ceil(float(n)/block[1])), 1)

    # kernel
    kernel = mod.get_function("gemm")

    # call kernel
    kernel(A, B, C, np.float32(alpha), np.float32(beta), np.int32(m), np.int32(n), np.int32(k), np.int32(lda), np.int32(ldb), np.int32(ldc), block=block, grid=grid)

=======

def gemm(A, B, C, alpha, beta, transa, transb, m, n, k):

    # transa = 'N'
    # transb = 'N'
    # m = 3
    # n = 3
    # k = 3
    # alpha = 1.0
    # beta = 1.0
    # A = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.float32)
    # B = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.float32)
    # C = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.float32)

    # print(A)
    # print(B)
    # print(C)

    # print(A.shape)
    # print(B.shape)
    # print(C.shape)

    # print(A.dtype)
    # print(B.dtype)
    # print(C.dtype)

    # print(A.strides)
    # print(B.strides)
    # print(C.strides)

    # print(A.ctypes.data)
    # print(B.ctypes.data)
    # print(C.ctypes.data)

    # print(A.flags)
    # print(B.flags)
    # print(C.flags)

    # print('----------------')

    # print(A.nbytes)
    # print(B.nbytes)
    # print(C.nbytes)

    # print(A.itemsize)
    # print(B.itemsize)
    # print(C.itemsize)

    # print(A.size)
    # print(B.size)
    # print(C.size)

    # print('----------------')

    # print(A.nbytes/A.itemsize)
    # print(B.nbytes/B.itemsize)
    # print(C.nbytes/C.itemsize)

    # print(A.size)
    # print(B.size)
    # print(C.size)

    # print('----------------')

    # print(A.nbytes/A.itemsize/A.size)
    # print(B.nbytes/B.itemsize/B.size)
    # print(C.nbytes/C.itemsize/C.size)

    # print(A.nbytes/A.itemsize/A.size)
    # print(B.nbytes/B.itemsize/B.size)
    # print(C.nbytes/C.itemsize/C.size)

    # print('----------------')

    # print(A.nbytes/A.itemsize/A.size)

=======

def gemm_pyCUDA(A, B, C, alpha, beta):
    # A, B, C are numpy arrays
    # alpha, beta are scalars

    # A is M x K
    # B is K x N
    # C is M x N

    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]

    # Initialize device
    dev = cuda.Device(0)
    ctx = dev.make_context()
    stream = cuda.Stream()

    # Allocate memory on device
    A_d = cuda.mem_alloc(A.nbytes)
    B_d = cuda.mem_alloc(B.nbytes)
    C_d = cuda.mem_alloc(C.nbytes)

    # Copy data from host to device
    cuda.memcpy_htod(A_d, A)
    cuda.memcpy_htod(B_d, B)
    cuda.memcpy_htod(C_d, C)

    # Prepare kernel
    mod = SourceModule("""
        __global__ void gemm(float *A, float *B, float *C, float alpha, float beta, int M, int K, int N) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;

            if (i < M && j < N) {
                float sum = 0.0;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = alpha * sum + beta * C[i * N + j];
            }
        }
    """)
    func = mod.get_function("gemm")

    # Call kernel
    func(A_d, B_d, C_d, np.float32(alpha), np.float32(beta), np.int32(M), np.int32(K), np.int32(N), block=(16, 16, 1), grid=(int(np.ceil(M/16)), int(np.ceil(N/16))), stream=stream)

    # Copy data from device to host
    cuda.memcpy_dtoh(C, C_d)

    # Free memory
    A_d.free()
    B_d.free()
    C_d.free()

    # Destroy context
    ctx.pop()

    return C

# Prompt: gemm cuPy
def
Synthesizing 10/10 solutions (Duplicates hidden)

=======

def gemm_cupy(a, b, c, alpha, beta):
    return alpha * cupy.dot(a, b) + beta * c

=======

def gemm_cuPy(A, B, C, alpha, beta, N, M, K, dtype):
    A = cp.array(A, dtype=dtype)
    B = cp.array(B, dtype=dtype)
    C = cp.array(C, dtype=dtype)
    C = alpha * cp.dot(A, B) + beta * C
    return C

=======

def gemm_cupy(a, b, c, alpha, beta):

=======

def gemm_cupy(A, B, C):
    return cp.dot(A, B) + C

=======

def gemm_cupy(A, B, C):
    C = cp.dot(A, B)
    return C

=======

def run_gemm_cupy(M, N, K, alpha, beta, A, B, C):
    # A = cupy.random.random((M, K))
    # B = cupy.random.random((K, N))
    # C = cupy.random.random((M, N))
    C = cupy.dot(A, B, out=C)
    C *= beta
    C += alpha * cupy.dot(A, B, out=C)
    return C

=======

def gemm_cupy(N, M, K, alpha, beta):

    # Initialize cuPy arrays
    A = cp.random.random((N, K)).astype(np.float32)
    B = cp.random.random((K, M)).astype(np.float32)
    C = cp.random.random((N, M)).astype(np.float32)

    # Start timer
    start = time.time()

    # Compute cuPy GEMM
    C = alpha * cp.dot(A, B) + beta * C

    # Stop timer
    end = time.time()

    # Return time
    return end - start

=======

def gemm_cupy(a, b, c, alpha, beta, N, M, K, repeat):
    for i in range(repeat):
        cupy.matmul(a, b, out=c)
    return c

=======

def gemm_cuPy(A, B, C, alpha, beta, m, n, k):
    #print("gemm_cuPy")
    #print("gemm_cuPy: A.shape = ", A.shape)
    #print("gemm_cuPy: B.shape = ", B.shape)
    #print("gemm_cuPy: C.shape = ", C.shape)
    #print("gemm_cuPy: alpha = ", alpha)
    #print("gemm_cuPy: beta = ", beta)
    #print("gemm_cuPy: m = ", m)
    #print("gemm_cuPy: n = ", n)
    #print("gemm_cuPy: k = ", k)
    #print("gemm_cuPy: A = ", A)
    #print("gemm_cuPy: B = ", B)
    #print("gemm_cuPy: C = ", C)
    #print("gemm_cuPy: A = ", A)
    #print("gemm_cuPy: B = ", B)
    #print("gemm_cuPy: C = ", C)
    #print("gemm_cuPy: A = ", A)
    #print("gemm_cuPy: B = ", B)
    #print("gemm_cuPy: C = ", C)
    #print("gemm_cuPy: A = ", A)
    #print("gemm_cuPy: B = ", B)
    #print("gemm_cuPy: C = ", C)
    #print("gemm_cuPy: A = ", A)
    #print("gemm_cuPy: B = ", B)
    #print("gemm_cuPy: C = ", C)
    #print("gemm_cuPy: A = ", A)
    #print("gemm_cuPy: B = ", B)
    #print("gemm_cuPy: C = ", C)
    #print("gemm_cuPy: A = ", A)
    #print("gemm_cuPy: B = ", B)
    #print("gemm_cuPy: C = ", C)
    #print("gemm_cuPy: A = ", A)
    #print("gemm_cuPy: B = ", B)
    #print("gemm_cuPy: C = ", C)
    #print("gemm_cuPy: A = ", A)
    #


# gemm numpy\ndef
Synthesizing 10/10 solutions (Duplicates hidden)

=======

def gemm_numpy(A, B, C, alpha, beta):
    C = alpha * np.dot(A, B) + beta * C
    return C

=======

def gemm_numpy(A, B, C, alpha, beta):
    C[:] = alpha * np.dot(A, B) + beta * C

=======

def gemm_numpy(A, B, C, alpha, beta):
    C = alpha * np.dot(A, B) + beta * C

=======

def gemm_numpy(A, B, C, alpha, beta):
    C[:] = alpha * np.dot(A, B) + beta * C
    return C

=======

def gemm_numpy(M, N, K, alpha, A, B, beta, C):
    C = alpha * np.dot(A, B) + beta * C
    return C

=======

def gemm_numpy(A, B, C, alpha=1, beta=1):
    return alpha * np.dot(A, B) + beta * C

=======

def gemm_numpy(A, B, C):
    return np.dot(A, np.dot(B, C))

=======

def gemm_numpy(A, B, C):
    C = np.dot(A, B)
    return C

=======

def gemm_np(A, B, C):
    return np.dot(A, B) + C
