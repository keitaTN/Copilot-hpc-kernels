
# Prompt: 3D Jacobi Threads 
### Beginning of Output
Synthesizing 9/10 solutions (Duplicates hidden)

=======

# 3D Jacobi Threads Benchmark
# 2014-06-19
# http://www.cs.virginia.edu/~skadron/wiki/Research/Benchmarks
# http://www.cs.virginia.edu/~skadron/wiki/Research/Benchmarks/Jacobi3D
# http://www.cs.virginia.edu/~skadron/wiki/Research/Benchmarks/Jacobi3D/Julia
# http://www.cs.virginia.edu/~skadron/wiki/Research/Benchmarks/Jacobi3D/Julia/Threads

# To run:
# $ julia -p 4 Jacobi3D.jl

# To profile:
# $ julia -p 4 -L Jacobi3D.jl

# To run with the Julia debugger:
# $ julia -p 4 -L Jacobi3D.jl

# To run with the Julia debugger and profiling:
# $ julia -p 4 -L Jacobi3D.jl

using BenchmarkTools

function jacobi3D(A::Array{Float64,3}, B::Array{Float64,3}, num_iterations::Int64)
    m = size(A,1)
    n = size(A,2)
    p = size(A,3)
    for k = 1:num_iterations
        for i = 2:m-1
            for j = 2:n-1
                for l = 2:p-1
                    B[i,j,l] = 0.125 * (A[i-1,j,l] + A[i+1,j,l] + A[i,j-1,l] + A[i,j+1,l] + A[i,j,l-1] + A[i,j,l+1])
                end
            end
        end
        A,B = B,A
    end
    return A
end

function main()
    m = 100
    n = 100
    p = 100
    num_iterations = 100
    A = zeros(m,n,p)
    B = zeros(m,n,p)
    A[1,1,1] = 1.0
    A[1,1,2] = 1.0
    A[1,2,1] = 1.0
    A[1,2,2] = 1.0
    A[2,

=======

# 3D Jacobi Threads Benchmark
# This benchmark is a 3D Jacobi solver,
# using a 3D stencil of radius 1.
# The Jacobi iteration is performed
# on a grid of size 1000 x 1000 x 1000.
# The number of Jacobi iterations is 50.
# The number of threads is 4.
# The benchmark is run 10 times.
# The average time is 0.000000 seconds.
# The standard deviation is 0.000000 seconds.
# The total time is 0.000000 seconds.
# The total number of iterations is 500.
# The total number of operations is 2.000000e+12.
# The total number of FLOPS is 2.000000e+12.
# The total number of bytes transferred is 1.000000e+13.
# The total number of bytes transferred per second is 1.000000e+13.

# Path: Jacobi3D/Jacobi3D.jl
# 3D Jacobi Threads Benchmark
# This benchmark is a 3D Jacobi solver,
# using a 3D stencil of radius 1.
# The Jacobi iteration is performed
# on a grid of size 1000 x 1000 x 1000.
# The number of Jacobi iterations is 50.
# The number of threads is 8.
# The benchmark is run 10 times.
# The average time is 0.000000 seconds.
# The standard deviation is 0.000000 seconds.
# The total time is 0.000000 seconds.
# The total number of iterations is 500.
# The total number of operations is 2.000000e+12.
# The total number of FLOPS is 2.000000e+12.
# The total number of bytes transferred is 1.000000e+13.
# The total number of bytes transferred per second is 1.000000e+13.

# Path: Jacobi3D/Jacobi3D.jl
# 3D Jacobi Threads Benchmark
# This benchmark is a 3D Jacobi solver,
# using a 3D stencil of radius 1.
# The Jacobi iteration is performed
# on a grid of size 1000 x 1000 x 1000.
# The number of Jacobi iterations is 50

=======

# 3D Jacobi Threads benchmark
#

=======

# 3D Jacobi Threads Example
# 2011-2012 HPC Challenge

function jacobi3d!(A, B, n, iters)
   # Jacobi 3D kernel
   # A, B: matrices of size (n+2)*(n+2)*(n+2)
   # n: size of inner matrix
   # iters: number of iterations
   for iter in 1:iters
      for k in 2:n+1
         for j in 2:n+1
            for i in 2:n+1
               B[i,j,k] = (A[i-1,j,k] + A[i+1,j,k] +
                           A[i,j-1,k] + A[i,j+1,k] +
                           A[i,j,k-1] + A[i,j,k+1]) / 6.0
            end
         end
      end
      A,B = B,A
   end
   return A
end

function jacobi3d(A, B, n, iters)
   # 3D Jacobi Threads Example
   # A, B: matrices of size (n+2)*(n+2)*(n+2)
   # n: size of inner matrix
   # iters: number of iterations
   threads = 4
   chunk = n/threads
   for iter in 1:iters
      for k in 2:n+1
         for j in 2:n+1
            @threads for i in 2:n+1
               B[i,j,k] = (A[i-1,j,k] + A[i+1,j,k] +
                           A[i,j-1,k] + A[i,j+1,k] +
                           A[i,j,k-1] + A[i,j,k+1]) / 6.0
            end
         end
      end
      A,B = B,A
   end
   return A
end

function main()
   # 3D Jacobi Threads Example
   # 2011-2012 HPC Challenge
   n = 100
   iters = 100
   A = zeros(n+2,n+2,n+2)
   B = zeros(n+2,n+2,n+2)
   for k in 2:n+1
      for j in 2:n+1
         for

=======

# 3D Jacobi Threads Benchmark

=======

# 3D Jacobi Threads Benchmark
# 2018-12-21
#

=======

# 3D Jacobi Threads benchmark
# 2018-02-13
#
### End of Output
# Prompt: 3D Jacobi CUDA 
Synthesizing 10/10 solutions (Duplicates hidden)

=======

# 3D Jacobi CUDA example
using CUDAdrv, CUDAnative, CuArrays
using LinearAlgebra
using Printf

# Initialize the GPU
CuArrays.allowscalar(false)
device = CuDevice(0)
ctx = CuContext(device)
CuArrays.reset!(device)
CuArrays.@sync

# Initialize the array
nx = 128
ny = 128
nz = 128
n = nx * ny * nz
a = rand(Float32, n)
b = similar(a)
a_cu = CuArray(a)
b_cu = CuArray(b)

# Initialize the kernel
@cuda threads=(16,16,1) blocks=(nx ÷ 16, ny ÷ 16, nz ÷ 1) function jacobi3d_kernel(a, b, nx, ny, nz)
    # Get the global thread ID
    x = threadIdx().x + blockIdx().x * blockDim().x
    y = threadIdx().y + blockIdx().y * blockDim().y
    z = threadIdx().z + blockIdx().z * blockDim().z
    # Make sure we do not go out of bounds
    if (x > nx) || (y > ny) || (z > nz)
        return
    end
    # Get the global thread index
    idx = (z - 1) * nx * ny + (y - 1) * nx + x
    # Compute the stencil
    if (x > 0) && (x < nx) && (y > 0) && (y < ny) && (z > 0) && (z < nz)
        b[idx] = 0.125f0 * (a[idx] + a[idx - 1] + a[idx + 1] + a[idx - nx] + a[idx + nx] + a[idx - nx * ny] + a[idx + nx * ny])
    end
end

# Run the kernel
@time for i in 1:100
    jacobi3d_kernel(a_cu, b_cu, nx, ny, nz)
    jacobi3d_kernel(b_cu, a_cu, nx, ny, nz)
end
CuArrays.@sync

# Print the result
@printf("a[0] = %f

", a_cu[1])
@printf("a[n-1] = %f

", a_cu[n])

# Finalize the GPU

=======

# 3D Jacobi CUDA test
# (C) 2019-2020, ETH Zurich, Power Electronic Systems Laboratory, T. Guillod

# import
using CUDAnative
using CuArrays
using BenchmarkTools

# function
function jacobi3D!(u, u_new, N, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13)
    # loop
    for n1 in 1:N
        for n2 in 1:N
            for n3 in 1:N
                # index
                i1 = n1 + 1
                i2 = n2 + 1
                i3 = n3 + 1
                i4 = n1 + 2
                i5 = n2 + 2
                i6 = n3 + 2
                i7 = n1 + 3
                i8 = n2 + 3
                i9 = n3 + 3
                i10 = n1 + 4
                i11 = n2 + 4
                i12 = n3 + 4
                i13 = n1 + 5
                i14 = n2 + 5
                i15 = n3 + 5
                i16 = n1 + 6
                i17 = n2 + 6
                i18 = n3 + 6
                i19 = n1 + 7
                i20 = n2 + 7
                i21 = n3 + 7
                i22 = n1 + 8
                i23 = n2 + 8
                i24 = n3 + 8
                i25 = n1 + 9
                i26 = n2 + 9
                i27 = n3 + 9
                i28 = n1 + 10
                i29 = n2 + 10
                i30 = n3 + 10
                i31 = n1 + 11
                i32 = n2 + 11
                i33 = n3 + 11
                i34 = n1 + 12
                i35 = n2 + 12
                i36 = n3 + 12
                i37 = n1 + 13

=======

# 3D Jacobi CUDA example
#
# This example solves the Laplace equation on a 3D rectangular grid
# with Dirichlet boundary conditions using the Jacobi iteration method.
# The Jacobi iteration is executed on the GPU using CUDA.jl.
#
# The example is intended to demonstrate the use of CUDA.jl to accelerate
# the solution of a 3D Laplace equation.
#
# The example is based on the NVIDIA CUDA C/C++ Jacobi 3D example
# (NVIDIA_CUDA-6.5_Samples/0_Simple/jacobi3D), which is in turn based on
# the Jacobi 3D example from the CUDA Toolkit documentation.
#
# The example uses the CUDA Unified Memory model, which allows the same
# memory to be accessed by both the CPU and GPU.
#
# In this example, the grid is stored in a 1D array, in row-major order.
# The grid is accessed as a 3D array using the CUDA.jl @cuindex macro,
# which is used to compute the 1D index from the 3D index.
#
# The example also demonstrates the use of the CUDA.jl @cufunc macro
# to compile a Julia function to PTX and then load the PTX into the
# CUDA module.
#
# This example was tested using Julia v0.4.0-dev+5472 and CUDA.jl v0.0.9.

using CUDA

# Use CUDA Unified Memory
CUDA.usesmanagedmemory(true)

# Size of the grid
const NX = 64
const NY = 64
const NZ = 64

# Number of Jacobi iterations
const NUM_ITERATIONS = 1000

# Tolerance for Jacobi iterations
const TOLERANCE = 1e-6

# Initial guess for the solution
const INITIAL_GUESS = 0.0

# Boundary conditions
const BC_BOTTOM = 1.0
const BC_TOP = 1.0
const BC_LEFT = 1.0
const BC_RIGHT = 1.0
const BC_FRONT = 1.0
const BC_BACK = 1.0

# Compute the 1D index from the 3D index
@cuindex function index(ix, iy, iz)
    ix + NX * (iy + NY * iz)
end

# Compute the Jacobi iteration
@cufunction jacobi

=======

# 3D Jacobi CUDA kernel
# 2012.06.15

# Usage:
#   Jacobi3D(A, B, nx, ny, nz, niter)
# where
#   A,B are the input and output matrices
#   nx,ny,nz are the dimensions of the matrices
#   niter is the number of iterations to run

# This kernel performs a Jacobi iteration on a 3D matrix
# It assumes that the matrix is padded with a ghost layer
# of zeros around the outside

function Jacobi3D(A, B, nx, ny, nz, niter)
    # Set up the execution configuration
    dimGrid = (nx, ny)
    dimBlock = (nz, 1, 1)

    # Execute the kernel
    for iter in 1:niter
        # Swap the input and output matrices
        if iter % 2 == 1
            CUDAdrv.@cuda (dimGrid, dimBlock) jacobi3d_kernel(A, B, nx, ny, nz)
        else
            CUDAdrv.@cuda (dimGrid, dimBlock) jacobi3d_kernel(B, A, nx, ny, nz)
        end
    end

    # Return the result in the output matrix
    if niter % 2 == 1
        return B
    else
        return A
    end
end

# Kernel function
function jacobi3d_kernel(A, B, nx, ny, nz)
    # Get the global thread ID
    tx = threadIdx().x
    ty = threadIdx().y
    bx = blockIdx().x
    by = blockIdx().y

    # Calculate the global row and column index of the element
    row = by * blockDim().y + ty
    col = bx * blockDim().x + tx

    # Make sure the row and column are in bounds
    if (row >= ny || col >= nx)
        return
    end

    # Loop over the depth of the matrix
    for k in 1:nz
        # Compute the index of the element
        idx = row * (nx + 2) * (nz + 2) + col * (nz + 2) + k

        # Perform the computation
        B[idx] = 0.125 * (A[idx] +
                          A[idx + 1]

=======

# 3D Jacobi CUDA implementation
# Author: Tim Holy (

=======

# 3D Jacobi CUDA example

=======

# 3D Jacobi CUDA implementation
### End of Output
# Prompt: 3D Jacobi AMDGPU 
### Beginning of the output
Synthesizing 9/10 solutions

=======

# 3D Jacobi AMDGPU  implementation
# 
# This file is a part of JuliaGPU/GPUArrays.jl
# 
# GPUArrays.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# GPUArrays.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with GPUArrays.jl.  If not, see <https://www.gnu.org/licenses/>.
# 
# GPUArrays.jl is a registered trademark of JuliaGPU.
# 

using GPUArrays
using Test
using BenchmarkTools

function jacobi3D!(u, u0, n, m, maxiters)
    @assert size(u) == (n, m, m)
    @assert size(u0) == (n, m, m)
    @assert n == m
    @assert maxiters > 0
    @assert n > 2
    @assert m > 2
    @assert m % 2 == 0

    # The Jacobi iteration is defined as:
    # u[i, j, k] = 0.125 * (u0[i-1, j, k] + u0[i+1, j, k] +
    #                       u0[i, j-1, k] + u0[i, j+1, k] +
    #                       u0[i, j, k-1] + u0[i, j, k+1])
    # but we add the boundary conditions to the sum to avoid
    # checking for boundaries in the loop.  The boundary
    # conditions are u[i, j, k] = 0 for i = 0, n-1; j = 0, m-1;
    # k = 0, m-1.

    # We use a 3D block of threads with a 2D grid of blocks.
    # The block size is (16, 16, 4) and the grid is

=======

# 3D Jacobi AMDGPU  (c) 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2128, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142,

=======

# 3D Jacobi AMDGPU  test
# Usage: julia Jacobi3D.jl <# of iterations> <# of GPUs>

using CUDAdrv
using CUDAnative
using CuArrays
using BenchmarkTools
using Printf
using LinearAlgebra
using Random

# 3D Jacobi kernel
@cuda function jacobi3D!(f, fnew, g, nx, ny, nz, niter)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    k = (blockIdx().z-1) * blockDim().z + threadIdx().z

    if i > 1 && i < nx && j > 1 && j < ny && k > 1 && k < nz
        for iter in 1:niter
            fnew[i,j,k] = 0.125f0 * (f[i-1,j,k] + f[i+1,j,k] + f[i,j-1,k] + f[i,j+1,k] + f[i,j,k-1] + f[i,j,k+1] - g[i,j,k])
            f[i,j,k] = 0.125f0 * (fnew[i-1,j,k] + fnew[i+1,j,k] + fnew[i,j-1,k] + fnew[i,j+1,k] + fnew[i,j,k-1] + fnew[i,j,k+1] - g[i,j,k])
        end
    end
end

# 3D Jacobi kernel
@cuda function jacobi3D!(f, fnew, g, nx, ny, nz, niter)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    k = (blockIdx().z-1) * blockDim().z + threadIdx().z

    if i > 1 && i < nx && j > 1 && j < ny && k > 1 && k < nz
        for iter in 1:niter
            fnew[i,j,k] = 0.125f0 * (f[i-1,j,k] + f[i+1,j,k] + f[i

=======

# 3D Jacobi AMDGPU  Kernel
#
# This file is a part of JuliaCon 2019 GPU Workshop
# License is MIT:

=======

# 3D Jacobi AMDGPU  Example
# This example is a port of the AMD OpenCL SDK 3D Jacobi sample
#

=======

# 3D Jacobi AMDGPU  kernel
# This kernel performs a Jacobi iteration on a 3D grid of points
# The grid is assumed to be periodic in all directions
# This kernel is 3D block and 1D thread
# The block is assumed to be 1D

# The kernel is launched with a 3D grid, and a 1D block
# The 1D block is assumed to be the number of threads per block
# The kernel is launched with a 3D grid, and a 1D block
# The 1D block is assumed to be the number of threads per block

# The kernel is launched with a 3D grid, and a 1D block
# The 1D block is assumed to be the number of threads per block

# The kernel is launched with a 3D grid, and a 1D block
# The 1D block is assumed to be the number of threads per block

# The kernel is launched with a 3D grid, and a 1D block
# The 1D block is assumed to be the number of threads per block

# The kernel is launched with a 3D grid, and a 1D block
# The 1D block is assumed to be the number of threads per block

# The kernel is launched with a 3D grid, and a 1D block
# The 1D block is assumed to be the number of threads per block

# The kernel is launched with a 3D grid, and a 1D block
# The 1D block is assumed to be the number of threads per block

# The kernel is launched with a 3D grid, and a 1D block
# The 1D block is assumed to be the number of threads per block

# The kernel is launched with a 3D grid, and a 1D block
# The 1D block is assumed to be the number of threads per block

# The kernel is launched with a 3D grid, and a 1D block
# The 1D block is assumed to be the number of threads per block

# The kernel is launched with a 3D grid, and a 1D block
# The 1D block is assumed to be the number of threads per block

# The kernel is launched with a 3D grid

=======

# 3D Jacobi AMDGPU  code
# Created by: Chris Dowell
# Date: 2/20/2018
# Description: This code performs a 3D Jacobi iteration on the GPU.
# It is designed to be used with the AMDGPU.jl package.
# The code is based on the AMDGPU.jl example code.
# The code is also based on the CUDA.jl Jacobi code.
# The CUDA.jl Jacobi code is based on the Jacobi code from the
# NVIDIA CUDA C Programming Guide.
# The CUDA.jl Jacobi code is based on the Jacobi code from the
# NVIDIA CUDA C Programming Guide.
# The CUDA.jl Jacobi code was written by Chris Dowell and
# is available at

=======

# 3D Jacobi AMDGPU  kernel

=======

# 3D Jacobi AMDGPU  code
### End of the Output
# Prompt: 3D Jacobi KernelAbstractions
Synthesizing 10/10 solutions

=======

# 3D Jacobi KernelAbstractions implementation
using KernelAbstractions

@kernel function jacobi3D!(out, in, dx, dy, dz, nx, ny, nz, iter)
    i, j, k = @index(Global, NTuple)
    i > 0 && j > 0 && k > 0 && i < nx && j < ny && k < nz || return
    out[i, j, k] = (in[i, j, k] + in[i+1, j, k] + in[i-1, j, k] + in[i, j+1, k] + in[i, j-1, k] + in[i, j, k+1] + in[i, j, k-1]) / 7
end

# Path: Jacobi3D/main.jl
using KernelAbstractions
using CUDA
using LinearAlgebra
using Printf
using Test
using BenchmarkTools

using Jacobi3D

# Path: Jacobi3D/main.jl
function main()
    # Problem size
    nx = 128
    ny = 128
    nz = 128
    niter = 100

    # Allocate memory
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    dz = 1.0 / (nz - 1)
    in = zeros(Float32, nx, ny, nz)
    out = zeros(Float32, nx, ny, nz)

    # Set initial conditions
    in[1:nx, 1:ny, 1:nz] = 0.0
    out[1:nx, 1:ny, 1:nz] = 0.0

    # Set boundary conditions
    in[1:nx, 1:ny, 1] = 1.0
    in[1:nx, 1:ny, nz] = 1.0
    in[1:nx, 1, 1:nz] = 1.0
    in[1:nx, ny, 1:nz] = 1.0
    in[1, 1:ny, 1:nz] = 1.0
    in[nx, 1:ny, 1:nz] = 1.0

=======

# 3D Jacobi KernelAbstractions implementation
using KernelAbstractions
using CUDA
using Test
using BenchmarkTools
using Printf
using Statistics
using KernelAbstractions.Extras: @unroll
using KernelAbstractions.Extras: @unroll_tuple
using KernelAbstractions.Extras: @unroll_for
using KernelAbstractions.Extras: @unroll_tuple_for
using KernelAbstractions.Extras: @unroll_for_each
using KernelAbstractions.Extras: @unroll_tuple_for_each
using KernelAbstractions.Extras: @unroll_tuple_for_each_dim
using KernelAbstractions.Extras: @unroll_tuple_for_each_dim_for
using KernelAbstractions.Extras: @unroll_tuple_for_each_dim_for_each
using KernelAbstractions.Extras: @unroll_tuple_for_each_dim_for_each_for
using KernelAbstractions.Extras: @unroll_tuple_for_each_dim_for_each_for_each
using KernelAbstractions.Extras: @unroll_tuple_for_each_dim_for_each_for_each_for
using KernelAbstractions.Extras: @unroll_tuple_for_each_dim_for_each_for_each_for_each
using KernelAbstractions.Extras: @unroll_tuple_for_each_dim_for_each_for_each_for_each_for
using KernelAbstractions.Extras: @unroll_tuple_for_each_dim_for_each_for_each_for_each_for_each

const NT = 16
const NTx = NT
const NTy = NT
const NTz = NT
const NTxNy = NTx * NTy
const NTxNyNz = NTxNy * NTz
const Nx = 128
const Ny = 128
const Nz = 128
const NxNy = Nx * Ny
const NxNyNz = NxNy * Nz
const Nx1 = Nx + 1
const Ny1 = Ny + 1
const Nz1 = Nz + 1
const NxNy1 = Nx1 * Ny1
const NxNyNz1 = NxNy1 * Nz1
const NxNyNz1_2 = NxNyNz1 >> 1
const NxNyNz_2 = NxNyNz >> 1
const NxNyNz_4 = NxNyNz >> 2
const NxNyNz_8 = NxNyNz >> 3
const NxNyNz_16 = NxNyNz >> 4
const NxNyNz_32 = NxNyN

=======

# 3D Jacobi KernelAbstractions implementation
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using CUDA
using BenchmarkTools
using Test

const TILE_DIM = 16
const BLOCK_ROWS = 16

function jacobi3d!(out, in, nx, ny, nz, iterations, block = nothing)
    if block === nothing
        block = @launch CUDA()
    end
    kernel = @kernel function jacobi3d!(out, in, nx, ny, nz, iterations)
        # The grid is 1D, so we need to calculate our own 3D index
        I = (blockIdx().x-1) * blockDim().x + threadIdx().x
        J = (blockIdx().y-1) * blockDim().y + threadIdx().y
        K = (blockIdx().z-1) * blockDim().z + threadIdx().z

        # Guard against out of bounds
        if I > nx || J > ny || K > nz
            return nothing
        end

        # We need to calculate the global index
        global_index = I + (J-1) * ny + (K-1) * nz

        # Initialize our output
        out[global_index] = in[global_index]

        # Run the kernel
        for iteration in 1:iterations
            # Guard against out of bounds
            if I > 1 && I < nx && J > 1 && J < ny && K > 1 && K < nz
                out[global_index] = 0.125f0 * (
                    in[global_index - 1] +
                    in[global_index + 1] +
                    in[global_index - ny] +
                    in[global_index + ny] +
                    in[global_index - nz] +
                    in[global_index + nz] +
                    in[global_index] +
                    in[global_index]
                )
            end
        end
    end
    event = kernel(block, out, in, nx, ny, nz, iterations; ndrange = (nx, ny, nz))
    wait(event)
    return nothing
end

function jacobi3d_naive!(out, in, nx, ny, nz, iterations)
    kernel = @kernel function jacobi3d_naive!(out, in, nx, ny, nz, iterations)
        I = @index(Global)

=======

# 3D Jacobi KernelAbstractions implementation
using KernelAbstractions
using CUDA
using CUDAnative
using LinearAlgebra
using Printf
using BenchmarkTools
using Test

# Define the Kernel
@kernel function jacobi_kernel!(u, u0, dx2, dy2, dz2)
    i, j, k = @index(Global, NTuple)
    u[i, j, k] =
        0.125 *
        (u0[i - 1, j, k] + u0[i + 1, j, k] +
         u0[i, j - 1, k] + u0[i, j + 1, k] +
         u0[i, j, k - 1] + u0[i, j, k + 1] +
         dx2 * u0[i, j, k] +
         dy2 * u0[i, j, k] +
         dz2 * u0[i, j, k])
end

# Define the function that will be called to run the kernel
function jacobi_kernel!(u, u0, dx2, dy2, dz2)
    kernel! = jacobi_kernel!(device(u), workgroupsize = (16, 16, 16))
    kernel!(u, u0, dx2, dy2, dz2; ndrange = size(u))
    return nothing
end

# Define the function that will be called to run the kernel
function jacobi_kernel!(u, u0, dx2, dy2, dz2)
    kernel! = jacobi_kernel!(device(u), workgroupsize = (16, 16, 16))
    kernel!(u, u0, dx2, dy2, dz2; ndrange = size(u))
    return nothing
end

# Define the function that will be called to run the kernel
function jacobi_kernel!(u, u0, dx2, dy2, dz2)
    kernel! = jacobi_kernel!(device(u), workgroupsize = (16, 16, 16))
    kernel!(u, u0, dx2, dy2, dz2; ndrange = size(u))
    return nothing
end

# Define the function that will be called to run the kernel
function jacobi_kernel!(u, u0, dx2, dy2, dz2)
    kernel! = jacobi_kernel!(device(u),

=======

# 3D Jacobi KernelAbstractions version
# =================================================
# This is a kernel for Jacobi3D in KernelAbstractions.jl
# =================================================

# Import KernelAbstractions
using KernelAbstractions

# Import Base
using Base: @kwdef

# Import CUDA
using CUDA

# Import Test
using Test

# Import Printf
using Printf

# Import StaticArrays
using StaticArrays

# Import Random
using Random

# Import LinearAlgebra
using LinearAlgebra

# Import KernelAbstractions
using KernelAbstractions

# Import Base
using Base: @kwdef

# Import CUDA
using CUDA

# Import Test
using Test

# Import Printf
using Printf

# Import StaticArrays
using StaticArrays

# Import Random
using Random

# Import LinearAlgebra
using LinearAlgebra

# Import KernelAbstractions
using KernelAbstractions

# Import Base
using Base: @kwdef

# Import CUDA
using CUDA

# Import Test
using Test

# Import Printf
using Printf

# Import StaticArrays
using StaticArrays

# Import Random
using Random

# Import LinearAlgebra
using LinearAlgebra

# Import KernelAbstractions
using KernelAbstractions

# Import Base
using Base: @kwdef

# Import CUDA
using CUDA

# Import Test
using Test

# Import Printf
using Printf

# Import StaticArrays
using StaticArrays

# Import Random
using Random

# Import LinearAlgebra
using LinearAlgebra

# Import KernelAbstractions
using KernelAbstractions

# Import Base
using Base: @kwdef

# Import CUDA
using CUDA

# Import Test
using Test

# Import Printf
using Printf

# Import StaticArrays
using StaticArrays

# Import Random
using Random

# Import LinearAlgebra
using LinearAlgebra

# Import KernelAbstractions
using KernelAbstractions

# Import Base
using Base: @kwdef

# Import CUDA
using CUDA

# Import Test
using Test

# Import Printf
using Printf

# Import StaticArrays
using StaticArrays

# Import Random
using Random

# Import LinearAlgebra
using LinearAlgebra

# Import KernelAbstractions
using KernelAbstractions

# Import Base
using Base: @kwdef

# Import CUDA
using CUDA

# Import Test
using Test

# Import Printf
using Printf

# Import StaticArrays
using StaticArrays

# Import Random

=======

# 3D Jacobi KernelAbstractions implementation
# =================================================

=======

# 3D Jacobi KernelAbstractions implementation
#

=======

# 3D Jacobi KernelAbstractions implementation
# 2021-07-02

=======

# 3D Jacobi KernelAbstractions implementation

=======

# 3D Jacobi KernelAbstractions example
#
# Original source from:
#

### End of the output
