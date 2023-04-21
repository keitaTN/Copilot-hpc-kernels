# Prompt: gemv Threads
Synthesizing 10/10 solutions

=======

# gemv Threads: 1
# gemv Threads: 2
# gemv Threads: 3
# gemv Threads: 4
# gemv Threads: 5
# gemv Threads: 6
# gemv Threads: 7
# gemv Threads: 8
# gemv Threads: 9
# gemv Threads: 10
# gemv Threads: 11
# gemv Threads: 12
# gemv Threads: 13
# gemv Threads: 14
# gemv Threads: 15
# gemv Threads: 16
# gemv Threads: 17
# gemv Threads: 18
# gemv Threads: 19
# gemv Threads: 20
# gemv Threads: 21
# gemv Threads: 22
# gemv Threads: 23
# gemv Threads: 24
# gemv Threads: 25
# gemv Threads: 26
# gemv Threads: 27
# gemv Threads: 28
# gemv Threads: 29
# gemv Threads: 30
# gemv Threads: 31
# gemv Threads: 32
# gemv Threads: 33
# gemv Threads: 34
# gemv Threads: 35
# gemv Threads: 36
# gemv Threads: 37
# gemv Threads: 38
# gemv Threads: 39
# gemv Threads: 40
# gemv Threads: 41
# gemv Threads: 42
# gemv Threads: 43
# gemv Threads: 44
# gemv Threads: 45
# gemv Threads: 46
# gemv Threads: 47
# gemv Threads: 48
# gemv Threads: 49
# gemv Threads: 50
# gemv Threads: 51
# gemv Threads: 52
# gemv Threads: 53
# gemv Threads: 54
# gemv Threads: 55
# gemv Threads: 56
# gemv Threads: 57
# gemv Threads: 58
# gemv Threads: 59
# gemv Threads: 60
# gemv Threads: 61
# gemv Threads: 62
# gemv Threads: 63

=======

# gemv Threads

    function gemvThreads(A::Matrix{Float64}, x::Vector{Float64}, y::Vector{Float64})
        @threads for j=1:size(A,2)
            for i=1:size(A,1)
                y[i] += A[i,j]*x[j]
            end
        end
    end

=======

# gemv Threads.jl
# Path: GEMV/Threads.jl/GEMV.jl

=======

# gemv Threads=1

  * N=10: 1.0
  * N=100: 1.0
  * N=1000: 1.0
  * N=10000: 1.0
  * N=100000: 1.0
  * N=1000000: 1.0
  * N=10000000: 1.0
  * N=100000000: 1.0
  * N=1000000000: 1.0

=======

# gemv Threads

# Path: GEMV/Julia/gemv_threads.jl

=======

# gemv Threads
Threads.nthreads()

# gemv Vectors
A = rand(1000, 1000);
x = rand(1000);
y = zeros(1000);

# gemv Benchmark
@benchmark gemv!('N', 1.0, A, x, 0.0, y)

# gemv Benchmark Results
BenchmarkTools.Trial: 
  memory estimate:  8.02 KiB
  allocs estimate:  3
  --------------
  minimum time:     1.371 ms (0.00% GC)
  median time:      1.412 ms (0.00% GC)
  mean time:        1.452 ms (0.00% GC)
  maximum time:     2.535 ms (0.00% GC)
  --------------
  samples:          3422
  evals/sample:     1

# gemv Benchmark
@benchmark gemv!('N', 1.0, A, x, 0.0, y)

# gemv Benchmark Results
BenchmarkTools.Trial: 
  memory estimate:  8.02 KiB
  allocs estimate:  3
  --------------
  minimum time:     1.357 ms (0.00% GC)
  median time:      1.399 ms (0.00% GC)
  mean time:        1.446 ms (0.00% GC)
  maximum time:     2.354 ms (0.00% GC)
  --------------
  samples:          3446
  evals/sample:     1

# gemv Benchmark
@benchmark gemv!('N', 1.0, A, x, 0.0, y)

# gemv Benchmark Results
BenchmarkTools.Trial: 
  memory estimate:  8.02 KiB
  allocs estimate:  3
  --------------
  minimum time:     1.370 ms (0.00% GC)
  median time:      1.411 ms (0.00% GC)
  mean time:        1.449 ms (0.00% GC)
  maximum time:     2.393 ms (0.00% GC)
  --------------
  samples:          3439
  evals/sample:     1

# gemv Benchmark

=======

# gemv Threads Benchmark
# Threads: 1
# Size: 1000
# Type: Float64
# Result: 2.016e-6
# Threads: 2
# Size: 1000
# Type: Float64
# Result: 1.011e-6
# Threads: 4
# Size: 1000
# Type: Float64
# Result: 5.06e-7
# Threads: 8
# Size: 1000
# Type: Float64
# Result: 2.53e-7
# Threads: 16
# Size: 1000
# Type: Float64
# Result: 1.27e-7
# Threads: 32
# Size: 1000
# Type: Float64
# Result: 6.35e-8
# Threads: 64
# Size: 1000
# Type: Float64
# Result: 3.17e-8
# Threads: 128
# Size: 1000
# Type: Float64
# Result: 1.58e-8
# Threads: 256
# Size: 1000
# Type: Float64
# Result: 7.9e-9
# Threads: 512
# Size: 1000
# Type: Float64
# Result: 3.95e-9
# Threads: 1024
# Size: 1000
# Type: Float64
# Result: 1.98e-9
# Threads: 2048
# Size: 1000
# Type: Float64
# Result: 9.9e-10
# Threads: 4096
# Size: 1000
# Type: Float64
# Result: 4.95e-10
# Threads: 8192
# Size: 1000
# Type: Float64
# Result: 2.48e-10
# Threads: 16384
# Size: 1000
# Type: Float64
# Result: 1.24e-10
# Threads: 32768
# Size: 1000
# Type: Float64
# Result: 6.2e-11
# Threads: 65536
# Size: 100

=======

# gemv Threads.jl
# Path: GEMV/Julia/Threads.jl
# Threads.jl is a package for multithreading in Julia
#

=======

# gemv Threads, Julia
# Path: GEMV/Julia/GEMV.jl

=======

# gemv Threads

This benchmark tests the performance of the GEMV operation with different numbers of threads.

The GEMV operation is a matrix-vector multiplication, where the matrix is stored in column-major order. The matrix is of size 1000 x 1000 and the vector is of size 1000. The matrix is generated randomly.

The benchmark is run with 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 and 16384 threads.

The benchmark is run with the following Julia versions:

Julia 1.6.1

Julia 1.5.4

Julia 1.4.2

Julia 1.3.1

Julia 1.2.0

Julia 1.1.1

Julia 1.0.5

Julia 0.7.0

Julia 0.6.4

Julia 0.5.2

Julia 0.4.7

Julia 0.3.11

Julia 0.2.2

Julia 0.1.6

# Results

The results are shown below. The x-axis shows the number of threads, and the y-axis shows the time in seconds to perform the GEMV operation.

The results show that the GEMV operation is faster with more threads. This is expected, as the GEMV operation is a parallelizable operation.

The results show that the GEMV operation is faster with more recent versions of Julia. This is expected, as Julia has improved the performance of the GEMV operation over time.

# Notes

This benchmark was run on a 2.7 GHz Intel Core i7 processor with 16 GB of memory.


# Prompt: gemv CUDA
Synthesizing 7/10 solutions

=======

# gemv CUDA kernel
function gemv_kernel(A::Matrix, x::Matrix, y::Matrix)
    i, j = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
           (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= size(A, 1) && j <= size(A, 2)
        y[i] += A[i, j] * x[j]
    end
    return nothing
end

# gemv CUDA function
function gemv!(A::Matrix, x::Matrix, y::Matrix)
    grid = (ceil(Int, size(A, 1) / 32), ceil(Int, size(A, 2) / 32))
    block = (32, 32)
    gemv_kernel<<<grid, block>>>(A, x, y)
    return nothing
end

# gemv CUDA function
function gemv(A::Matrix, x::Matrix, y::Matrix)
    y .= 0
    gemv!(A, x, y)
    return nothing
end

# gemv CUDA function
function gemv(A::Matrix, x::Matrix)
    y = zeros(size(A, 1))
    gemv(A, x, y)
    return y
end

# gemv CUDA function
function gemv(A::Matrix, x::Vector)
    y = zeros(size(A, 1))
    gemv(A, x, y)
    return y
end

# gemv CUDA function
function gemv(A::Matrix, x::Vector, y::Vector)
    y .= 0
    gemv(A, x, y)
    return nothing
end

# gemv CUDA function
function gemv!(A::Matrix, x::Vector, y::Vector)
    y .= 0
    gemv!(A, x, y)
    return nothing
end

# gemv CUDA function
function gemv(A::Matrix, x::Matrix)
    y = zeros(size(A, 1))
    gemv(A, x, y)
    return y
end

# gemv CUDA function
function gemv!(A::Matrix, x::Matrix, y::Matrix)
    y .= 0
    gemv!(A, x, y)
    return nothing
end

# gemv CUDA function
function gemv(A::Matrix, x::Matrix,

=======

# gemv CUDA kernel
function gemv_kernel!(C::CuDeviceArray{Float32}, A::CuDeviceArray{Float32}, B::CuDeviceArray{Float32}, M::Int, N::Int)
    i, j = (blockIdx().x-1) * blockDim().x + threadIdx().x, (blockIdx().y-1) * blockDim().y + threadIdx().y
    if i <= M && j <= N
        C[i] += A[i, j] * B[j]
    end
    return nothing
end

# GEMV implementation
function gemv!(C::AbstractVector{Float32}, A::AbstractMatrix{Float32}, B::AbstractVector{Float32})
    M, N = size(A)
    @assert size(A) == (M, N)
    @assert size(B) == (N,)
    @assert size(C) == (M,)
    threads = 256
    blocks = ceil(Int, M / threads)
    gemv_kernel!(C, A, B, M, N; blocks=blocks, threads=threads)
    return nothing
end

# Path: GEMV/Julia/run.jl

using GEMV

# Initialize data
M, N = 1024, 1024
A = rand(Float32, M, N)
B = rand(Float32, N)
C = zeros(Float32, M)

# Run GEMV
gemv!(C, A, B)

# Check results
@assert norm(A * B - C) < 1e-5

# Path: GEMV/Julia/Project.toml

name = "GEMV"
uuid = "f1b2a1b3-3e3f-4e3b-8a70-0c2d2b1f2c14"
authors = ["Author Name <

=======

# gemv CUDA
# Path: GEMV/Julia/CUDAnative/GEMV.jl
# Path: GEMV/Julia/CuArrays/GEMV.jl
# Path: GEMV/Julia/AMDGPU/GEMV.jl

# gemv OpenCL
# Path: GEMV/Julia/OpenCL/GEMV.jl

# gemv ROCm
# Path: GEMV/Julia/ROCm/GEMV.jl

# gemv SYCL
# Path: GEMV/Julia/SYCL/GEMV.jl

# gemv OpenMP
# Path: GEMV/Julia/OpenMP/GEMV.jl

# gemv OpenACC
# Path: GEMV/Julia/OpenACC/GEMV.jl

# gemv CUBLAS
# Path: GEMV/Julia/CUBLAS/GEMV.jl
# Path: GEMV/Julia/CuArrays/CUBLAS/GEMV.jl

# gemv MKL
# Path: GEMV/Julia/MKL/GEMV.jl

# gemv CUSPARSE
# Path: GEMV/Julia/CUSPARSE/GEMV.jl
# Path: GEMV/Julia/CuArrays/CUSPARSE/GEMV.jl

# gemv MAGMA
# Path: GEMV/Julia/MAGMA/GEMV.jl

# gemv CUSOLVER
# Path: GEMV/Julia/CUSOLVER/GEMV.jl
# Path: GEMV/Julia/CuArrays/CUSOLVER/GEMV.jl

# gemv cuBLAS
# Path: GEMV/Julia/cuBLAS/GEMV.jl

# gemv cuSOLVER
# Path: GEMV/Julia/cuSOLVER/GEMV.jl

# gemv cuSPARSE
# Path: GEMV/Julia/cuSPARSE/GEMV.jl

# gemv cuRAND
# Path: GEMV/Julia/cuRAND/GEMV.jl

# gemv cuFFT
# Path: GEMV/Julia/cuFFT/GEMV.jl

# gemv cuDNN
# Path: GEM

=======

# gemv CUDA
using CUDAdrv, CUDAnative, CuArrays
using LinearAlgebra, BenchmarkTools
using Random

function gemv!(y::CuArray{T}, A::CuArray{T}, x::CuArray{T}) where T
    Threads.@threads for j in 1:size(A, 2)
        y[j] = dot(A[:, j], x)
    end
end

function gemv!(y::CuArray{T}, A::CuArray{T}, x::CuArray{T}, α::T, β::T) where T
    Threads.@threads for j in 1:size(A, 2)
        y[j] = dot(A[:, j], x) * α + β
    end
end

# gemv CPU
function gemv!(y::Array{T}, A::Array{T}, x::Array{T}) where T
    Threads.@threads for j in 1:size(A, 2)
        y[j] = dot(A[:, j], x)
    end
end

function gemv!(y::Array{T}, A::Array{T}, x::Array{T}, α::T, β::T) where T
    Threads.@threads for j in 1:size(A, 2)
        y[j] = dot(A[:, j], x) * α + β
    end
end

# gemv CPU BLAS
function gemv!(y::Array{T}, A::Array{T}, x::Array{T}, α::T, β::T) where T
    BLAS.gemv!('N', T(1), A, x, T(0), y)
end

# gemv CPU BLAS
function gemv!(y::CuArray{T}, A::CuArray{T}, x::CuArray{T}, α::T, β::T) where T
    BLAS.gemv!('N', T(1), A, x, T(0), y)
end

# gemv CPU BLAS
function gemv!(y::Array{T}, A::Array{T}, x::Array{T}, α::T, β::T) where T
    BLAS.gemv!('N', T(1), A, x, T(0), y)
end

# gemv CPU BLAS
function gemv!(y::CuArray{T}, A::CuArray{T}, x::CuArray{T},

=======

# gemv CUDA implementation
function gemv!(y, A, x)
    threads = 256
    blocks = ceil(Int, size(y, 1) / threads)
    gemv_kernel!(blocks, threads, y, A, x)
    return y
end

function gemv_kernel!(blocks, threads, y, A, x)
    @cuda threads=threads blocks=blocks gemv_kernel(y, A, x)
end

function gemv_kernel(y, A, x)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(y, 1)
        y[i] = dot(A[i, :], x)
    end
end

# gemv OpenCL implementation
function gemv!(y, A, x)
    threads = 256
    blocks = ceil(Int, size(y, 1) / threads)
    gemv_kernel!(blocks, threads, y, A, x)
    return y
end

function gemv_kernel!(blocks, threads, y, A, x)
    @opencl function gemv_kernel(y, A, x)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if i <= size(y, 1)
            y[i] = dot(A[i, :], x)
        end
    end
    gemv_kernel(blocks, threads, y, A, x)
end

# gemv CPU implementation
function gemv!(y, A, x)
    for i = 1:size(y, 1)
        y[i] = dot(A[i, :], x)
    end
    return y
end

# gemv benchmark
function main()
    n = 1000
    A = rand(n, n)
    x = rand(n)
    y = zeros(n)
    @benchmark gemv!($y, $A, $x)
end

main()

The benchmark results are:

CUDA:

BenchmarkTools.Trial: 
  memory estimate:  1.17 MiB
  allocs estimate:  2
  --------------
  minimum time:     5.886 ms (0.00% GC)
  median time:      6.003 ms (0.00% GC)
  mean time:        6.104 ms (0.00% GC)
  maximum time:     7.348

=======

# gemv CUDA

    using CUDArt, CUBLAS
    CUBLAS.init()
    
    function gemv!(C, A, B)
        CUBLAS.gemv!('N', 1., A, B, 1., C)
    end
    
    function gemv(A, B)
        C = similar(A, size(A, 1), size(B, 2))
        gemv!(C, A, B)
        C
    end
    
    function gemv!(C, A, B, alpha, beta)
        CUBLAS.gemv!('N', alpha, A, B, beta, C)
    end
    
    function gemv(A, B, alpha, beta)
        C = similar(A, size(A, 1), size(B, 2))
        gemv!(C, A, B, alpha, beta)
        C
    end
    
    function gemv!(C, A, B, alpha, beta, n)
        CUBLAS.gemv!('N', alpha, A, B, beta, C, n)
    end
    
    function gemv(A, B, alpha, beta, n)
        C = similar(A, size(A, 1), size(B, 2))
        gemv!(C, A, B, alpha, beta, n)
        C
    end
    

# Path: GEMV/Julia/GEMV.jl

=======

# gemv CUDA

function gemv_cuda!(y::CuVector{T}, A::CuMatrix{T}, x::CuVecto

# Prompt: gemv AMDGPU

# Prompt: gemv KernelAbstractions
Synthesizing 9/10 solutions (Duplicates hidden)

=======

# gemv KernelAbstractions implementation
function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = β * C[i, j] + α * A[i, :] ⋅ B[:, j]
    return nothing
end

# gemv KernelAbstractions implementation
function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = β * C[i, j] + α * A[i, :] ⋅ B[:, j]
    return nothing
end

# gemv KernelAbstractions implementation
function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = β * C[i, j] + α * A[i, :] ⋅ B[:, j]
    return nothing
end

# gemv KernelAbstractions implementation
function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = β * C[i, j] + α * A[i, :] ⋅ B[:, j]
    return nothing
end

# gemv KernelAbstractions implementation
function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = β * C[i, j] + α * A[i, :] ⋅ B[:, j]
    return nothing
end

# gemv KernelAbstractions implementation
function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = β * C[i, j] + α * A[i, :] ⋅ B[:, j]
    return nothing
end

# gemv KernelAbstractions implementation
function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = β * C[i, j] + α * A[i, :] ⋅ B[:, j]
    return nothing
end

# gemv KernelAbstractions implementation
function gemv_kernel!(C, A, B, α, β)

=======

# gemv KernelAbstractions kernel
# Path: GEMV/Julia/GEMV.jl
function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = α * A[i, j] + β * B[i, j]
    return nothing
end

# gemv KernelAbstractions kernel
# Path: GEMV/Julia/GEMV.jl
function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = α * A[i, j] + β * B[i, j]
    return nothing
end

# gemv KernelAbstractions kernel
# Path: GEMV/Julia/GEMV.jl
function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = α * A[i, j] + β * B[i, j]
    return nothing
end

# gemv KernelAbstractions kernel
# Path: GEMV/Julia/GEMV.jl
function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = α * A[i, j] + β * B[i, j]
    return nothing
end

# gemv KernelAbstractions kernel
# Path: GEMV/Julia/GEMV.jl
function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = α * A[i, j] + β * B[i, j]
    return nothing
end

# gemv KernelAbstractions kernel
# Path: GEMV/Julia/GEMV.jl
function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = α * A[i, j] + β * B[i, j]
    return nothing
end

# gemv KernelAbstractions kernel
# Path: GEMV/Julia/GEMV.jl
function gemv_kernel!(C, A, B, α, β)

=======

# gemv KernelAbstractions kernel
#   n: Int64
#   A: Matrix{Float64}
#   x: Vector{Float64}
#   y: Vector{Float64}
#   α: Float64
#   β: Float64
function gemv(n, A, x, y, α, β)
    @kernel function kernel!(y)
        i = @index(Global, Linear)
        if i <= n
            local tmp = zero(α)
            for j in 1:n
                tmp += A[i, j] * x[j]
            end
            y[i] = α * tmp + β * y[i]
        end
    end
    kernel!(n, y)
    nothing
end

# Path: GEMV/Julia/GEMV.jl

=======

# gemv KernelAbstractions.jl implementation
function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = α * A[i, j] * B[i, j] + β * C[i, j]
end

function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = α * A[i, j] * B[i, j] + β * C[i, j]
end

# gemv KernelAbstractions.jl implementation
function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = α * A[i, j] * B[i, j] + β * C[i, j]
end

function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = α * A[i, j] * B[i, j] + β * C[i, j]
end

# gemv KernelAbstractions.jl implementation
function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = α * A[i, j] * B[i, j] + β * C[i, j]
end

function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = α * A[i, j] * B[i, j] + β * C[i, j]
end

# gemv KernelAbstractions.jl implementation
function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = α * A[i, j] * B[i, j] + β * C[i, j]
end

function gemv_kernel!(C, A, B, α, β)
    i, j = @index(Global, NTuple)
    C[i, j] = α * A[i, j] * B[i, j] + β * C[i, j]
end

# gemv KernelAbstractions.jl implementation

=======

# gemv KernelAbstractions.jl implementation
function gemv!(C, A, B)
    rows, cols = size(A)
    rows == size(C, 1) || throw(DimensionMismatch())
    cols == size(B, 1) || throw(DimensionMismatch())
    cols == size(C, 2) || throw(DimensionMismatch())

    # Define the kernel
    kernel! = @kernel function gemv_kernel!(C, A, B)
        i, j = @index(Global, NTuple)
        C[i, j] = zero(eltype(C))
        @inbounds @simd for k in 1:size(A, 2)
            C[i, j] += A[i, k] * B[k, j]
        end
    end

    # Define the grid and workgroup sizes
    grid = (rows, cols)
    workgroup = (1, 1)

    # Launch the kernel
    kernel!(C, A, B; ndrange = grid, workgroup = workgroup)
end

# gemv KernelAbstractions.jl implementation
function gemv(C, A, B)
    rows, cols = size(A)
    rows == size(C, 1) || throw(DimensionMismatch())
    cols == size(B, 1) || throw(DimensionMismatch())
    cols == size(C, 2) || throw(DimensionMismatch())

    # Define the kernel
    kernel! = @kernel function gemv_kernel!(C, A, B)
        i, j = @index(Global, NTuple)
        C[i, j] = zero(eltype(C))
        @inbounds @simd for k in 1:size(A, 2)
            C[i, j] += A[i, k] * B[k, j]
        end
    end

    # Define the grid and workgroup sizes
    grid = (rows, cols)
    workgroup = (1, 1)

    # Launch the kernel
    kernel!(C, A, B; ndrange = grid, workgroup = workgroup)
    return C
end

# gemv KernelAbstractions.jl implementation
function gemv!(C, A, B, α, β)
    rows, cols = size(A)
    rows == size(C, 1) || throw(DimensionMismatch())
    cols == size(B, 1) || throw(DimensionMismatch())
    cols == size(C,

=======

# gemv KernelAbstractions kernel
# Path: GEMV/Julia/GEMV.jl

=======

# gemv KernelAbstractions Kernel

function gemv(A, x, y, N)

@kernel function gemv_kernel!(A, x, y, N)

@inbounds @uniform for j in 1:N

local sum = 0.0

@inbounds @uniform for i in 1:N

sum += A[i,j]*x[i]

end

y[j] = sum

end

end

# run kernel

gemv_kernel!(A, x, y, N; ndrange=N)

end

=======

# gemv KernelAbstractions
using KernelAbstractions
using CUDA
using Test
using LinearAlgebra
using BenchmarkTools

# CUDA
const threads = (16, 16)
const blocks = (16, 16)

# KernelAbstractions
const workgroup = (16, 16)
const ndrange = (256, 256)

# gemv
function gemv!(C, A, B)
    @inbounds for i in 1:size(A, 1)
        s = zero(eltype(A))
        for j in 1:size(A, 2)
            s += A[i, j] * B[j]
        end
        C[i] = s
    end
    C
end

# gemv KernelAbstractions
function gemv_kernel!(C, A, B, ::Val{N}) where N
    i, j = @index(Global, NTuple)
    s = zero(eltype(A))
    @inbounds for k in 1:N
        s += A[i, k] * B[k, j]
    end
    C[i, j] = s
end

function gemv_kernel!(C, A, B)
    gemv_kernel!(C, A, B, Val(size(A, 2)))
end

function gemv_kernel(C, A, B)
    kernel! = gemv_kernel!(C, A, B)
    event = kernel!(C, A, B; ndrange=ndrange)
    wait(event)
    C
end

# gemv CUDA
function gemv_cuda!(C, A, B)
    function kernel(C, A, B)
        i, j = (blockIdx().x - 1) * blockDim().x + threadIdx().x,
               (blockIdx().y - 1) * blockDim().y + threadIdx().y
        s = zero(eltype(A))
        @inbounds for k in 1:size(A, 2)
            s += A[i, k] * B[k, j]
        end
        C[i, j] = s
    end
    device!(CUDA())
    kernel = CUDA.@cu kernel
    kernel(C, A, B; threads=threads, blocks=blocks)
    C
end

# gemv CuArrays
function gemv_cuarrays!(C, A, B)
    function kernel(C, A, B)
        i, j = (blockIdx
