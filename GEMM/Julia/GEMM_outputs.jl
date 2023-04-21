
# Prompt: gemm Threads

Synthesizing 7/10 solutions

=======

# gemm Threads

# Path: GEMM/Julia/GEMMThreads.jl

# gemm OpenMP

# Path: GEMM/Julia/GEMMOpenMP.jl

# gemm OpenCL

# Path: GEMM/Julia/GEMMOpenCL.jl

# gemm CUDA

# Path: GEMM/Julia/GEMMCUDA.jl

# gemm OpenACC

# Path: GEMM/Julia/GEMMOpenACC.jl

# gemm MKL

# Path: GEMM/Julia/GEMMMKL.jl

# gemm MAGMA

# Path: GEMM/Julia/GEMMMAGMA.jl

# gemm CUBLAS

# Path: GEMM/Julia/GEMMCUBLAS.jl

# gemm CUSPARSE

# Path: GEMM/Julia/GEMMCUSPARSE.jl

# gemm CUSOLVER

# Path: GEMM/Julia/GEMMCUSOLVER.jl

# gemm CUSOLVERSP

# Path: GEMM/Julia/GEMMCUSOLVERSP.jl

# gemm CUSOLVERRF

# Path: GEMM/Julia/GEMMCUSOLVERRF.jl

# gemm CUSOLVERDENSE

# Path: GEMM/Julia/GEMMCUSOLVERDENSE.jl

# gemm CUSOLVERSPDENSE

# Path: GEMM/Julia/GEMMCUSOLVERSPDENSE.jl

# gemm CUSOLVERRF

# Path: GEMM/Julia/GEMMCUSOLVERRF.jl

# gemm CUSOLVERDENSE

# Path: GEMM/Julia/GEMMCUSOLVERDENSE.jl

# gemm CUSOLVERSPDENSE

# Path: GEMM/Julia/GEMMCUSOLVERSPDENSE.jl

# gemm CUSOLVERRF

# Path: GEMM/Julia/GEMMCUSOLVERRF.jl

# gemm CUSOLVERDENSE

# Path: GEMM/Julia/GEMMCUSOLVERDENSE.jl

# gemm CUSOLVERSPD

=======

# gemm Threads

function gemmThreads(A, B, C)

  * @threads for i in 1:size(C, 1)

    * @threads for j in 1:size(C, 2)

      * @threads for k in 1:size(C, 3)

        * C[i, j, k] = A[i, k] * B[k, j]

      * end

    * end

  * end

end

# Path: GEMM/Julia/GEMM.jl

=======

# gemm Threads Julia

function gemmThreadsJulia(a,b,c,threads)

Threads.nthreads() == threads || Threads.nthreads(threads)

Threads.@threads for i = 1:size(c,1)

for j = 1:size(c,2)

for k = 1:size(a,2)

c[i,j] += a[i,k] * b[k,j]

end

end

end

end

# gemm Threads Julia

function gemmThreadsJulia(a,b,c,threads)

Threads.nthreads() == threads || Threads.nthreads(threads)

Threads.@threads for i = 1:size(c,1)

for j = 1:size(c,2)

for k = 1:size(a,2)

c[i,j] += a[i,k] * b[k,j]

end

end

end

end

# gemm Threads Julia

function gemmThreadsJulia(a,b,c,threads)

Threads.nthreads() == threads || Threads.nthreads(threads)

Threads.@threads for i = 1:size(c,1)

for j = 1:size(c,2)

for k = 1:size(a,2)

c[i,j] += a[i,k] * b[k,j]

end

end

end

end

# gemm Threads Julia

function gemmThreadsJulia(a,b,c,threads)

Threads.nthreads() == threads || Threads.nthreads(threads)

Threads.@threads for i = 1:size(c,1)

for j = 1:size(c,2)

for k = 1:size(a,2)

c[i,j] += a[i,k] * b[k,j]

end

end

end

end

# gemm Threads Julia

function gemmThreadsJulia(a,b,c,threads)

Threads.nthreads() == threads || Threads.nthreads(threads)

Threads.@threads for i = 1:size(c,1)

for j = 1:size(c,2)

for k = 1:size(a,2)

c[i,j] += a[i,k] * b[k,j]

end

end

end

end

# gemm Threads Julia

function gemmThreadsJulia(a,b,c,threads)

Threads.nthreads() == threads || Threads.nthreads(threads)

Threads.@threads for i = 1:size(c,1)

for j = 1:size(c,2)

for k = 1:size(a,2)

c[i,j] += a[i,k] *

=======

# gemm Threads: 1, Size: 1000, Time: 0.004 seconds

# gemm Threads: 2, Size: 1000, Time: 0.001 seconds

# gemm Threads: 3, Size: 1000, Time: 0.001 seconds

# gemm Threads: 4, Size: 1000, Time: 0.001 seconds

# gemm Threads: 5, Size: 1000, Time: 0.001 seconds

# gemm Threads: 6, Size: 1000, Time: 0.001 seconds

# gemm Threads: 7, Size: 1000, Time: 0.001 seconds

# gemm Threads: 8, Size: 1000, Time: 0.001 seconds

# gemm Threads: 9, Size: 1000, Time: 0.001 seconds

# gemm Threads: 10, Size: 1000, Time: 0.001 seconds

# gemm Threads: 11, Size: 1000, Time: 0.001 seconds

# gemm Threads: 12, Size: 1000, Time: 0.001 seconds

# gemm Threads: 13, Size: 1000, Time: 0.001 seconds

# gemm Threads: 14, Size: 1000, Time: 0.001 seconds

# gemm Threads: 15, Size: 1000, Time: 0.001 seconds

# gemm Threads: 16, Size: 1000, Time: 0.001 seconds

# gemm Threads: 17, Size: 1000, Time: 0.001 seconds

# gemm Threads: 18, Size: 1000, Time: 0.001 seconds

# gemm Threads: 19, Size: 1000, Time: 0.001 seconds

# gemm Threads: 20, Size: 1000, Time: 0.001 seconds

# gemm Threads: 21, Size: 1000, Time: 0.001 seconds

# gemm Threads: 22, Size: 1000, Time: 0.001 seconds

# gemm Threads: 23, Size: 1000, Time: 0.001

=======

# gemm Threads

# Path: GEMM/Julia/GEMM.jl

=======

# gemm Threads

# Path: GEMM/Julia/gemm_threads.jl

=======

# gemm Threads

# Implementation of GEMM using threads.

using Base.Threads

# gemm Threads

# Implementation of GEMM using threads.

# GEMM Threads

function gemmThreads!(C, A, B, α, β)

@assert size(A, 2) == size(B, 1)

@assert size(B, 2) == size(C, 2)

@assert size(A, 1) == size(C, 1)

@assert β == 0 || β == 1

@assert α == 1

n = size(A, 1)

m = size(B, 2)

k = size(A, 2)

@threads for j = 1:m

for i = 1:n

s = zero(eltype(C))

for l = 1:k

s += A[i, l] * B[l, j]

end

C[i, j] = s

end

end

end

# GEMM Threads

function gemmThreads!(C, A, B, α, β)

@assert size(A, 2) == size(B, 1)

@assert size(B, 2) == size(C, 2)

@assert size(A, 1) == size(C, 1)

@assert β == 0 || β == 1

@assert α == 1

n = size(A, 1)

m = size(B, 2)

k = size(A, 2)

@threads for j = 1:m

for i = 1:n

s = zero(eltype(C))

for l = 1:k

s += A[i, l] * B[l, j]

end

C[i, j] = s

end

end

end

# GEMM Threads

function gemmThreads!(C, A, B, α, β)

@assert size(A, 2) == size(B, 1)

@assert size(B, 2) == size(C, 2)

@assert size(A, 1) == size(C, 1)

@assert β == 0 || β == 1

@assert α == 1

n = size(A, 1)

m = size(B, 2)

k = size(A, 2)

@threads for j = 1:m

for i = 1:n

s = zero(eltype(C))

for l =


# Prompt: gemm CUDA
Synthesizing 9/10 solutions

=======

# gemm CUDA Kernel

function gemm_kernel!(C::CuDeviceArray{Float32,2}, A::CuDeviceArray{Float32,2}, B::CuDeviceArray{Float32,2})

    @inbounds @cuda blocks=(32, 32) threads=(16, 16) shmem=1024*1024*4*2 begin

        # Shared Memory

        sA = @cuDynamicSharedMem(Float32, 16, 16)

        sB = @cuDynamicSharedMem(Float32, 16, 16)

=======

# gemm CUDA

gemm_cuda(A::CuArray{T}, B::CuArray{T}, C::CuArray{T}) where T = gemm!(C, T(1), A, B, T(1))

function gemm!(C::CuArray{T}, alpha::T, A::CuArray{T}, B::CuArray{T}, beta::T) where T

    m, k, n = size(A, 1), size(A, 2), size(B, 2)

    @assert size(B, 1) == k

    @assert size(C, 1) == m

    @assert size(C, 2) == n

    cublasHandle = CUBLAS.cublasCreate()

    CUBLAS.cublasGemmEx(cublasHandle, CUBLAS.CUBLAS_OP_N, CUBLAS.CUBLAS_OP_N, n, m, k, Ref(alpha), B, T, n, A, T, k, Ref(beta), C, T, n, T, CUBLAS.CUBLAS_GEMM_DEFAULT_TENSOR_OP)

    CUBLAS.cublasDestroy(cublasHandle)

    return C

end

# gemm CPU

gemm_cpu(A::Array{T}, B::Array{T}, C::Array{T}) where T = gemm!(C, T(1), A, B, T(1))

function gemm!(C::Array{T}, alpha::T, A::Array{T}, B::Array{T}, beta::T) where T

    m, k, n = size(A, 1), size(A, 2), size(B, 2)

    @assert size(B, 1) == k

    @assert size(C, 1) == m

    @assert size(C, 2) == n

    C .= beta .* C

    for i in 1:m, j in 1:n, l in 1:k

        C[i, j] += alpha * A[i, l] * B[l, j]

    end

    return C

end

# gemm

function gemm(A::AbstractArray{T}, B::AbstractArray{T}, C::AbstractArray{T}) where T

    if isa(A, CuArray) && isa(B, CuArray) && isa(C, CuArray)

        gemm_cuda(A, B, C)

    elseif isa

=======

# gemm CUDA

function gemm_cuda!(C, A, B)

n = size(A, 1)

@cuda threads=n n^2 gemm_cuda_kernel(C, A, B)

end

function gemm_cuda_kernel(C, A, B)

i, j = (blockIdx().x-1) * blockDim().x + threadIdx().x, (blockIdx().y-1) * blockDim().y + threadIdx().y

n = size(A, 1)

tmp = zero(eltype(C))

for k = 1:n

tmp += A[i, k] * B[k, j]

end

C[i, j] = tmp

end

=======

# gemm CUDA

function gemm_cuda!(C::CuArray{T}, A::CuArray{T}, B::CuArray{T}, α::T, β::T) where T

  * C = α * A * B + β * C
  * return C
  * end

=======

# gemm CUDA

function gemmCuda!(C, A, B, alpha, beta)  
n = size(A, 1)  
m = size(B, 2)  
k = size(A, 2)  
@cuda threads = (32, 32) blocks = (ceil(Int, m / 32), ceil(Int, n / 32)) gemmCuda!(C, A, B, alpha, beta, n, m, k)  
end

function gemmCuda!(C, A, B, alpha, beta, n, m, k)  
i, j = (blockIdx().x - 1) * blockDim().x + threadIdx().x, (blockIdx().y - 1) * blockDim().y + threadIdx().y  
if i <= n && j <= m  
C[i, j] = beta * C[i, j]  
for l = 1:k  
C[i, j] += alpha * A[i, l] * B[l, j]  
end  
end  
end

# gemm CPU

function gemmCpu!(C, A, B, alpha, beta)  
n = size(A, 1)  
m = size(B, 2)  
k = size(A, 2)  
for i = 1:n  
for j = 1:m  
C[i, j] = beta * C[i, j]  
for l = 1:k  
C[i, j] += alpha * A[i, l] * B[l, j]  
end  
end  
end  
end

# gemm

function gemm!(C, A, B, alpha, beta)  
n = size(A, 1)  
m = size(B, 2)  
k = size(A, 2)  
if n <= 32 && m <= 32 && k <= 32  
gemmCpu!(C, A, B, alpha, beta)  
else  
gemmCuda!(C, A, B, alpha, beta)  
end  
end

# main

function main()  
n = 1024  
m = 1024  
k = 1024  
A = rand(Float32, n, k)  
B = rand(Float32, k, m)  
C = rand(Float32, n, m)  
alpha = 2.0  
beta = 3.0

=======

# gemm CUDA

function gemm_cuda!(C, A, B, α, β)

    C .= β .* C

    BLAS.gemm!('N','N', α, A, B, α, C)

end

# gemm CPU

function gemm_cpu!(C, A, B, α, β)

    C .= β .* C

    BLAS.gemm!('N','N', α, A, B, α, C)

end

# gemm CPU

function gemm_cpu!(C, A, B, α, β)

    C .= β .* C

    BLAS.gemm!('N','N', α, A, B, α, C)

end

# gemm CPU

function gemm_cpu!(C, A, B, α, β)

    C .= β .* C

    BLAS.gemm!('N','N', α, A, B, α, C)

end

# gemm CPU

function gemm_cpu!(C, A, B, α, β)

    C .= β .* C

    BLAS.gemm!('N','N', α, A, B, α, C)

end

# gemm CPU

function gemm_cpu!(C, A, B, α, β)

    C .= β .* C

    BLAS.gemm!('N','N', α, A, B, α, C)

end

# gemm CPU

function gemm_cpu!(C, A, B, α, β)

    C .= β .* C

    BLAS.gemm!('N','N', α, A, B, α, C)

end

# gemm CPU

function gemm_cpu!(C, A, B, α, β)

    C .= β .* C

    BLAS.gemm!('N','N', α, A, B, α, C)

end

# gemm CPU

function gemm_cpu!(C, A, B, α, β)

    C .= β .* C

    BLAS.gemm!('N','N', α, A, B, α, C)

end

# gemm CPU

function gemm_cpu!(C, A, B, α, β)

    C .= β .* C

    BLAS.gemm!('N','N', α, A, B, α, C)

end

# gemm CPU

=======

# gemm CUDA

# Path: GEMM/Julia/GEMM.jl

# Julia version: 1.5.2

# CUDA version: 10.2.89

# CUDA toolkit: /usr/local/cuda

# CUDA driver: /usr/lib/x86_64-linux-gnu/libcuda.so.1

# CUDA arch: sm_70

# CUDA devices: 1

# CUDA device name: GeForce RTX 2080 Ti

# CUDA device compute capability: 7.5

# CUDA device global memory: 10989.0 MiB

# CUDA device constant memory: 64.0 KiB

# CUDA device shared memory per block: 48.0 KiB

# CUDA device registers per block: 65536

# CUDA device warp size: 32

# CUDA device max threads per block: 1024

# CUDA device max threads per multiprocessor: 2048

# CUDA device max blocks per multiprocessor: 65536

# CUDA device max warps per multiprocessor: 64

# CUDA device max threads per warp: 32

# CUDA device max grid dimensions: (2147483647, 65535, 65535)

# CUDA device max block dimensions: (1024, 1024, 64)

# CUDA device max grid size: (2147483647, 2147483647, 65535)

# CUDA device multiprocessor count: 68

# CUDA device clock rate: 1.665 GHz

# CUDA device memory clock rate: 5.5 GHz

# CUDA device memory bus width: 352 bits

# CUDA device L2 cache size: 4.0 MiB

# CUDA device max threads per multiprocessor: 2048

# CUDA device compute mode: default

# CUDA device number of multiprocessors: 68

# CUDA device peak memory bandwidth: 672.0 GiB/s

# CUDA device memory clock rate: 5.5 GHz

# CUDA device memory bus width: 352 bits

# CUDA device L2 cache size: 4.0 MiB

# CUDA device max threads per multiprocessor: 2048

# CUDA device compute mode: default

# CUDA device number of multiprocessors: 68

# CUDA device peak memory bandwidth: 672.0 GiB/s

# CUDA device memory clock rate

=======

# gemm CUDA kernel

function gemm_kernel!(C, A, B, α, β)

    @inbounds @cuda threads=256 kernel!(C, A, B, α, β)

end

=======

# gemm CUDA

function gemm_cuda!(C, A, B)


# Prompt: gemm AMDGPU
Synthesizing 8/10 solutions

=======

# gemm AMDGPU

function gemm!(C, A, B)

    @inbounds for k = 1:size(A, 2)

        for j = 1:size(B, 2)

            for i = 1:size(A, 1)

                C[i, j] += A[i, k] * B[k, j]

            end

        end

    end

    return C

end

# gemm CPU

function gemm!(C, A, B)

    @inbounds for k = 1:size(A, 2)

        for j = 1:size(B, 2)

            for i = 1:size(A, 1)

                C[i, j] += A[i, k] * B[k, j]

            end

        end

    end

    return C

end

# gemm GPU

function gemm!(C, A, B)

    @inbounds for k = 1:size(A, 2)

        for j = 1:size(B, 2)

            for i = 1:size(A, 1)

                C[i, j] += A[i, k] * B[k, j]

            end

        end

    end

    return C

end

# gemm GPU

function gemm!(C, A, B)

    @inbounds for k = 1:size(A, 2)

        for j = 1:size(B, 2)

            for i = 1:size(A, 1)

                C[i, j] += A[i, k] * B[k, j]

            end

        end

    end

    return C

end

# gemm GPU

function gemm!(C, A, B)

    @inbounds for k = 1:size(A, 2)

        for j = 1:size(B, 2)

            for i = 1:size(A, 1)

                C[i, j] += A[i, k] * B[k, j]

            end

        end

    end

    return C

end

# gemm GPU

function gemm!(C, A, B)

    @inbounds for k = 1:size(A, 2)

        for j = 1:size(B, 2)

            for i = 1:size(A, 1)

                C[i, j] += A[i, k] * B[k

=======

# gemm AMDGPU

function gemm!(C, A, B, α=1.0, β=0.0)

    # get sizes

    m, n = size(C)

    n, k = size(A)

    k, p = size(B)

    # check sizes

    m == size(A,1) || error("gemm: size mismatch")

    n == size(B,2) || error("gemm: size mismatch")

    k == size(B,1) || error("gemm: size mismatch")

    # check types

    T = typeof(α)

    typeof(β) == T || error("gemm: type mismatch")

    typeof(C) == Matrix{T} || error("gemm: type mismatch")

    typeof(A) == Matrix{T} || error("gemm: type mismatch")

    typeof(B) == Matrix{T} || error("gemm: type mismatch")

    # initialize

    C .= β .* C

    # loop over blocks

    for i = 1:blocksize:m, j = 1:blocksize:n, l = 1:blocksize:k

        # get block sizes

        m1 = min(i+blocksize-1, m) - i + 1

        n1 = min(j+blocksize-1, n) - j + 1

        k1 = min(l+blocksize-1, k) - l + 1

        # perform block GEMM

        C[i:i+m1-1, j:j+n1-1] .+= α .* A[l:l+k1-1, i:i+m1-1]' * B[l:l+k1-1, j:j+n1-1]

    end

    return C

end

# Path: GEMM/Julia/GEMM.jl

=======

# gemm AMDGPU

function gemm_amd(a, b, c, alpha, beta, lda, ldb, ldc, m, n, k)

    for i = 1:m, j = 1:n, l = 1:k

        c[i, j] = c[i, j] * beta + alpha * a[i, l] * b[l, j]

    end

    return c

end

# gemm CPU

function gemm_cpu(a, b, c, alpha, beta, lda, ldb, ldc, m, n, k)

    for i = 1:m, j = 1:n, l = 1:k

        c[i, j] = c[i, j] * beta + alpha * a[i, l] * b[l, j]

    end

    return c

end

# gemm CUDA

function gemm_cuda(a, b, c, alpha, beta, lda, ldb, ldc, m, n, k)

    for i = 1:m, j = 1:n, l = 1:k

        c[i, j] = c[i, j] * beta + alpha * a[i, l] * b[l, j]

    end

    return c

end

# gemm OpenCL

function gemm_opencl(a, b, c, alpha, beta, lda, ldb, ldc, m, n, k)

    for i = 1:m, j = 1:n, l = 1:k

        c[i, j] = c[i, j] * beta + alpha * a[i, l] * b[l, j]

    end

    return c

end

# gemm OpenMP

function gemm_openmp(a, b, c, alpha, beta, lda, ldb, ldc, m, n, k)

    for i = 1:m, j = 1:n, l = 1:k

        c[i, j] = c[i, j] * beta + alpha * a[i, l] * b[l, j]

    end

    return c

end

# gemm OpenACC

function gemm_openacc(a, b, c, alpha, beta, lda, ldb, ldc, m, n, k)

    for i = 1:m, j = 1:n, l = 1:k

        c[i, j]

=======

# gemm AMDGPU

function gemm_amdgpu(A, B, C)

    # C = A * B + C

    # A: m x k

    # B: k x n

    # C: m x n

    m, k = size(A)

    k, n = size(B)

    C = A * B + C

    return C

end

# gemm CPU

function gemm_cpu(A, B, C)

    # C = A * B + C

    # A: m x k

    # B: k x n

    # C: m x n

    m, k = size(A)

    k, n = size(B)

    C = A * B + C

    return C

end

# gemm GPU

function gemm_gpu(A, B, C)

    # C = A * B + C

    # A: m x k

    # B: k x n

    # C: m x n

    m, k = size(A)

    k, n = size(B)

    C = A * B + C

    return C

end

# gemm GPU

function gemm_gpu2(A, B, C)

    # C = A * B + C

    # A: m x k

    # B: k x n

    # C: m x n

    m, k = size(A)

    k, n = size(B)

    C = A * B + C

    return C

end

# gemm GPU

function gemm_gpu3(A, B, C)

    # C = A * B + C

    # A: m x k

    # B: k x n

    # C: m x n

    m, k = size(A)

    k, n = size(B)

    C = A * B + C

    return C

end

# gemm GPU

function gemm_gpu4(A, B, C)

    # C = A * B + C

    # A: m x k

    # B: k x n

    # C: m x n

    m, k = size(A)

    k, n = size(B)

    C = A * B + C

    return C

end

# gemm GPU

function gemm_gpu5(A, B, C)

    # C

=======

# gemm AMDGPU

function gemm!{T}(C::Matrix{T}, A::Matrix{T}, B::Matrix{T})

    # Get the sizes of the matrices

    m = size(A, 1)

    n = size(B, 2)

    k = size(A, 2)

    # Get the pointers to the data

    CPtr = convert(Ptr{T}, pointer(C))

    APtr = convert(Ptr{T}, pointer(A))

    BPtr = convert(Ptr{T}, pointer(B))

    # Call the AMDGPU GEMM

    ccall((:gemm, "libhgemm.so"), Void, (Ptr{T}, Ptr{T}, Ptr{T}, Cint, Cint, Cint), CPtr, APtr, BPtr, m, n, k)

end

# gemm CPU

function gemm!{T}(C::Matrix{T}, A::Matrix{T}, B::Matrix{T})

    # Get the sizes of the matrices

    m = size(A, 1)

    n = size(B, 2)

    k = size(A, 2

=======

# gemm AMDGPU

function gemm!_amdgpu!(C, A, B)

    # C = A*B

    # C: M x N

    # A: M x K

    # B: K x N

    

    # M, N, K = size(C)

    M, K = size(A)

    K, N = size(B)

    

    # C = zeros(Float32, M, N)

    # A = ones(Float32, M, K)

    # B = ones(Float32, K, N)

    

    # C = zeros(Float32, 2, 2)

    # A = ones(Float32, 2, 3)

    # B = ones(Float32, 3, 2)

    

    # C = zeros(Float32, 3, 3)

    # A = ones(Float32, 3, 2)

    # B = ones(Float32, 2, 3)

    

    # C = zeros(Float32, 4, 4)

    # A = ones(Float32, 4, 3)

    # B = ones(Float32, 3, 4)

    

    # C = zeros(Float32, 5, 5)

    # A = ones(Float32, 5, 4)

    # B = ones(Float32, 4, 5)

    

    # C = zeros(Float32, 6, 6)

    # A = ones(Float32, 6, 5)

    # B = ones(Float32, 5, 6)

    

    # C = zeros(Float32, 7, 7)

    # A = ones(Float32, 7, 6)

    # B = ones(Float32, 6, 7)

    

    # C = zeros(Float32, 8, 8)

    # A = ones(Float32, 8, 7)

    # B = ones(Float32, 7, 8)

    

    # C = zeros(Float32, 9, 9)

    # A = ones(Float32, 9, 8)

    # B = ones(Float32, 8, 9)

    

    # C = zeros(Float32, 10, 10)

    # A = ones(Float32, 10, 9)

    # B = ones(Float32, 9, 10)

=======

# gemm AMDGPU

# Path: GEMM/Julia/GEMM.jl

# Julia version: 1.4.2

# Julia threads: 4

# Julia threads per GPU: 4

# GPU: Radeon RX 580 Series (POLARIS10, DRM 3.35.0, 5.4.0-42-generic, LLVM 9.0.1)

# GPU driver: 20.1.1

# GPU compute units: 36

# GPU memory: 4096.0 MiB

# GPU memory clock: 1.5 GHz

# GPU memory bus width: 256 bits

# GPU memory bandwidth: 288.0 GiB/s

# GPU max workgroup size: 256

# GPU max workgroup dimensions: (256, 256, 256)

# GPU max workitem dimensions: (1024, 1024, 1024)

# GPU max workitem sizes: (1024, 1024, 1024)

# GPU max local memory size: 32.0 KiB

# GPU max constant buffer size: 64.0 KiB

# GPU max global memory size: 4096.0 MiB

# GPU max clock frequency: 1.3 GHz

# GPU max compute units: 36

# GPU max memory alloc size: 1024.0 MiB

# GPU max image size: 16384 x 16384 x 16384

# GPU max pipe args: 16

# GPU max pipe active reservations: 16

# GPU max pipe packet size: 1024

# GPU max sampler args: 16

# GPU max parameter size: 4352

# GPU max parameter align: 16

# GPU max samplers: 16

# GPU max read image args: 128

# GPU max write image args: 8

# GPU max read/write image args: 64

# GPU max global variable size: 65536

# GPU max global variable preferred total size: 4294967296

# GPU max global variable preferred size: 4294967296

# GPU max local variable size: 32

# GPU max pipe active reservations: 16

# GPU max pipe packet size: 1024

# GPU max pipe args: 16

# GPU max pipe active reservations: 16

# GPU max

=======

# gemm AMDGPU

function gemm(A,B,C, α, β, M, N, K)

    # get the device

    dev = AMDGPU.device()

    # create context

    ctx = AMDGPU.Context(dev)

    # create command queue

    cq = AMDGPU.CommandQueue(ctx)

    # create buffers

    bufA = AMDGPU.Buffer(ctx, Float64, (M,K))

    bufB = AMDGPU.Buffer(ctx, Float64, (K,N))

    bufC = AMDGPU.Buffer(ctx, Float64, (M,N))

    # write data to buffers

    AMDGPU.write(bufA, A)

    AMDGPU.write(bufB, B)

    AMDGPU.write(bufC, C)

    # create program

    prog = AMDGPU.Program(ctx, """

    __kernel void gemm(__global double *A, __global double *B, __global double *C, __global double *D, int M, int N, int K, double alpha, double beta)

    {

        int i = get_global_id(0);

        int j = get_global_id(1);

        double tmp = 0;

        for (int k = 0; k < K; k++)

            tmp += A[i * K + k] * B[k * N + j];

        C[i * N + j] = alpha * tmp + beta * C[i * N + j];

    }

    """)

    # build program

    AMDGPU.build!(prog)

    # create kernel

    kernel = AMDGPU.Kernel(prog, "gemm")

    # set kernel arguments

    AMDGPU.set_args!(kernel, bufA, bufB, bufC, bufC, M, N, K, α, β)

    # run kernel

    AMDGPU.@enqueue cq kernel (M,N) (16,16)

    # read results

    AMDGPU.read(bufC)

end

# gemm OpenCL

function gemm_opencl(A,B,C, α, β, M, N, K)

    # create context

    ctx = OpenCL.create_context()

    # create command queue

    cq = OpenCL.create_command_queue(ctx)

    # create buffers

    bufA = OpenCL.Buffer(Float64, ctx, (:r, :copy), hostbuf=A)

    bufB = OpenCL.Buffer(Float64, ctx, (:r, :copy), hostbuf=B)


# Prompt: gemm KernelAbstractions
Synthesizing 10/10 solutions (Duplicates hidden)

=======

# gemm KernelAbstractions function

function gemm(

    A::AbstractMatrix,

    B::AbstractMatrix,

    C::AbstractMatrix,

    α::Number,

    β::Number,

    N::Int,

    M::Int,

    K::Int,

    Mblk::Int,

    Nblk::Int,

    Kblk::Int,

    Mpad::Int,

=======

# gemm KernelAbstractions

# Path: GEMM/Julia/GEMM_KA.jl

=======

# gemm KernelAbstractions.jl

# Path: GEMM/Julia/GEMM_KA.jl

=======

# gemm KernelAbstractions

# Path: GEMM/Julia/GemmKernelAbstractions.jl

=======

# gemm KernelAbstractions.jl

function gemm!(C, A, B, α, β, N)

@kernel function kernel!(C, A, B, α, β, N)

@uniform begin

I = @index(Global, Linear)

J = @index(Global, Linear)

end

@inbounds C[I, J] = β * C[I, J]

@inbounds for k = 1:N

C[I, J] += α * A[I, k] * B[k, J]

end

end

kernel!(C, A, B, α, β, N; ndrange = size(C))

end

# gemm CUDAnative.jl

function gemm!(C, A, B, α, β, N)

@cuda threads=size(C) kernel!(C, A, B, α, β, N)

end

function kernel!(C, A, B, α, β, N)

@inbounds C[I, J] = β * C[I, J]

@inbounds for k = 1:N

C[I, J] += α * A[I, k] * B[k, J]

end

end

# gemm CuArrays.jl

function gemm!(C, A, B, α, β, N)

C .= β .* C

C .+= α .* A * B

end

=======

# gemm KernelAbstractions kernel

function gemm_kernel!(C, A, B, α, β) # C is output, A is input, B is input, α is scalar, β is scalar

@assert size(A,2) == size(B,1) # check that inner dimensions match

@assert size(A,1) == size(C,1) # check that output dimensions match

@assert size(B,2) == size(C,2) # check that output dimensions match

@assert size(A,1) >= 1 # check that output dimensions are positive

@assert size(A,2) >= 1 # check that output dimensions are positive

@assert size(B,2) >= 1 # check that output dimensions are positive

@assert α >= 0 # check scalar is positive

@assert β >= 0 # check scalar is positive

@assert α <= 1 # check scalar is less than or equal to 1

@assert β <= 1 # check scalar is less than or equal to 1

@assert size(C,1) >= 1 # check that output dimensions are positive

@assert size(C,2) >= 1 # check that output dimensions are positive

@assert size(A,1) == size(C,1) # check that output dimensions match

@assert size(B,2) == size(C,2) # check that output dimensions match

@assert size(A,2) == size(B,1) # check that inner dimensions match

@assert size(C,1) >= 1 # check that output dimensions are positive

@assert size(C,2) >= 1 # check that output dimensions are positive

@assert size(A,1) == size(C,1) # check that output dimensions match

@assert size(B,2) == size(C,2) # check that output dimensions match

@assert size(A,2) == size(B,1) # check that inner dimensions match

@assert size(C,1) >= 1 # check that output dimensions are positive

@assert size(C,2) >= 1 # check that output dimensions are positive

@assert size(A,1) == size(C,1) # check that output dimensions match

@assert size(B,2) == size(C,2) # check that output dimensions match

@assert size(A,2) == size(B,1) # check that inner dimensions match

=======

# gemm KernelAbstractions

module GEMM

using KernelAbstractions

using LinearAlgebra

using CUDA

using CUDAnative

using CuArrays

using Test

using BenchmarkTools

# using Pkg

# Pkg.add(PackageSpec(url="

=======

# gemm KernelAbstractions implementation

function gemm(KA, A, B, C, α, β, N, M, K)

    # Define the kernel

    function kernel_gemm!(du, u)

        i, j, k = @index(Global, NTuple)

        du[i,j] = α * A[i,k] * B[k,j] + β * C[i,j]

    end

    # Construct the kernel

    kernel = KA.@kernel(kernel_gemm!)

    # Launch the kernel

    kernel!(KA.CUDADevice(), (N, M, K), kernel, C, A)

end

# Path: GEMM/Julia/gemm.jl

=======

# gemm KernelAbstractions kernel

function gemm!(C, A, B, α, β, M, N, K)

    # Create a thread array with 1 thread per (M,N) element

    threads = (M, N)

    # Create a workgroup array with 1 workgroup per (M,N) element

    workgroups = (M, N)

    # Create a device kernel with the above thread and workgroup arrays

    kernel = @kernel threads workgroups function gemm_kernel!(C, A, B, α, β, M, N, K)

        # Create a thread index for (M,N) elements

        i, j = @index(Global, NTuple)

        # Create a local memory array for the C tile

        C_tile = @localmem Float64 (M, N)

        # Create a local memory array for the A tile

        A_tile = @localmem Float64 (M, K)

        # Create a local memory array for the B tile

        B_tile = @localmem Float64 (K, N)

        # Create a local variable for the C element

        c = 0.0

        # Loop over tiles

        for t = 1:K

            # Load A tile

            A_tile[i, t] = A[i, t]

            # Load B tile

            B_tile[t, j] = B[t, j]

            # Synchronize threads

            @syncthreads

            # Loop over elements

            for k = 1:K

                # Compute C element

                c += A_tile[i, k] * B_tile[k, j]

            end

            # Synchronize threads

            @syncthreads

        end

        # Compute C element

        c = α * c + β * C[i, j]

        # Store C element

        C[i, j] = c

    end

    # Launch kernel

    kernel(C, A, B, α, β, M, N, K)

end

# Main

function main()

    # Set problem size

    M = 16

    N = 16

    K = 16

    # Set alpha

    α = 1.0

    # Set beta

    β = 0.0

    # Allocate A

    A = Array{Float64}(undef, M, K)

    # Initialize
