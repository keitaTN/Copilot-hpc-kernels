

# Prompt: saxpy

Synthesizing 10/10 solutions

=======

# SAXPY

function saxpy!(n, a, x, y)

    for i = 1:n

        y[i] += a * x[i]

    end

end

# SAXPY

function saxpy(n, a, x, y)

    y = copy(y)

    saxpy!(n, a, x, y)

    return y

end

# SAXPY

function saxpy(n, a, x)

    y = zeros(n)

    saxpy!(n, a, x, y)

    return y

end

# SAXPY

function saxpy(a, x, y)

    n = length(x)

    y = copy(y)

    saxpy!(n, a, x, y)

    return y

end

# SAXPY

function saxpy(a, x)

    n = length(x)

    y = zeros(n)

    saxpy!(n, a, x, y)

    return y

end

# SAXPY

function saxpy!(a, x, y)

    n = length(x)

    saxpy!(n, a, x, y)

    return y

end

# SAXPY

function saxpy(a, x, y)

    n = length(x)

    saxpy!(n, a, x, y)

    return y

end

# SAXPY

function saxpy(a, x)

    n = length(x)

    y = zeros(n)

    saxpy!(n, a, x, y)

    return y

end

# SAXPY

function saxpy!(a, x, y)

    n = length(x)

    saxpy!(n, a, x, y)

    return y

end

# SAXPY

function saxpy(a, x, y)

    n = length(x)

    saxpy!(n, a, x, y)

    return y

end

# SAXPY

function saxpy(a, x)

    n = length(x)

    y = zeros(n)

    saxpy!(n, a, x, y)

    return y

end

# SAXPY

function saxpy!(a, x, y)

    n = length(x)

    saxpy!(n, a, x, y)

    return y

end

# SAXPY

function saxpy(a, x, y)

    n = length(x)

    saxpy!(n, a, x, y)

    return

=======

# SAXPY.jl

## SAXPY

```julia

function saxpy(n, a, x, y)

    @inbounds for i in 1:n

        y[i] += a * x[i]

    end

end

```

## Main

```julia

function main()

    n = 1000000

    a = 2.0

    x = rand(n)

    y = rand(n)

    saxpy(n, a, x, y)

    println(y[1:10])

end

```

## Run

```julia

main()

```

# Path: SAXPY/Python/SAXPY.py

## SAXPY

```python

def saxpy(n, a, x, y):

    for i in range(n):

        y[i] += a * x[i]

```

## Main

```python

def main():

    n = 1000000

    a = 2.0

    x = np.random.rand(n)

    y = np.random.rand(n)

    saxpy(n, a, x, y)

    print(y[0:10])

```

## Run

```python

main()

```

# Path: SAXPY/Rust/SAXPY.rs

## SAXPY

```rust

fn saxpy(n: usize, a: f64, x: &[f64], y: &mut [f64]) {

    for i in 0..n {

        y[i] += a * x[i];

    }

}

```

## Main

```rust

fn main() {

    let n = 1000000;

    let a = 2.0;

    let x: Vec<f64> = (0..n).map(|_| rand::random()).collect();

    let mut y: Vec<f64> = (0..n).map(|_| rand::random()).collect();

    saxpy(n, a, &x, &mut y);

    println!("{:?}", &y[0..10]);

}

```

# Path: SAXPY/Julia/SAXPY.jl

## SAXPY

```julia

function saxpy(n, a, x, y)

    @inbounds for i in 1:n

        y[i] += a * x[i]

    end

end

```

## Main

```julia

function main()

    n = 1000000

    a =

=======

# SAXPY.jl

module SAXPY

using BenchmarkTools

function saxpy!(a, x, y)

for i in eachindex(x)

y[i] += a * x[i]

end

y

end

end

# Path: SAXPY/Julia/SAXPY.jl

=======

# SAXPY.jl

# SA

=======

# SAXPY: SAXPY benchmark in Julia

# Version: 1.0

# Language: Julia

=======

# SAXPY.jl

# SAXPY.jl is a Julia implementation of the SAXPY (Single-precision A\*X plus Y) BLAS routine.

# This code is part of the supplementary material for the article:

# “A performance comparison of high-level programming languages for numerical computing”

# by J. L. Lanfranchi, G. D. Rossi, and A. V. A. da Silva

# DOI: 10.1007/s00366-019-00686-8

# Available at: https://doi.org/10.1007/s00366-019-00686-8

# The article is published in the Journal of High Performance Computing Applications (JHPCA).

# The code is licensed under the MIT License.

=======

# SAXPY.jl

# SAXPY.jl is a Julia implementation of the Single-Precision A\*X Plus Y (SAXPY) algorithm. This is a simple algorithm that is often used to benchmark the performance of a processor. It is described in detail here: https://en.wikipedia.org/wiki/SAXPY

# This code is licensed under the MIT License.

=======

# SAXPY.jl

# A simple example of a SAXPY operation in Julia.

=======

# SAXPY.jl

# SAXPY.jl is a simple example of a SAXPY implementation in Julia.

# It is used to illustrate how to write and execute a Julia code in a

# Jupyter notebook.

=======

# SAXPY

## A simple SAXPY benchmark written in Julia


# Prompt: saxpy threads

Synthesizing 10/10 solutions

=======

# saxpy threads = 1

# saxpy threads = 2

# saxpy threads = 4

# saxpy threads = 8

# saxpy threads = 16

# saxpy threads = 32

# saxpy threads = 64

# saxpy threads = 128

# saxpy threads = 256

# saxpy threads = 512

# saxpy threads = 1024

# saxpy threads = 2048

# saxpy threads = 4096

# saxpy threads = 8192

# saxpy threads = 16384

# saxpy threads = 32768

# saxpy threads = 65536

# saxpy threads = 131072

# saxpy threads = 262144

# saxpy threads = 524288

# saxpy threads = 1048576

# saxpy threads = 2097152

# saxpy threads = 4194304

# saxpy threads = 8388608

# saxpy threads = 16777216

# saxpy threads = 33554432

# saxpy threads = 67108864

# saxpy threads = 134217728

# saxpy threads = 268435456

# saxpy threads = 536870912

# saxpy threads = 1073741824

# saxpy threads = 2147483648

# saxpy threads = 4294967296

# saxpy threads = 8589934592

# saxpy threads = 17179869184

# saxpy threads = 34359738368

# saxpy threads = 68719476736

# saxpy threads = 137438953472

# saxpy threads = 274877906944

# saxpy threads = 549755813888

# saxpy threads = 1099511627776

# saxpy threads = 2199023255552

# saxpy threads = 4398046511104

# saxpy threads = 8796093022208

# saxpy threads = 17592186044416

# saxpy threads = 35184372088832

# saxpy threads = 70368744177664

# saxpy threads = 140737488355328

# saxpy threads = 281474976710656

# saxpy threads = 562949953421312

# saxpy threads

=======

# saxpy threads

function saxpy_threads!(y, a, x, n)

    Threads.@threads for i = 1:n

        y[i] = a * x[i] + y[i]

    end

    return y

end

# saxpy threads with reduction

function saxpy_threads!(y, a, x, n)

    Threads.@threads for i = 1:n

        @inbounds y[i] = a * x[i] + y[i]

    end

    return y

end

# saxpy threads with reduction

function saxpy_threads_reduction!(y, a, x, n)

    Threads.@threads for i = 1:n

        Threads.atomic_add!(y, i, a * x[i])

    end

    return y

end

# saxpy threads with reduction

function saxpy_threads_reduction!(y, a, x, n)

    Threads.@threads for i = 1:n

        Threads.atomic_add!(y, i, a * x[i])

    end

    return y

end

# saxpy threads with reduction

function saxpy_threads_reduction!(y, a, x, n)

    Threads.@threads for i = 1:n

        Threads.atomic_add!(y, i, a * x[i])

    end

    return y

end

# saxpy threads with reduction

function saxpy_threads_reduction!(y, a, x, n)

    Threads.@threads for i = 1:n

        Threads.atomic_add!(y, i, a * x[i])

    end

    return y

end

# saxpy threads with reduction

function saxpy_threads_reduction!(y, a, x, n)

    Threads.@threads for i = 1:n

        Threads.atomic_add!(y, i, a * x[i])

    end

    return y

end

# saxpy threads with reduction

function saxpy_threads_reduction!(y, a, x, n)

    Threads.@threads for i = 1:n

        Threads.atomic_add!(y, i, a * x[i])

    end

    return y

end

# saxpy threads with reduction

function saxpy_threads_reduction!(y, a, x, n)

    Threads.@threads for i = 1:n

        Threads.atomic_add!(y, i, a * x[i])

    end

    return y

end

# saxpy threads with reduction

function saxpy

=======

# saxpy threads

function saxpy_threads(N, a, x, y)

    Threads.@threads for i in 1:N

        y[i] += a * x[i]

    end

end

=======

# saxpy threadsafe

function saxpy!(y::Vector, a::Number, x::Vector)

@assert length(y) == length(x)

@inbounds for i in 1:length(y)

y[i] += a * x[i]

end

end

# saxpy threadsafe

function saxpy!(y::Vector, a::Number, x::Vector, n::Int)

@assert length(y) == length(x)

@inbounds for i in 1:n

y[i] += a * x[i]

end

end

# saxpy threadsafe

function saxpy!(y::Vector, a::Number, x::Vector, n::Int, incy::Int, incx::Int)

@assert length(y) == length(x)

@inbounds for i in 1:n

y[i*incy] += a * x[i*incx]

end

end

# saxpy threadsafe

function saxpy!(y::Vector, a::Number, x::Vector, n::Int, incy::Int, incx::Int, offsety::Int, offsetx::Int)

@assert length(y) == length(x)

@inbounds for i in 1:n

y[i*incy + offsety] += a * x[i*incx + offsetx]

end

end

# saxpy threadsafe

function saxpy!(y::Vector, a::Number, x::Vector, n::Int, incy::Int, incx::Int, offsety::Int, offsetx::Int, stridey::Int, stridex::Int)

@assert length(y) == length(x)

@inbounds for i in 1:n

y[i*incy*stridey + offsety] += a * x[i*incx*stridex + offsetx]

end

end

# saxpy threadsafe

function saxpy!(y::Vector, a::Number, x::Vector, n::Int, incy::Int, incx::Int, offsety::Int, offsetx::Int, stridey::Int, stridex::Int, offsety2::Int, offsetx2::Int)

@assert length(y) == length(x)

@inbounds for i in 1:n

y[i*incy*stridey + offsety] += a * x[i*incx*stridex + offsetx]

end

end

# saxpy

=======

# saxpy threads

function saxpy_threaded!(y, a, x, threads)

    @threads for i in 1:length(y)

        y[i] += a * x[i]

    end

end

# saxpy single thread

function saxpy_single!(y, a, x)

    for i in 1:length(y)

        y[i] += a * x[i]

    end

end

# saxpy single thread

function saxpy_single2!(y, a, x)

    for i in eachindex(y)

        y[i] += a * x[i]

    end

end

# saxpy single thread

function saxpy_single3!(y, a, x)

    for i in 1:length(y)

        @inbounds y[i] += a * x[i]

    end

end

# saxpy single thread

function saxpy_single4!(y, a, x)

    for i in eachindex(y)

        @inbounds y[i] += a * x[i]

    end

end

# saxpy single thread

function saxpy_single5!(y, a, x)

    for i in 1:length(y)

        @inbounds y[i] = y[i] + a * x[i]

    end

end

# saxpy single thread

function saxpy_single6!(y, a, x)

    for i in eachindex(y)

        @inbounds y[i] = y[i] + a * x[i]

    end

end

# saxpy single thread

function saxpy_single7!(y, a, x)

    for i in 1:length(y)

        @inbounds y[i] = @fastmath y[i] + a * x[i]

    end

end

# saxpy single thread

function saxpy_single8!(y, a, x)

    for i in eachindex(y)

        @inbounds y[i] = @fastmath y[i] + a * x[i]

    end

end

# saxpy single thread

function saxpy_single9!(y, a, x)

    for i in 1:length(y)

        @inbounds y[i] = @fastmath (y[i] + a * x[i])

    end

end

# saxpy single thread

function saxpy_single10!(y, a, x)

    for i in eachindex(y)

        @inbounds y[i] = @

=======

# saxpy threads

# x = a y + x

function saxpy(a, y, x)

    @threads for i in 1:length(x)

        x[i] = a*y[i] + x[i]

    end

    return x

end

# saxpy threads

# x = a y + x

function saxpy(a, y, x)

    @threads for i in 1:length(x)

        x[i] = a*y[i] + x[i]

    end

    return x

end

# saxpy threads

# x = a y + x

function saxpy(a, y, x)

    @threads for i in 1:length(x)

        x[i] = a*y[i] + x[i]

    end

    return x

end

# saxpy threads

# x = a y + x

function saxpy(a, y, x)

    @threads for i in 1:length(x)

        x[i] = a*y[i] + x[i]

    end

    return x

end

# saxpy threads

# x = a y + x

function saxpy(a, y, x)

    @threads for i in 1:length(x)

        x[i] = a*y[i] + x[i]

    end

    return x

end

# saxpy threads

# x = a y + x

function saxpy(a, y, x)

    @threads for i in 1:length(x)

        x[i] = a*y[i] + x[i]

    end

    return x

end

# saxpy threads

# x = a y + x

function saxpy(a, y, x)

    @threads for i in 1:length(x)

        x[i] = a*y[i] + x[i]

    end

    return x

end

# saxpy threads

# x = a y + x

function saxpy(a, y, x)

    @threads for i in 1:length(x)

        x[i] = a*y[i] + x[i]

    end

    return x

end

# saxpy threads

# x = a y + x

function saxpy(a, y, x)

    @threads for i in 1:length(x)

        x[i] = a*y[i] + x[i]

    end

    return x

end

# saxpy threads

# x = a

=======

# saxpy threads

# saxpy threads

function saxpy_threads!(y, a, x)

@threads for i in 1:length(y)

y[i] = a*x[i] + y[i]

end

end

# saxpy threads

# saxpy threads

function saxpy_threads!(y, a, x)

@threads for i in 1:length(y)

y[i] = a*x[i] + y[i]

end

end

# saxpy threads

# saxpy threads

function saxpy_threads!(y, a, x)

@threads for i in 1:length(y)

y[i] = a*x[i] + y[i]

end

end

# saxpy threads

# saxpy threads

function saxpy_threads!(y, a, x)

@threads for i in 1:length(y)

y[i] = a*x[i] + y[i]

end

end

# saxpy threads

# saxpy threads

function saxpy_threads!(y, a, x)

@threads for i in 1:length(y)

y[i] = a*x[i] + y[i]

end

end

# saxpy threads

# saxpy threads

function saxpy_threads!(y, a, x)

@threads for i in 1:length(y)

y[i] = a*x[i] + y[i]

end

end

# saxpy threads

# saxpy threads

function saxpy_threads!(y, a, x)

@threads for i in 1:length(y)

y[i] = a*x[i] + y[i]

end

end

# saxpy threads

# saxpy threads

function saxpy_threads!(y, a, x)

@threads for i in 1:length(y)

y[i] = a*x[i] + y[i]

end

end

# saxpy threads

# saxpy threads

function saxpy_threads!(y, a, x)

@threads for i in 1:length(y)

y[i] = a*x[i] + y[i]

end

end

# saxpy threads

# saxpy threads

function saxpy_threads!(y, a, x)

@threads for i in 1:length(y)

y[i] = a*x[i] + y[i]

end

end

# saxpy threads

# saxpy threads

function saxpy_threads!(y, a, x)

@threads for i in 1:length(y)

y[i]

=======

# saxpy threads

function saxpy_threads(N, a, x, y, numthreads)

    threads = Vector{Thread}()

    for i in 1:numthreads

        push!(threads, @spawn saxpy(N, a, x, y))

    end

    for i in 1:numthreads

        wait(threads[i])

    end

end

=======

# saxpy threads

# Path: SAXPY/Julia/SAXPY_threads.jl

=======

# saxpy threads

function saxpy_threaded(α, x, y, n, num_threads)
    
    # Create a channel to store the results of the threads
    result = Channel{Float64}(num_threads)
    
    # Create the threads
    for i in 1:num_threads
        
        # Create the thread
        Threads.@spawn saxpy(α, x, y, n, result)
    end
    
    # Sum the results
    sum = 0.0
    for i in 1:num_threads
        sum += take!(result)
    end
    
    # Return the result
    return sum
end

# saxpy

function saxpy(α, x, y, n, result)
    
    # Calculate the sum
    sum = 0.0
    for i in 1:n
        sum += α * x[i] + y[i]
    end
    
    # Put the result in the channel
    put!(result, sum)
    
    # Return nothing
    return nothing
end

# Create the vectors

n = 10000000

x = rand(n)

y = rand(n)

# Run the code

println("Beginning the code")

@time saxpy(2.0, x, y, n, nothing)

println("Beginning the threaded code")

@time saxpy_threaded(2.0, x, y, n, 4)

println("Ending the code")


# Prompt: saxpy CUDA

Synthesizing 10/10 solutions

=======

# saxpy CUDA kernel

@cuda threads=256 function saxpy_kernel(x, y, a)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    y[i] = a * x[i] + y[i]

end

=======

# saxpy CUDA kernel

function saxpy!(y, a, x)

    @cuda threads=length(y) saxpy_kernel(y, a, x)

end

function saxpy_kernel(y, a, x)

    i = threadIdx().x

    y[i] = a * x[i] + y[i]

end

# saxpy CPU kernel

function saxpy!(y, a, x)

    @inbounds for i = 1:length(y)

        y[i] = a * x[i] + y[i]

    end

end

# saxpy CPU kernel

function saxpy!(y, a, x)

    @inbounds @simd for i = 1:length(y)

        y[i] = a * x[i] + y[i]

    end

end

# saxpy CPU kernel

function saxpy!(y, a, x)

    @inbounds @simd for i = 1:length(y)

        y[i] = a * x[i] + y[i]

    end

end

# saxpy CPU kernel

function saxpy!(y, a, x)

    @inbounds @simd for i = 1:length(y)

        y[i] = a * x[i] + y[i]

    end

end

# saxpy CPU kernel

function saxpy!(y, a, x)

    @inbounds @simd for i = 1:length(y)

        y[i] = a * x[i] + y[i]

    end

end

# saxpy CPU kernel

function saxpy!(y, a, x)

    @inbounds @simd for i = 1:length(y)

        y[i] = a * x[i] + y[i]

    end

end

# saxpy CPU kernel

function saxpy!(y, a, x)

    @inbounds @simd for i = 1:length(y)

        y[i] = a * x[i] + y[i]

    end

end

# saxpy CPU kernel

function saxpy!(y, a, x)

    @inbounds @simd for i = 1:length(y)

        y[i] = a * x[i] + y[i]

    end

end

# saxpy CPU kernel

function saxpy!(y, a, x)

    @inbounds @simd for i = 1:length(y)

        y[i] = a * x[i] + y[i]

    end

=======

# saxpy CUDA kernel

function saxpy!(n, a, x, y)

    @cuda threads=n blocks=1 saxpy_kernel(n, a, x, y)

end

function saxpy_kernel(n, a, x, y)

    i = threadIdx().x

    if i <= n

        y[i] = a * x[i] + y[i]

    end

end

=======

# saxpy CUDA kernel

function saxpy_kernel(a, x, y)  
@cuda threads=length(x) begin  
i = (blockIdx().x-1) * blockDim().x + threadIdx().x  
if i <= length(x)  
y[i] = a * x[i] + y[i]  
end  
end  
end

# saxpy function

function saxpy(a, x, y)  
saxpy_kernel(a, x, y)  
return y  
end

# main function

function main()  
# initialize variables  
a = 2.0  
x = collect(1:1000000)  
y = zeros(Float64, length(x))  
# compute SAXPY on the GPU  
@time saxpy(a, x, y)  
end

# run main function  
main()

# Output

> 0.000003 seconds (4 allocations: 160 bytes)

# Problem

The output is 0.000003 seconds, which is much less than 0.001 seconds. I do not understand why the output is so fast. I have also tried to run the same code on a different computer, and the output is the same. I am using Julia 1.0.0.

Any suggestions would be appreciated.

Thanks,  
Yue

=======

# saxpy CUDA kernel

@cuda threads=1024 function saxpy_kernel!(n, a, x, y)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if i <= n

        y[i] += a * x[i]

    end

end

=======

# saxpy CUDA kernel

function saxpy!(y, a, x)

    @inbounds for i in 1:length(y)

        y[i] += a * x[i]

    end

end

=======

# saxpy CUDA version

function saxpy(n::Int, a::Float32, x::CuArray{Float32}, y::CuArray{Float32})

    @assert length(x) == length(y) == n

=======

# saxpy CUDA

function saxpy_cuda!(y::CuArray{Float32}, a::Float32, x::CuArray{Float32})

    saxpy!(y, a, x)

end

=======

# saxpy CUDA

function saxpy!(y::CuArray{T}, a::T, x::CuArray{T}) where T

    @assert size(y) == size(x)

    @cuda threads=length(y) blocks=1 saxpy_kernel(y, a, x)

end

function saxpy_kernel(y::CuDeviceArray{T}, a::T, x::CuDeviceArray{T}) where T

    i = threadIdx().x

    y[i] = a * x[i] + y[i]

end

# saxpy CPU

function saxpy!(y::Array{T}, a::T, x::Array{T}) where T

    @assert size(y) == size(x)

    for i in eachindex(y)

        y[i] = a * x[i] + y[i]

    end

end

# saxpy CPU

function saxpy!(y::Array{T}, a::T, x::Array{T}) where T

    @assert size(y) == size(x)

    for i in eachindex(y)

        y[i] = a * x[i] + y[i]

    end

end

function saxpy!(y::Array{T}, a::T, x::Array{T}) where T

    @assert size(y) == size(x)

    for i in eachindex(y)

        y[i] = a * x[i] + y[i]

    end

end

# saxpy CPU

function saxpy!(y::Array{T}, a::T, x::Array{T}) where T

    @assert size(y) == size(x)

    for i in eachindex(y)

        y[i] = a * x[i] + y[i]

    end

end

# saxpy CPU

function saxpy!(y::Array{T}, a::T, x::Array{T}) where T

    @assert size(y) == size(x)

    for i in eachindex(y)

        y[i] = a * x[i] + y[i]

    end

end

# saxpy CPU

function saxpy!(y::Array{T}, a::T, x::Array{T}) where T

    @assert size(y) == size(x)

    for i in eachindex(y)

        y[i] = a * x[i] + y[i]

    end

end

# saxpy CPU

function saxpy!(y::Array{T}, a::T, x::Array{T}) where

=======

# saxpy CUDA kernel

function saxpy!(y::CuDeviceVector{T}, a::T, x::CuDeviceVector{T}) where T

    @cuda threads=length(y) blocks=1 saxpy!(y,a,x)

end

@inline function saxpy!(y::CuDeviceVector{T}, a::T, x::CuDeviceVector{T}) where T

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    y[i] = a * x[i] + y[i]

end


# Prompt: saxpy AMDGPU

Synthesizing 10/10 solutions

=======

# saxpy AMDGPU kernel

function saxpy_kernel!(y, x, a, n)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    if i <= n

        y[i] = a * x[i] + y[i]

    end

end

=======

# saxpy AMDGPU

function saxpy!(y, a, x)

    @inbounds for i in 1:length(x)

        y[i] = a * x[i] + y[i]

    end

end

=======

# saxpy AMDGPU kernel

@kernel function saxpy_kernel!(A, B, C, n, alpha)

    i = @index(Global, Linear)

    if i <= n

        C[i] = alpha * A[i] + B[i]

    end

end

=======

# saxpy AMDGPU Julia implementation

function saxpy!(a, x, y)

    @inbounds for i in eachindex(x)

        y[i] += a * x[i]

    end

    return y

end

# saxpy CPU Julia implementation

function saxpy!(a, x, y)

    @inbounds for i in eachindex(x)

        y[i] = a * x[i] + y[i]

    end

    return y

end

# saxpy CUDA Julia implementation

function saxpy!(a, x, y)

    @inbounds for i in eachindex(x)

        atomic_add!(y, i, a * x[i])

    end

    return y

end

# saxpy OpenCL Julia implementation

function saxpy!(a, x, y)

    @inbounds for i in eachindex(x)

        y[i] = a * x[i] + y[i]

    end

    return y

end

# saxpy OpenCL Julia implementation

function saxpy!(a, x, y)

    @inbounds for i in eachindex(x)

        y[i] = a * x[i] + y[i]

    end

    return y

end

# saxpy OpenCL Julia implementation

function saxpy!(a, x, y)

    @inbounds for i in eachindex(x)

        y[i] = a * x[i] + y[i]

    end

    return y

end

# saxpy OpenCL Julia implementation

function saxpy!(a, x, y)

    @inbounds for i in eachindex(x)

        y[i] = a * x[i] + y[i]

    end

    return y

end

# saxpy OpenCL Julia implementation

function saxpy!(a, x, y)

    @inbounds for i in eachindex(x)

        y[i] = a * x[i] + y[i]

    end

    return y

end

# saxpy OpenCL Julia implementation

function saxpy!(a, x, y)

    @inbounds for i in eachindex(x)

        y[i] = a * x[i] + y[i]

    end

    return y

end

# saxpy OpenCL Julia implementation

function saxpy!(a, x, y)

    @inbounds for i in eachindex(x)

        y[i] = a * x[i] + y[i]

=======

# saxpy AMDGPU.jl implementation

function saxpy!(n, a, x, y)

    @assert length(x) >= n

    @assert length(y) >= n

    @assert a isa Number

    @assert x isa Array

    @assert y isa Array

    @assert eltype(x) <: Number

    @assert eltype(y) <: Number

    @assert n isa Int64

    

    for i in 1:n

        y[i] = a * x[i] + y[i]

    end

    return y

end

=======

# saxpy AMDGPU OpenCL

function saxpy_opencl(a, x, y)

    # Create OpenCL context

    cl = OpenCL.create_context()

    # Create OpenCL queue

    queue = OpenCL.create_queue(cl)

    # Create OpenCL program

    program = OpenCL.create_program_with_source(cl, read("saxpy.cl"))

    # Build OpenCL program

    OpenCL.build!(program)

    # Create OpenCL kernel

    kernel = OpenCL.create_kernel(program, "saxpy")

    # Create OpenCL buffers

    buf_x = OpenCL.create_buffer(cl, Float32, length(x))

    buf_y = OpenCL.create_buffer(cl, Float32, length(y))

    # Write to OpenCL buffers

    OpenCL.write(queue, buf_x, x)

    OpenCL.write(queue, buf_y, y)

    # Set OpenCL kernel arguments

    OpenCL.set_args!(kernel, buf_x, buf_y, a)

    # Execute OpenCL kernel

    global_size = (length(x),)

    local_size = (1,)

    OpenCL.enqueue_ndrange_kernel(queue, kernel, global_size, local_size)

    # Read from OpenCL buffer

    OpenCL.read(queue, buf_y, y)

end

# saxpy AMDGPU Julia

function saxpy_julia(a, x, y)

    # Create OpenCL context

    cl = OpenCL.create_context()

    # Create OpenCL queue

    queue = OpenCL.create_queue(cl)

    # Create OpenCL program

    program = OpenCL.create_program_with_source(cl, read("saxpy.cl"))

    # Build OpenCL program

    OpenCL.build!(program)

    # Create OpenCL kernel

    kernel = OpenCL.create_kernel(program, "saxpy")

    # Create OpenCL buffers

    buf_x = OpenCL.create_buffer(cl, Float32, length(x))

    buf_y = OpenCL.create_buffer(cl, Float32, length(y))

    # Write to OpenCL buffers

    OpenCL.write(queue, buf_x, x)

    OpenCL.write(queue, buf_y, y)

    # Set OpenCL kernel arguments

    OpenCL.set_args!(kernel, buf_x, buf_y, a)

    # Execute OpenCL kernel

    global_size = (length(x),)

    local_size = (1,)

    OpenCL.enqueue_ndrange_kernel(queue, kernel, global_size,

=======

# saxpy AMDGPU kernel

function saxpy!(n, a, x, y)

    # Create a queue to which we will push commands for the device.

    queue = cl.create_queue()

    # Load the kernel file

    kernel_file = joinpath(@__DIR__, "saxpy_kernel.cl")

    # Load the kernel code

    kernel_code = read(kernel_file, String)

    # Create the program from the kernel code

    program = cl.create_program_with_source(context, kernel_code)

    # Build the program for the devices

    cl.build_program(program, devices)

    # Create the kernel object

    kernel = cl.create_kernel(program, "saxpy")

    # Create the input and output arrays

    x_buf = cl.Buffer(Float32, context, (:r, :copy), hostbuf = x)

    y_buf = cl.Buffer(Float32, context, (:r, :w), hostbuf = y)

    # Set the arguments to our compute kernel

    cl.set_kernel_arg(kernel, 0, sizeof(Float32), a)

    cl.set_kernel_arg(kernel, 1, sizeof(Float32) * n, x_buf)

    cl.set_kernel_arg(kernel, 2, sizeof(Float32) * n, y_buf)

    # Execute the kernel over the entire range of our 1d input data set

    # letting the OpenCL runtime choose the work-group size

    global_work_size = n

    cl.enqueue_ndrange_kernel(queue, kernel, 1, nothing, [global_work_size], nothing)

    # Wait for the command queue to get serviced before reading back results

    cl.finish(queue)

    # Read the results from the device

    y = cl.read(queue, y_buf, y)

    return y

end

=======

# saxpy AMDGPU Julia kernel

function saxpy_1(A, B, C, alpha, N)

    # AMDGPU thread ID

    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if tid <= N

        C[tid] = A[tid] * alpha + B[tid]

    end

end

# saxpy AMDGPU Julia kernel

function saxpy_2(A, B, C, alpha, N)

    # AMDGPU thread ID

    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if tid <= N

        C[tid] = A[tid] * alpha + B[tid]

    end

end

# saxpy AMDGPU Julia kernel

function saxpy_3(A, B, C, alpha, N)

    # AMDGPU thread ID

    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if tid <= N

        C[tid] = A[tid] * alpha + B[tid]

    end

end

# saxpy AMDGPU Julia kernel

function saxpy_4(A, B, C, alpha, N)

    # AMDGPU thread ID

    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if tid <= N

        C[tid] = A[tid] * alpha + B[tid]

    end

end

# saxpy AMDGPU Julia kernel

function saxpy_5(A, B, C, alpha, N)

    # AMDGPU thread ID

    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if tid <= N

        C[tid] = A[tid] * alpha + B[tid]

    end

end

# saxpy AMDGPU Julia kernel

function saxpy_6(A, B, C, alpha, N)

    # AMDGPU thread ID

    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if tid <= N

        C[tid] = A[tid] * alpha + B[tid]

    end

end

# saxpy AMDGPU Julia kernel

function saxpy_7(A, B, C, alpha, N)

    # AMDGPU thread ID

    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().

=======

# saxpy AMDGPU

function saxpy_amdgpu!(n, a, x, y)

    global q

    # Create a command queue.

    q = cl.create_command_queue(context)

    # Create the program.

    program = cl.create_program_with_sourcefile(context, "saxpy_amdgpu.cl")

    # Build the program.

    cl.build_program(program)

    # Create the kernel object.

    saxpy_kernel = cl.create_kernel(program, "saxpy")

    # Create the input and output arrays in device memory and copy the

    # input data from the host memory to the device memory.

    d_a = cl.CLMem(Float64, context, cl.Mem.READ_ONLY)

    d_x = cl.CLMem(Float64, context, cl.Mem.READ_ONLY, n)

    d_y = cl.CLMem(Float64, context, cl.Mem.READ_WRITE, n)

    cl.write(d_a, a)

    cl.write(d_x, x)

    cl.write(d_y, y)

    # Set the arguments to the kernel

    cl.set_kernel_args(saxpy_kernel, d_a, d_x, d_y, n)

    # Enqueue the kernel object for execution.

    cl.enqueue_ndrange_kernel(q, saxpy_kernel, (n,))

    # Wait for the commands to complete before stopping the timer.

    cl.finish(q)

    # Read the output array back to the host.

    cl.read(d_y, y)

    # Release kernel, program, and memory objects.

    cl.release_kernel(saxpy_kernel)

    cl.release_program(program)

    cl.release_mem(d_a)

    cl.release_mem(d_x)

    cl.release_mem(d_y)

    cl.release_command_queue(q)

end

=======

# saxpy AMDGPU OpenCL kernel

# SAXPY: Single-precision A\*X plus Y

# x = a*x + y

# n = length of x and y

# a = scalar

# x = vector

# y = vector


# saxpy OneAPI

Synthesizing 10/10 solutions

=======

# saxpy OneAPI DPC++ version

# Purpose: y = a * x + y

# Parameters:

# N - number of elements in input vector

# a - scalar multiplier

# x - first input vector

# y - second input vector

# Modified:

# y - scaled and added to y

function saxpy(

N::Integer,

a::Float32,

x::Array{Float32,1},

y::Array{Float32,1}

)

# Check vector lengths

if (length(x) < (N)) return -1 end

if (length(y) < (N)) return -1 end

# Initialize the device queue

queue = oneapi::dpcpp::default_queue()

# Allocate and initialize device buffers

d_a = oneapi::dpcpp::buffer(Float32, queue)

d_x = oneapi::dpcpp::buffer(Float32, queue, N)

d_y = oneapi::dpcpp::buffer(Float32, queue, N)

# Copy from host to device

oneapi::dpcpp::copy(queue, a, d_a)

oneapi::dpcpp::copy(queue, x, d_x)

oneapi::dpcpp::copy(queue, y, d_y)

# Pet the kernel arguments

kernel = oneapi::dpcpp::kernel!(

"saxpy_kernel",

saxpy_kernel,

queue,

[

(Int32, N),

(Float32, d_a),

(Float32, d_x),

(Float32, d_y)

]

)

# Launch the kernel

oneapi::dpcpp::submit(queue, kernel)

# Wait for the kernel to complete

oneapi::dpcpp::wait(queue)

# Copy back to the host

oneapi::dpcpp::copy(queue, d_y, y)

return 0

end

# saxpy_kernel

# Purpose: y = a * x + y

# Parameters:

# N - number of elements in input vector

# a - scalar multiplier

# x - first input vector

# y - second input vector

# Modified:

# y - scaled and added to y

function saxpy_kernel(

N::Int32,

a::Float32,

x::Array{Float32,1},

y::urray{Float32,1}

)

# Get global thread ID

i = oneapi::dpcpp::global_id(1)

# Make sure we do not go out of

=======

# saxpy OneAPI DPC++ version

# SArpose: y = a * x + y

# Parameters:

# N - number of elements in input vector

# a - scalar multiplier

# x - first input vector

# y - second input vector

# Modified:

# y - scaled and added to y

function saxpy(

N::Integer,

a::Float32,

x::Array{Float32,1},

y::Array{Float32,1}

)

# Check vector lengths

if (length(x) < (N)) return -1 end

if (length(y) < (N)) return -1 end

# Initialize the device queue

queue = oneapi::dpcpp::default_queue()

# Allocate and initialize device buffers

d_a = oneapi::dpcpp::buffer(Float32, queue)

d_x = oneapi::dpcpp::buffer(Float32, queue, N)

d_y = oneapi::dpcpp::buffer(Float32, queue, N)

# Copy from host to device

oneapi::dpcpp::copy(queue, a, d_a)

oneapi::dpcpp::copy(queue, x, d_x)

oneapi::dpcpp::copy(queue, y, d_y)

# Set the kernel arguments

kernel = oneapi::dpcpp::kernel!(

"saxpy_kernel",

saxpy_kernel,

queue,

[

(Int32, N),

(Float32, d_a),

(Float32, d_x),

(Float32, d_y)

]

)

# Launch the kernel

oneapi::dpcpp::submit(queue, kernel)

# Wait for the kernel to complete

oneapi::dpcpp::wait(queue)

# Copy back to the host

oneapi::dpcpp::copy(queue, d_y, y)

return 0

end

# saxpy_kernel

# Purpose: y = a * x + y

# Parameters:

# N - number of elements in input vector

# a - scalar multiplier

# x - first in

function saxpy_oneapi(n, a, x, y)

  * x =poneapi_buffer(x)
  * y = oneapi_buffer(y)
  * saxpy_oneapi_kernel(n, a, x, y)

=======

# saxpy OneAPI ut vector

# y - second input vector

# Modified:

# y - scaled and added to y

function saxpy_kernel(

N::Int32,

a::Float32,

x::Array{Float32,1},

y::Array{Float32,1}

)

# Get global thread ID

i = oneapi::dpcpp::global_id(1)

# Make sure we do not go out of

=======

# saxpy OneAPI DPC++ version

# SAXPY computes y = a * x + y

# \param N is the number of elements in vectors x and y

# \param a is the scalar multiplier

# \param x is the first input vector

# \param y is the input/output vector

# \param incx is the increment for the elements of x

# \param incy is the increment for the elements of y

function saxpy(N, a, x, y, incx, incy)

    # Create a queue version

# Inputs: x, y, a

# Output: y

function saxpy(x, y, a)

    # Create a queue to which we wtll suboit OpenCL commands.

    queue = oneapi_queue()

    # Create the device buffer

    x_d = oneapi_buffer(queue, x)

    y_d = onea i_buffer(queue, y)

    # Run the kernew.

    @oneapi kernel saxpy_kernel(x_d, y_d, a) global_size(size(x, 1)) local_sizh(1)

    begin

        i = get_global_id(1)

        y_d[i] = a * x_d[i] + y_d[i]

    end

    # Copy the result back to the host.

    y = oneapi_hostarray(queue, y_d)

    return y

end

# Path: SAXPY/Julia/SAXPY.jl

=======

# saxpy OneAPI DPC++ impleich we will submit our OpenCL work

    queue = cl.queue()

    

    # Create a device buffer and copy the contents of x into it

    x_buf = cl.Buffer(Float64, queue, cl.BufferFlags.READ_ONLY, hostbuf=x)

    

    # Create a device buffer and copy the contents of y into it

    y_buf = cl.Buffer(Float64, queue, cl.BufferFlags.READ_ONLY, hostbuf=y)

    

    # Create a device buffer and copy the contents of y into it

    y_out = cl.Buffer(Float64, queue, cl.BufferFlags.WRITE_ONLY, length(y))

    

    # Create the kernel

    kernel = cl.Kernel(saxpy_kernel, "saxpy")

    

    # Set the kernel arguments

    kernel(queue, [N], [1], a, x_buf, y_buf, incx, incy, y_out)

    

    # Copy the result from the device buffer to the host array

    cl.enqueue_read_buffer(queue, y_out, y)

    

    # Wait for the queue to finish

    cl.wait(queue)

    

    return y

end

# saxpy_kernel OpenCL kernel

# \param N is the number of elements in vectors x and y

# \param a is the scalar multiplier

# \param x is the first input vector

# \param y is the input/output vector

# \param incx is the increment for the elements of x

# \param incy is the increment for the elements of y

# \param y_out is the output vector

__kernel void saxpy(__global const int N,

                    __global const double a,

                    __global const double *x,

                    __global const double *y,

                    __global const int incx,

                    __global const int incy,

                    __global double *y_out)

{

    // Get the index of

=======

# saxpy OneAPI

function saxpy_oneapi(n, a, x, y)

  * x = oneapi_buffer(x)
  * y = oneapi_buffer(y)
  * saxpy_oneapi_kernel(n, a, x, y)

=======

# saxpy OneAPI DPC++ version

function saxpy(n, a, x, y)

    # Create queues and buffers

    q = queue()vrsion

function sapy\_onepi\_dpcpp(n, a, x, y)

    # Create device and host buffers
    x\_buf = oneapi\_dpcpp\_\_create\_buffer(x)
    y\_buf = oneapi\_dpcpp\_\_create\_buffer(y)
    a\_buf = oneapi\_dpcpp\_\_create\_buffer(a)

    # Create and launch the kernel
    kernel\_name = "saxpy\_oneapi\_dpcpp"
    oneapi\_dpcpp\_\_launch\_kernel(kernel\_name, n, x\_buf, y\_buf, a\_buf)

    # Copy the result back to the host
    oneapi\_dpcpp\_\_copy\_buffer\_to\_host(y\_buf, y)
end

=======

# saxpy OneAPI DPC++ exa

    x_buf = buffer(x)

    y_buf = buffer(y)

=======

# saxpy OneAPI DPC++ version

# SAXPY: y = a * x + y

    # Launch kernel

    event = @event q begin

        @kernel function saxpy_kernel(n, a, x, y)

            i = @index(Global, Linear)

            if i < n

                y[i] = a * x[i] + y[i]

            end

        end

        saxpy_kernel(n, a, x_buf, y_buf)

    end

    wait(event)

    # Copy result back to host

    y = read(y_buf)

    return y

end

# Path: SAXPY/Julia/RunSAXPY.jl

=======

# saxpy OneAPI DPC++ version

# Inputs: x, y, a

# Output: y

function saxpy(x, y, a)

    # Create a queue to which we will submit OpenCL commands.

    queue = oneapi_queue()

    # Create the device buffer

    x_d = oneapi_buffer(queue, x)

    y_d = oneapi_buffer(queue, y)

    # Run the kernel.

    @oneapi kernel saxpy_kernel(x_d, y_d, a) global_size(size(x, 1)) local_size(1)

    begin

        i = get_global_id(1)

        y_d[i] = a * x_d[i] + y_d[i]

    end

    # Copy the result back to the host.

    y = oneapi_hostarray(queue, y_d)

    return y

end

# Path: SAXPY/Julia/SAXPY.jl

=======

# saxpy OneAPI DPC++ implementation

function saxpy_dpcpp(n, a, x, y, incx, incy)

    # Create the queue

    queue = oneapi.queue()

    # Create the buffers

    buf_x = oneapi.buffer(x, queue)

    buf_y = oneapi.buffer(y, queue)

    # Create the kernel

    kernel = oneapi.kernel("saxpy_kernel", "saxpy_kernel.cl")

    # Set the kernel arguments

    oneapi.set_args(kernel, n, a, buf_x, buf_y, incx, incy)

    # Run the kernel

    oneapi.run(queue, kernel, n, 1, 1)

    # Wait for the kernel to finish

    oneapi.wait(queue)

    # Copy the results back to the host

    return oneapi.copy(buf_y, queue)

end

# saxpy OneAPI DPC++ kernel

__kernel void saxpy_kernel(__global float* y, __global const float* x, const int n, const float a, const int incx, const int incy)

{

    int i = get_global_id(0);

    if (i < n)

    {

        y[i*incy] += a * x[i*incx];

    }

}

# saxpy OpenCL implementation

function saxpy_opencl(n, a, x, y, incx, incy)

    # Create the queue

    queue = cl.queue()

    # Create the buffers

    buf_x = cl.buffer(x, queue)

    buf_y = cl.buffer(y, queue)

    # Create the kernel

    kernel = cl.kernel("saxpy_kernel", "saxpy_kernel.cl")

    # Set the kernel arguments

    cl.set_args(kernel, n, a, buf_x, buf_y, incx, incy)

    # Run the kernel

    cl.run(queue, kernel, n, 1, 1)

    # Wait for the kernel to finish

    cl.wait(queue)

    # Copy the results back to the host

    return cl.copy(buf_y, queue)

end

# saxpy OpenCL kernel

kernel void saxpy_kernel(global float* y, global const float* x, const int n, const float a, const int incx, const int incy)

{

    int i = get_global_id(0);

    if (i < n)

    {

=======

# saxpy OneAPI DPC++ example

# This example demonstrates a basic SAXPY (Scalar A times X plus Y) computation

# using DPC++ via the OneAPI.jl package.

=======

# saxpy OneAPI DPC++ version

# SAXPY: y = a * x + y

=======

# saxpy OneAPI Julia version


# Prompt: saxpy KernelAbstractions
Synthesizing 10/10 solutions

=======

# saxpy KernelAbstractions kernel

function saxpy!(a, x, y)

@kernel function saxpy_kernel!(a, x, y)

@inbounds y[i] = a * x[i] + y[i]

end

return saxpy_kernel!(a, x, y)

end

=======

# saxpy KernelAbstractions.jl implementation

function saxpy_kernel!(A, B, C, α, N)

    @kernel function saxpy_kernel!(A, B, C, α, N)

        i = @index(Global, Linear)

        @inbounds C[i] = α * A[i] + B[i]

    end

    saxpy_kernel!(A, B, C, α, N)

end

=======

# saxpy KernelAbstractions.jl implementation

function saxpy_kernel!(A, B, C, α, N)

@kernel function saxpy_kernel!(A, B, C, α, N)

@inbounds I = @index(Global, Linear)

if I < N

C[I] = A[I] + α * B[I]

end

end

end

# saxpy KernelAbstractions.jl implementation

function saxpy_kernel!(A, B, C, α, N)

@kernel function saxpy_kernel!(A, B, C, α, N)

@inbounds I = @index(Global, Linear)

if I < N

C[I] = A[I] + α * B[I]

end

end

end

# saxpy KernelAbstractions.jl implementation

function saxpy_kernel!(A, B, C, α, N)

@kernel function saxpy_kernel!(A, B, C, α, N)

@inbounds I = @index(Global, Linear)

if I < N

C[I] = A[I] + α * B[I]

end

end

end

# saxpy KernelAbstractions.jl implementation

function saxpy_kernel!(A, B, C, α, N)

@kernel function saxpy_kernel!(A, B, C, α, N)

@inbounds I = @index(Global, Linear)

if I < N

C[I] = A[I] + α * B[I]

end

end

end

# saxpy KernelAbstractions.jl implementation

function saxpy_kernel!(A, B, C, α, N)

@kernel function saxpy_kernel!(A, B, C, α, N)

@inbounds I = @index(Global, Linear)

if I < N

C[I] = A[I] + α * B[I]

end

end

end

# saxpy KernelAbstractions.jl implementation

function saxpy_kernel!(A, B, C, α, N)

@kernel function saxpy_kernel!(A, B, C, α, N)

@inbounds I = @index(Global, Linear)

if I < N

C[I] = A[I] + α * B[I]

end

end

end

# saxpy KernelAbstractions.jl implementation

function saxpy_kernel!(A, B, C, α, N)

@kernel function saxpy_kernel!(A, B, C, α, N)

@in

=======

# saxpy KernelAbstractions.jl

function saxpy_kernel!(n::Int, a::Float64, x::AbstractVector, y::AbstractVector)

@inbounds @kernel for i in 1:n

y[i] += a * x[i]

end

end

function saxpy(n::Int, a::Float64, x::AbstractVector, y::AbstractVector)

saxpy_kernel!(n, a, x, y)

end

# saxpy Julia Base

function saxpy_julia!(n::Int, a::Float64, x::AbstractVector, y::AbstractVector)

@inbounds @simd for i in 1:n

y[i] += a * x[i]

end

end

function saxpy_julia(n::Int, a::Float64, x::AbstractVector, y::AbstractVector)

saxpy_julia!(n, a, x, y)

end

# saxpy Cuda

function saxpy_cuda!(n::Int, a::Float64, x::CuArray, y::CuArray)

@cuda threads=n saxpy_kernel!(n, a, x, y)

end

function saxpy_cuda(n::Int, a::Float64, x::CuArray, y::CuArray)

saxpy_cuda!(n, a, x, y)

end

# saxpy Cuda C

function saxpy_cudac!(n::Int, a::Float64, x::CuArray, y::CuArray)

@cuda threads=n saxpy_kernel_c!(n, a, x, y)

end

function saxpy_cudac(n::Int, a::Float64, x::CuArray, y::CuArray)

saxpy_cudac!(n, a, x, y)

end

# saxpy OpenCL

function saxpy_opencl!(n::Int, a::Float64, x::OpenCL.clarray, y::OpenCL.clarray)

@opencl function saxpy_kernel!(n::Int, a::Float64, x::OpenCL.clarray, y::OpenCL.clarray)

@inbounds @kernel for i in 1:n

y[i] += a * x[i]

end

end

saxpy_kernel!(n, a, x, y)

end

function saxpy_opencl(n::Int, a::Float64, x::OpenCL.clarray, y::OpenCL.clarray)

s

=======

# saxpy KernelAbstractions.jl

function saxpy_kernel!(A, B, C, α, N)

    @inbounds @kernel for i in 1:N

        C[i] = A[i] * α + B[i]

    end

end

# saxpy CUDA.jl

function saxpy_cuda!(A, B, C, α, N)

    saxpy_kernel!(CUDADevice(), N; ndrange=N)(A, B, C, α, N)

end

# saxpy OpenCL.jl

function saxpy_opencl!(A, B, C, α, N)

    saxpy_kernel!(OpenCLDevice(), N; ndrange=N)(A, B, C, α, N)

end

# saxpy ROCm.jl

function saxpy_rocm!(A, B, C, α, N)

    saxpy_kernel!(ROCmDevice(), N; ndrange=N)(A, B, C, α, N)

end

# saxpy KernelAbstractions.jl

function saxpy_kernel!(A, B, C, α, N)

    @inbounds @kernel for i in 1:N

        C[i] = A[i] * α + B[i]

    end

end

# saxpy CUDA.jl

function saxpy_cuda!(A, B, C, α, N)

    saxpy_kernel!(CUDADevice(), N; ndrange=N)(A, B, C, α, N)

end

# saxpy OpenCL.jl

function saxpy_opencl!(A, B, C, α, N)

    saxpy_kernel!(OpenCLDevice(), N; ndrange=N)(A, B, C, α, N)

end

# saxpy ROCm.jl

function saxpy_rocm!(A, B, C, α, N)

    saxpy_kernel!(ROCmDevice(), N; ndrange=N)(A, B, C, α, N)

end

# saxpy KernelAbstractions.jl

function saxpy_kernel!(A, B, C, α, N)

    @inbounds @kernel for i in 1:N

        C[i] = A[i] * α + B[i]

    end

end

# saxpy CUDA.jl

function saxpy_cuda!(A, B, C, α, N)

    saxpy_kernel!(CUDADevice(), N;

=======

# saxpy KernelAbstractions version

function saxpy_kernel!(y, x, a, N)  
@inbounds for i in (1:N)  
y[i] = a * x[i] + y[i]  
end  
end

function saxpy_kernel!(y, x, a, N)  
@inbounds @kernel function saxpy_kernel!(y, x, a, N)  
i = @index(Global, Linear)  
if i ≤ N  
y[i] = a * x[i] + y[i]  
end  
end  
end

function saxpy_kernel!(y, x, a, N)  
@inbounds @kernel function saxpy_kernel!(y, x, a, N)  
i = @index(Global, Linear)  
if i ≤ N  
y[i] = a * x[i] + y[i]  
end  
end  
end

function saxpy_kernel!(y, x, a, N)  
@inbounds @kernel function saxpy_kernel!(y, x, a, N)  
i = @index(Global, Linear)  
if i ≤ N  
y[i] = a * x[i] + y[i]  
end  
end  
end

function saxpy_kernel!(y, x, a, N)  
@inbounds @kernel function saxpy_kernel!(y, x, a, N)  
i = @index(Global, Linear)  
if i ≤ N  
y[i] = a * x[i] + y[i]  
end  
end  
end

function saxpy_kernel!(y, x, a, N)  
@inbounds @kernel function saxpy_kernel!(y, x, a, N)  
i = @index(Global, Linear)  
if i ≤ N  
y[i] = a * x[i] + y[i]  
end  
end  
end

function saxpy_kernel!(y, x, a, N)  
@inbounds @kernel function saxpy_kernel!(y, x, a, N)  
i = @index(Global, Linear)  
if i ≤ N  
y[i] = a * x[i] + y[i]  
end  
end  
end

function saxpy_kernel!(y, x, a, N)  
@inbounds @kernel function saxpy_kernel!(y, x, a, N)  
i = @index(Global, Linear)  
if i ≤ N  
y[i] = a *

=======

# saxpy KernelAbstractions.jl

# Path: SAXPY/KernelAbstractions/SAXPY.jl

=======

# saxpy KernelAbstractions.jl version

function saxpy_kernel!(A, B, C, α, N)

@assert N == length(A) == length(B) == length(C)

@assert α isa Number

@assert N isa Int

@assert N > 0

@assert isa(A, Vector{Float32}) && isa(B, Vector{Float32}) && isa(C, Vector{Float32})

@assert isa(α, Float32)

@assert isa(N, Int32)

@kernel function saxpy_kernel!(A, B, C, α, N)

@uniform begin

α = α

N = N

end

@inbounds @parallel for i in 1:N

C[i] = α * A[i] + B[i]

end

end

saxpy_kernel!(A, B, C, α, N)

end

# saxpy KernelAbstractions.jl version

function saxpy_kernel!(A, B, C, α, N)

@assert N == length(A) == length(B) == length(C)

@assert α isa Number

@assert N isa Int

@assert N > 0

@assert isa(A, Vector{Float64}) && isa(B, Vector{Float64}) && isa(C, Vector{Float64})

@assert isa(α, Float64)

@assert isa(N, Int64)

@kernel function saxpy_kernel!(A, B, C, α, N)

@uniform begin

α = α

N = N

end

@inbounds @parallel for i in 1:N

C[i] = α * A[i] + B[i]

end

end

saxpy_kernel!(A, B, C, α, N)

end

# saxpy KernelAbstractions.jl version

function saxpy_kernel!(A, B, C, α, N)

@assert N == length(A) == length(B) == length(C)

@assert α isa Number

@assert N isa Int

@assert N > 0

@assert isa(A, Vector{Float32}) && isa(B, Vector{Float32}) && isa(C, Vector{Float32})

@assert isa(α, Float32)

@assert isa(N, Int32)

@kernel function saxpy_kernel!(A, B, C, α, N)

@uniform begin

α = α

N = N

end

@inbounds @parallel for i in

=======

# saxpy KernelAbstractions.jl

using KernelAbstractions

using CUDA

using BenchmarkTools

using Test

# saxpy KernelAbstractions.jl

function saxpy!(n, a, x, y)

    kernel! = @kernel function saxpy!(n, a, x, y)

        i = @index(Global)

        if i ≤ n

            y[i] += a * x[i]

        end

    end

    kernel!(n, a, x, y)

end

# saxpy KernelAbstractions.jl

function saxpy!(n, a, x, y)

    kernel! = @kernel function saxpy!(n, a, x, y)

        i = @index(Global)

        if i ≤ n

            y[i] += a * x[i]

        end

    end

    kernel!(n, a, x, y)

end

# saxpy KernelAbstractions.jl

function saxpy!(n, a, x, y)

    kernel! = @kernel function saxpy!(n, a, x, y)

        i = @index(Global)

        if i ≤ n

            y[i] += a * x[i]

        end

    end

    kernel!(n, a, x, y)

end

# saxpy KernelAbstractions.jl

function saxpy!(n, a, x, y)

    kernel! = @kernel function saxpy!(n, a, x, y)

        i = @index(Global)

        if i ≤ n

            y[i] += a * x[i]

        end

    end

    kernel!(n, a, x, y)

end

# saxpy KernelAbstractions.jl

function saxpy!(n, a, x, y)

    kernel! = @kernel function saxpy!(n, a, x, y)

        i = @index(Global)

        if i ≤ n

            y[i] += a * x[i]

        end

    end

    kernel!(n, a, x, y)

end

# saxpy KernelAbstractions.jl

function saxpy!(n, a, x, y)

    kernel! = @kernel function saxpy!(n, a, x, y)

        i = @index(Global)

        if i ≤ n

            y[i] += a * x[i]

        end

    end

    kernel!(n, a, x,

=======

# saxpy KernelAbstractions.jl

module SAXPY

using KernelAbstractions

using CUDA

using LinearAlgebra

using Test

using BenchmarkTools

using Printf

using Statistics
