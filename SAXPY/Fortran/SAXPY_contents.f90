
! PROMPT: saxpy

! saxpy 1.0.0
! =================
!
! This is a simple program that performs the operation y := a*x + y
! where 'x' and 'y' are vectors and 'a' is a scalar. The data type
! of the vectors and scalar is double precision. The length of the
! vectors is specified by the parameter 'n'.
!
! This program is intended to illustrate how to use the OpenMP API
! for parallel programming in Fortran. It is not optimized for
! performance.
!
! This file is formatted for use with 'fprettify'.
!
!===============================================================================

!-------------------------------------------------------------------------------
! 1. Include the OpenMP API header file.
!-------------------------------------------------------------------------------

   use omp_lib

!-------------------------------------------------------------------------------
! 2. Define the problem size.
!-------------------------------------------------------------------------------

   integer, parameter :: n = 100000000

!-------------------------------------------------------------------------------
! 3. Declare and initialize the variables used in the calculations.
!-------------------------------------------------------------------------------

   double precision :: a, x(n), y(n)
   integer          :: i

   a = 3.0d0

   do i = 1, n
      x(i) = 1.0d0
      y(i) = 2.0d0
   end do

!-------------------------------------------------------------------------------
! 4. Create a parallel region and perform the saxpy operation.
!-------------------------------------------------------------------------------

   !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
      !$OMP DO
         do i = 1, n
            y(i) = a*x(i) + y(i)
         end do
      !$OMP END DO
   !$OMP END PARALLEL

!-------------------------------------------------------------------------------
! 5. Print the results.
!-------------------------------------------------------------------------------

   print *, 'y(1) = ', y(1)
   print *, 'y(n) = ', y(n)

!-------------------------------------------------------------------------------
! 6. End the program.
!-------------------------------------------------------------------------------

   end

!===============================================================================
! End of program.
!===============================================================================

!===============================================================================
! End of file.
!===============================================================================

Output

$ f95 saxpy.f95
$ ./a.out
 y(1) =   5.0000000000000000
 y(n) =   3.0000000000000000E+299

Example 2: Parallelizing a Loop

This example shows how to parallelize a loop using the OpenMP API for Fortran.


! Prompt: saxpy OpenMP

Synthesizing 9/10 solutions

=======

! saxpy OpenMP 1.0

! saxpy OpenMP 2.0

! saxpy OpenMP 3.0

! saxpy OpenMP 4.0

! saxpy OpenMP 5.0

! saxpy OpenMP 6.0

! saxpy OpenMP 7.0

! saxpy OpenMP 8.0

! saxpy OpenMP 9.0

! saxpy OpenMP 10.0

! saxpy OpenMP 11.0

! saxpy OpenMP 12.0

! saxpy OpenMP 13.0

! saxpy OpenMP 14.0

! saxpy OpenMP 15.0

! saxpy OpenMP 16.0

! saxpy OpenMP 17.0

! saxpy OpenMP 18.0

! saxpy OpenMP 19.0

! saxpy OpenMP 20.0

! saxpy OpenMP 21.0

! saxpy OpenMP 22.0

! saxpy OpenMP 23.0

! saxpy OpenMP 24.0

! saxpy OpenMP 25.0

! saxpy OpenMP 26.0

! saxpy OpenMP 27.0

! saxpy OpenMP 28.0

! saxpy OpenMP 29.0

! saxpy OpenMP 30.0

! saxpy OpenMP 31.0

! saxpy OpenMP 32.0

! saxpy OpenMP 33.0

! saxpy OpenMP 34.0

! saxpy OpenMP 35.0

! saxpy OpenMP 36.0

! saxpy OpenMP 37.0

! saxpy OpenMP 38.0

! saxpy OpenMP 39.0

! saxpy OpenMP 40.0

! saxpy OpenMP 41.0

! saxpy OpenMP 42.0

! saxpy OpenMP 43.0

! saxpy OpenMP 44.0

! saxpy OpenMP 45.0

! saxpy OpenMP 46.0

! saxpy OpenMP 47.0

! saxpy OpenMP 48.0

! saxpy OpenMP 49.0

! saxpy OpenMP 50.0

! saxpy Open

=======

! saxpy OpenMP

! saxpy OpenACC

! saxpy MPI

! saxpy MPI + OpenMP

! saxpy MPI + OpenACC

! saxpy MPI + OpenMP + OpenACC

! saxpy MPI + OpenMP + OpenACC + OpenCL

! saxpy MPI + OpenMP + OpenACC + OpenCL + CUDA

! saxpy MPI + OpenMP + OpenACC + OpenCL + CUDA + HIP

! saxpy MPI + OpenMP + OpenACC + OpenCL + CUDA + HIP + SYCL

! saxpy MPI + OpenMP + OpenACC + OpenCL + CUDA + HIP + SYCL + DPC++

! saxpy MPI + OpenMP + OpenACC + OpenCL + CUDA + HIP + SYCL + DPC++ + OpenSHMEM

! saxpy MPI + OpenMP + OpenACC + OpenCL + CUDA + HIP + SYCL + DPC++ + OpenSHMEM + Xeon Phi

! saxpy MPI + OpenMP + OpenACC + OpenCL + CUDA + HIP + SYCL + DPC++ + OpenSHMEM + Xeon Phi + FPGA

! saxpy MPI + OpenMP + OpenACC + OpenCL + CUDA + HIP + SYCL + DPC++ + OpenSHMEM + Xeon Phi + FPGA + NVLink

! saxpy MPI + OpenMP + OpenACC + OpenCL + CUDA + HIP + SYCL + DPC++ + OpenSHMEM + Xeon Phi + FPGA + NVLink + SHMEM

! saxpy MPI + OpenMP + OpenACC + OpenCL + CUDA + HIP + SYCL + DPC++ + OpenSHMEM + Xeon Phi + FPGA + NVLink + SHMEM + HBM

! saxpy MPI + OpenMP + OpenACC + OpenCL + CUDA + HIP + SYCL + DPC++ + OpenSHMEM + Xeon Phi + FPGA + NVLink + SHMEM + HBM + FPGAs

! saxpy MPI + OpenMP + OpenACC + OpenCL + CUDA + HIP + SYCL + DPC++ + OpenSHMEM + Xeon Phi + FPGA + NVLink + SHMEM + HBM + FPGAs + GPUs

! saxpy MPI + OpenMP + OpenACC + OpenCL + CUDA + HIP + SYCL + DPC++ + OpenSHMEM + Xeon Phi + FPGA + NVLink + SHMEM + HBM + FPGAs + GPUs

=======

! saxpy OpenMP 4.5

! saxpy OpenMP 5.0

! saxpy OpenACC

! saxpy OpenACC 2.0

! saxpy OpenACC 2.5

! saxpy OpenACC 2.7

! saxpy OpenACC 2.8

! saxpy OpenACC 3.0

! saxpy OpenACC 3.1

! saxpy OpenACC 3.1 with OpenMP 4.5

! saxpy OpenACC 3.1 with OpenMP 5.0

! saxpy OpenACC 3.1 with OpenACC 2.0

! saxpy OpenACC 3.1 with OpenACC 2.5

! saxpy OpenACC 3.1 with OpenACC 2.7

! saxpy OpenACC 3.1 with OpenACC 2.8

! saxpy OpenACC 3.1 with OpenACC 3.0

! saxpy OpenACC 3.1 with OpenACC 3.1

! saxpy OpenACC 3.1 with OpenACC 3.1 with OpenMP 4.5

! saxpy OpenACC 3.1 with OpenACC 3.1 with OpenMP 5.0

! saxpy OpenACC 3.1 with OpenACC 3.1 with OpenACC 2.0

! saxpy OpenACC 3.1 with OpenACC 3.1 with OpenACC 2.5

! saxpy OpenACC 3.1 with OpenACC 3.1 with OpenACC 2.7

! saxpy OpenACC 3.1 with OpenACC 3.1 with OpenACC 2.8

! saxpy OpenACC 3.1 with OpenACC 3.1 with OpenACC 3.0

! saxpy OpenACC 3.1 with OpenACC 3.1 with OpenACC 3.1

! saxpy OpenACC 3.1 with OpenACC 3.1 with OpenACC 3.1 with OpenMP 4.5

! saxpy OpenACC 3.1 with OpenACC 3.1 with OpenACC 3.1 with OpenMP 5.0

! saxpy OpenACC 3.1 with OpenACC 3.1 with OpenACC 3.1 with OpenACC 2

=======

! saxpy OpenMP

! saxpy MPI

! saxpy MPI + OpenMP

! saxpy MPI + OpenMP + CUDA

! saxpy MPI + OpenMP + CUDA + OpenACC

! saxpy MPI + OpenMP + CUDA + OpenACC + Xeon Phi

! saxpy MPI + OpenMP + CUDA + OpenACC + Xeon Phi + GPU

! saxpy MPI + OpenMP + CUDA + OpenACC + Xeon Phi + GPU + FPGA

! saxpy MPI + OpenMP + CUDA + OpenACC + Xeon Phi + GPU + FPGA + FPGA

! saxpy MPI + OpenMP + CUDA + OpenACC + Xeon Phi + GPU + FPGA + FPGA + FPGA

! saxpy MPI + OpenMP + CUDA + OpenACC + Xeon Phi + GPU + FPGA + FPGA + FPGA + FPGA

! saxpy MPI + OpenMP + CUDA + OpenACC + Xeon Phi + GPU + FPGA + FPGA + FPGA + FPGA + FPGA

! saxpy MPI + OpenMP + CUDA + OpenACC + Xeon Phi + GPU + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA

! saxpy MPI + OpenMP + CUDA + OpenACC + Xeon Phi + GPU + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA

! saxpy MPI + OpenMP + CUDA + OpenACC + Xeon Phi + GPU + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA

! saxpy MPI + OpenMP + CUDA + OpenACC + Xeon Phi + GPU + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA

! saxpy MPI + OpenMP + CUDA + OpenACC + Xeon Phi + GPU + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA

! saxpy MPI + OpenMP + CUDA + OpenACC + Xeon Phi + GPU + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA

! saxpy MPI + OpenMP + CUDA + OpenACC + Xeon Phi + GPU + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA

! saxpy MPI + OpenMP + CUDA + OpenACC + Xeon Phi + GPU + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA + FPGA

=======

! saxpy OpenMP parallel for reduction(+:sum) do i=1,n sum = sum + a(i)*x(i) enddo ! saxpy OpenMP parallel for reduction(+:sum) do i=1,n sum = sum + a(i)*x(i) enddo

! saxpy OpenMP parallel do reduction(+:sum) do i=1,n sum = sum + a(i)*x(i) enddo ! saxpy OpenMP parallel do reduction(+:sum) do i=1,n sum = sum + a(i)*x(i) enddo

! saxpy OpenMP parallel do default(none) shared(a,x,n) reduction(+:sum) do i=1,n sum = sum + a(i)*x(i) enddo ! saxpy OpenMP parallel do default(none) shared(a,x,n) reduction(+:sum) do i=1,n sum = sum + a(i)*x(i) enddo

! saxpy OpenMP parallel do default(none) shared(a,x,n) private(i) reduction(+:sum) do i=1,n sum = sum + a(i)*x(i) enddo ! saxpy OpenMP parallel do default(none) shared(a,x,n) private(i) reduction(+:sum) do i=1,n sum = sum + a(i)*x(i) enddo

! saxpy OpenMP parallel do default(none) shared(a,x,n) private(i) reduction(+:sum) do i=1,n sum = sum + a(i)*x(i) enddo ! saxpy OpenMP parallel do default(none) shared(a,x,n) private(i) reduction(+:sum) do i=1,n sum = sum + a(i)*x(i) enddo

! saxpy OpenMP parallel do default(none) shared(a,x,n) private(i) reduction(+:sum) do i=1,n sum = sum + a(i)*x(i) enddo ! saxpy OpenMP parallel do default(none) shared(a,x,n) private(i) reduction(+:sum) do i=1,n sum = sum + a(i)*x(i) enddo

! saxpy OpenMP parallel do default(none) shared(a,x,n) private(i) reduction(+:sum) do i=1,n sum = sum + a(i)*x(i) enddo ! saxpy OpenMP parallel do default(none) shared(a,x,n) private(i)

=======

! saxpy OpenMP

! saxpy MPI

! saxpy MPI + OpenMP

! saxpy MPI + OpenMP + MPI

! saxpy MPI + OpenMP + MPI + OpenMP

! saxpy MPI + OpenMP + MPI + OpenMP + MPI

! saxpy MPI + OpenMP + MPI + OpenMP + MPI + OpenMP

! saxpy MPI + OpenMP + MPI + OpenMP + MPI + OpenMP + MPI

! saxpy MPI + OpenMP + MPI + OpenMP + MPI + OpenMP + MPI + OpenMP

! saxpy MPI + OpenMP + MPI + OpenMP + MPI + OpenMP + MPI + OpenMP + MPI

! saxpy MPI + OpenMP + MPI + OpenMP + MPI + OpenMP + MPI + OpenMP + MPI + OpenMP

! saxpy MPI + OpenMP + MPI + OpenMP + MPI + OpenMP + MPI + OpenMP + MPI + OpenMP + MPI

! saxpy MPI + OpenMP + MPI + OpenMP + MPI + OpenMP + MPI + OpenMP + MPI + OpenMP + MPI + OpenMP

! saxpy MPI + OpenMP + MPI + OpenMP + MPI + OpenMP + MPI + OpenMP + MPI + OpenMP + MPI + OpenMP + MPI

! saxpy

=======

! saxpy OpenMP

! saxpy Cuda

! saxpy OpenCL

! saxpy OpenCL Intel

! saxpy OpenCL AMD

! saxpy OpenACC

! saxpy OpenACC Nvidia

! saxpy OpenACC AMD

! saxpy OpenACC Intel

! saxpy MPI

! saxpy MPI OpenMP

! saxpy MPI OpenACC

! saxpy MPI OpenACC Nvidia

! saxpy MPI OpenACC AMD

! saxpy MPI OpenACC Intel

! saxpy MPI OpenCL

! saxpy MPI OpenCL Intel

! saxpy MPI OpenCL AMD

! saxpy MPI Cuda

! saxpy MPI OpenMP OpenACC

! saxpy MPI OpenMP OpenACC Nvidia

! saxpy MPI OpenMP OpenACC AMD

! saxpy MPI OpenMP OpenACC Intel

! saxpy MPI OpenMP OpenCL

! saxpy MPI OpenMP OpenCL Intel

! saxpy MPI OpenMP OpenCL AMD

! saxpy MPI OpenMP Cuda

! saxpy MPI OpenACC OpenCL

! saxpy MPI OpenACC OpenCL Intel

! saxpy MPI OpenACC OpenCL AMD

! saxpy MPI OpenACC Cuda

! saxpy MPI OpenACC Nvidia OpenCL

! saxpy MPI OpenACC Nvidia OpenCL Intel

! saxpy MPI OpenACC Nvidia OpenCL AMD

! saxpy MPI OpenACC Nvidia Cuda

! saxpy MPI OpenACC AMD OpenCL

! saxpy MPI OpenACC AMD OpenCL Intel

! saxpy MPI OpenACC AMD OpenCL AMD

! saxpy MPI OpenACC AMD Cuda

! saxpy MPI OpenACC Intel OpenCL

! saxpy MPI OpenACC Intel OpenCL Intel

! saxpy MPI OpenACC Intel OpenCL AMD

! saxpy MPI OpenACC Intel Cuda

! saxpy MPI OpenCL OpenCL Intel

! saxpy MPI OpenCL OpenCL AMD

! saxpy MPI OpenCL Cuda

! saxpy MPI Cuda Cuda

! saxpy MPI OpenMP OpenACC OpenCL

! saxpy MPI OpenMP OpenACC OpenCL Intel

! saxpy MPI OpenMP OpenACC OpenCL AMD

! saxpy MPI OpenMP OpenACC Cuda

! saxpy MPI OpenMP OpenACC Nvidia OpenCL

! saxpy MPI OpenMP OpenACC Nvidia OpenCL Intel

! saxpy MPI OpenMP OpenACC Nvidia OpenCL AMD

! saxpy MPI OpenMP OpenACC Nvidia

=======

! saxpy OpenMP:

=======

! saxpy OpenMP ve


! Prompt: saxpy OpenMP
Synthesizing 8/10 solutions

=======

! saxpy OpenMP 1.0

! saxpy OpenMP 2.0

! saxpy OpenMP 3.0

! saxpy OpenMP 4.0

! saxpy OpenMP 5.0

! saxpy OpenMP 6.0

! saxpy OpenMP 7.0

! saxpy OpenMP 8.0

! saxpy OpenMP 9.0

! saxpy OpenMP 10.0

! saxpy OpenMP 11.0

! saxpy OpenMP 12.0

! saxpy OpenMP 13.0

! saxpy OpenMP 14.0

! saxpy OpenMP 15.0

! saxpy OpenMP 16.0

! saxpy OpenMP 17.0

! saxpy OpenMP 18.0

! saxpy OpenMP 19.0

! saxpy OpenMP 20.0

! saxpy OpenMP 21.0

! saxpy OpenMP 22.0

! saxpy OpenMP 23.0

! saxpy OpenMP 24.0

! saxpy OpenMP 25.0

! saxpy OpenMP 26.0

! saxpy OpenMP 27.0

! saxpy OpenMP 28.0

! saxpy OpenMP 29.0

! saxpy OpenMP 30.0

! saxpy OpenMP 31.0

! saxpy OpenMP 32.0

! saxpy OpenMP 33.0

! saxpy OpenMP 34.0

! saxpy OpenMP 35.0

! saxpy OpenMP 36.0

! saxpy OpenMP 37.0

! saxpy OpenMP 38.0

! saxpy OpenMP 39.0

! saxpy OpenMP 40.0

! saxpy OpenMP 41.0

! saxpy OpenMP 42.0

! saxpy OpenMP 43.0

! saxpy OpenMP 44.0

! saxpy OpenMP 45.0

! saxpy OpenMP 46.0

! saxpy OpenMP 47.0

! saxpy OpenMP 48.0

! saxpy OpenMP 49.0

! saxpy OpenMP 50.0

! saxpy Open

=======

! saxpy OpenMP ( 1.0 ) ! saxpy OpenMP ( 2.0 ) ! saxpy OpenMP ( 3.0 ) ! saxpy OpenMP ( 4.0 ) ! saxpy OpenMP ( 5.0 ) ! saxpy OpenMP ( 6.0 ) ! saxpy OpenMP ( 7.0 ) ! saxpy OpenMP ( 8.0 ) ! saxpy OpenMP ( 9.0 ) ! saxpy OpenMP ( 10.0 ) ! saxpy OpenMP ( 11.0 ) ! saxpy OpenMP ( 12.0 ) ! saxpy OpenMP ( 13.0 ) ! saxpy OpenMP ( 14.0 ) ! saxpy OpenMP ( 15.0 ) ! saxpy OpenMP ( 16.0 ) ! saxpy OpenMP ( 17.0 ) ! saxpy OpenMP ( 18.0 ) ! saxpy OpenMP ( 19.0 ) ! saxpy OpenMP ( 20.0 ) ! saxpy OpenMP ( 21.0 ) ! saxpy OpenMP ( 22.0 ) ! saxpy OpenMP ( 23.0 ) ! saxpy OpenMP ( 24.0 ) ! saxpy OpenMP ( 25.0 ) ! saxpy OpenMP ( 26.0 ) ! saxpy OpenMP ( 27.0 ) ! saxpy OpenMP ( 28.0 ) ! saxpy OpenMP ( 29.0 ) ! saxpy OpenMP ( 30.0 ) ! saxpy OpenMP ( 31.0 ) ! saxpy OpenMP ( 32.0 ) ! saxpy OpenMP ( 33.0 ) ! saxpy OpenMP ( 34.0 ) ! saxpy OpenMP ( 35.0 ) ! saxpy OpenMP ( 36.0 ) ! saxpy OpenMP ( 37.0 ) ! saxpy OpenMP ( 38.0 ) ! saxpy OpenMP ( 39.0 ) ! saxpy OpenMP ( 40.0 ) ! saxpy OpenMP ( 41.0 ) ! saxpy OpenMP ( 42.0 ) ! saxpy OpenMP ( 43.0 ) ! saxpy OpenMP ( 44.0 ) ! saxpy OpenMP ( 45.0 ) ! saxpy OpenMP ( 46.

=======

! saxpy OpenMP

! saxpy MPI

! saxpy MPI + OpenMP

! saxpy MPI + OpenMP + CUDA

! saxpy MPI + OpenMP + CUDA + HSA

! saxpy MPI + OpenMP + CUDA + HSA + HIP

! saxpy MPI + OpenMP + CUDA + HSA + HIP + SYCL

! saxpy MPI + OpenMP + CUDA + HSA + HIP + SYCL + OpenACC

! saxpy MPI + OpenMP + CUDA + HSA + HIP + SYCL + OpenACC + OpenCL

! saxpy MPI + OpenMP + CUDA + HSA + HIP + SYCL + OpenACC + OpenCL + TBB

! saxpy MPI + OpenMP + CUDA + HSA + HIP + SYCL + OpenACC + OpenCL + TBB + DPC++

! saxpy MPI + OpenMP + CUDA + HSA + HIP + SYCL + OpenACC + OpenCL + TBB + DPC++ + RAJA

! saxpy MPI + OpenMP + CUDA + HSA + HIP + SYCL + OpenACC + OpenCL + TBB + DPC++ + RAJA + Kokkos

! saxpy MPI + OpenMP + CUDA + HSA + HIP + SYCL + OpenACC + OpenCL + TBB + DPC++ + RAJA + Kokkos + SYCL-BLAS

! saxpy MPI + OpenMP + CUDA + HSA + HIP + SYCL + OpenACC + OpenCL + TBB + DPC++ + RAJA + Kokkos + SYCL-BLAS + SYCL-DNN

! saxpy MPI + OpenMP + CUDA + HSA + HIP + SYCL + OpenACC + OpenCL + TBB + DPC++ + RAJA + Kokkos + SYCL-BLAS + SYCL-DNN + SYCL-NNPACK

! saxpy MPI + OpenMP + CUDA + HSA + HIP + SYCL + OpenACC + OpenCL + TBB + DPC++ + RAJA + Kokkos + SYCL-BLAS + SYCL-DNN + SYCL-NNPACK + SYCL-FFT

! saxpy MPI + OpenMP + CUDA + HSA + HIP + SYCL + OpenACC + OpenCL + TBB + DPC++ + RAJA + Kokkos + SYCL-BLAS + SYCL-DNN + SYCL-NNPACK + SYCL-

=======

! saxpy OpenMP 1 2 4 8 16 32 64 128 256 512

=======

! saxpy OpenMP ! saxpy OpenACC ! saxpy OpenACC with data ! saxpy OpenACC with data and kernels ! saxpy OpenACC with data, kernels and present ! saxpy OpenACC with data, kernels, present and vector ! saxpy OpenACC with data, kernels, present, vector and cache ! saxpy OpenACC with data, kernels, present, vector, cache and collapse ! saxpy OpenACC with data, kernels, present, vector, cache, collapse and reduction ! saxpy OpenACC with data, kernels, present, vector, cache, collapse, reduction and update

! saxpy OpenACC with data, kernels, present, vector, cache, collapse, reduction, update and loop

! saxpy OpenACC with data, kernels, present, vector, cache, collapse, reduction, update, loop and independent

! saxpy OpenACC with data, kernels, present, vector, cache, collapse, reduction, update, loop, independent and device_type

! saxpy OpenACC with data, kernels, present, vector, cache, collapse, reduction, update, loop, independent, device_type and device_num

! saxpy OpenACC with data, kernels, present, vector, cache, collapse, reduction, update, loop, independent, device_type, device_num and gang

! saxpy OpenACC with data, kernels, present, vector, cache, collapse, reduction, update, loop, independent, device_type, device_num, gang and worker

! saxpy OpenACC with data, kernels, present, vector, cache, collapse, reduction, update, loop, independent, device_type, device_num, gang, worker and vector

! saxpy OpenACC with data, kernels, present, vector, cache, collapse, reduction, update, loop, independent, device_type, device_num, gang, worker, vector and seq

! saxpy OpenACC with data, kernels, present, vector, cache, collapse, reduction, update, loop, independent, device_type, device_num, gang, worker, vector, seq and auto

! saxpy OpenACC with data, kernels, present, vector, cache, collapse, reduction, update, loop, independent, device_type, device_num, gang, worker, vector, seq, auto and deviceptr

! saxpy OpenACC with data, kernels, present, vector, cache, collapse, reduction, update, loop, independent, device_type, device_num, gang, worker

=======

! saxpy OpenMP

! s

=======

! saxpy OpenMP 1

=======

! saxpy OpenMP 1 1 0.

! Prompt: saxpy OpenACC\nsubroutine
Synthesizing 10/10 solutions

=======

subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(n) :: x
  real, intent(inout), dimension(n) :: y

  integer :: i

  !$acc parallel loop copyin(x) copy(y)
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$acc end parallel loop
end subroutine saxpy

! saxpy OpenMP
subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(n) :: x
  real, intent(inout), dimension(n) :: y

  integer :: i

  !$omp parallel do
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$omp end parallel do
end subroutine saxpy

! saxpy OpenMP with SIMD
subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(n) :: x
  real, intent(inout), dimension(n) :: y

  integer :: i

  !$omp parallel do simd
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$omp end parallel do simd
end subroutine saxpy

! saxpy OpenMP with SIMD, aligned
subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(n) :: x
  real, intent(inout), dimension(n) :: y

  integer :: i

  !$omp parallel do simd aligned(x, y: 64)
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$omp end parallel do simd
end subroutine saxpy

! saxpy OpenMP with SIMD, aligned, private
subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a

=======

subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n), intent(inout) :: y
  !$acc parallel loop
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine

! saxpy OpenMP
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n), intent(inout) :: y
  !$omp parallel do
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine

! saxpy OpenMP with SIMD
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n), intent(inout) :: y
  !$omp parallel do simd
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine

! saxpy OpenMP with SIMD and aligned
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n), intent(inout) :: y
  !$omp parallel do simd aligned(x, y: 64)
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine

! saxpy OpenMP with SIMD and aligned, with private variables
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n), intent(inout) :: y
  !$omp parallel do simd private(x, y)
  do i =

=======

subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)

  !$acc parallel loop present(x, y)
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$acc end parallel loop

end subroutine saxpy

! saxpy OpenMP
subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)

  !$omp parallel do
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$omp end parallel do

end subroutine saxpy

! saxpy OpenMP with SIMD
subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)

  !$omp parallel do simd
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$omp end parallel do

end subroutine saxpy

! saxpy OpenMP with SIMD and aligned
subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)

  !$omp parallel do simd aligned(x, y: 32)
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$omp end parallel do

end subroutine saxpy

! saxpy OpenMP with SIMD and aligned
subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)

  !$omp parallel do simd aligned(x, y: 32)
  do i = 1

=======

subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)
  integer :: i
  !$acc parallel loop
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine saxpy

! saxpy OpenMP
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)
  integer :: i
  !$omp parallel do
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine saxpy

! saxpy OpenMP SIMD
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)
  integer :: i
  !$omp parallel do simd
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine saxpy

! saxpy OpenMP SIMD collapse
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)
  integer :: i
  !$omp parallel do simd collapse(2)
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine saxpy

! saxpy OpenMP SIMD collapse
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)
  integer :: i
  !$omp parallel do simd collapse(2)
  do i = 1

=======

subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)
  !$acc parallel loop
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine
```

## 4.4. OpenMP

OpenMP es una API de programación para la computación paralela, que se encuentra implementada en la mayoría de los compiladores modernos. Se puede utilizar en lenguajes como C, C++, Fortran, Python, Java, etc. OpenMP define directivas de compilación y rutinas de biblioteca para la programación paralela de código secuencial. OpenMP es una API de alto nivel, y por lo tanto es más fácil de utilizar que las API de bajo nivel como CUDA y OpenCL. Sin embargo, OpenMP es una API de programación de procesadores compartidos, y por lo tanto no es tan flexible como CUDA y OpenCL, que pueden ser utilizadas para programar tanto procesadores compartidos como procesadores de gráficos.

```fortran
! saxpy OpenMP
subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)
  !$omp parallel do
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine
```

## 4.5. MPI

MPI (Message Passing Interface) es una API de programación para la computación paralela, que se encuentra implementada en la mayoría de los compiladores modernos. Se puede utilizar en lenguajes como C, C++, Fortran, Python, Java, etc. MPI define directivas de compilación y rutinas de biblioteca para la programación paralela de código secuencial. MPI es una API de alto nivel, y por lo tanto es más fácil de utilizar que las API de bajo nivel como CUDA y OpenCL. Sin embargo, MPI es una API de programación de procesadores compartidos, y por lo tanto no es tan flexible como

=======

subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(n) :: x
  real, intent(inout), dimension(n) :: y
  integer :: i

  !$acc parallel loop
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$acc end parallel loop
end subroutine saxpy

! saxpy OpenMP
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(n) :: x
  real, intent(inout), dimension(n) :: y
  integer :: i

  !$omp parallel do
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$omp end parallel do
end subroutine saxpy

! saxpy OpenMP SIMD
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(n) :: x
  real, intent(inout), dimension(n) :: y
  integer :: i

  !$omp parallel do simd
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$omp end parallel do simd
end subroutine saxpy

! saxpy OpenMP SIMD + SIMDLEN
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(n) :: x
  real, intent(inout), dimension(n) :: y
  integer :: i

  !$omp parallel do simd
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$omp end parallel do simd
end subroutine saxpy

! saxpy OpenMP SIMD + SIMDLEN + COLLAPSE
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n

=======

subroutine saxpy(n, a, x, y)
  implicit none
  integer, value :: n
  real, value :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n), intent(inout) :: y
  !$acc kernels
  do i = 1, n
    y(i) = a * x(i) + y(i)
  enddo
  !$acc end kernels
end subroutine saxpy

! saxpy OpenMP
subroutine saxpy(n, a, x, y)
  implicit none
  integer, value :: n
  real, value :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n), intent(inout) :: y
  !$omp parallel do
  do i = 1, n
    y(i) = a * x(i) + y(i)
  enddo
  !$omp end parallel do
end subroutine saxpy

! saxpy OpenACC + OpenMP
subroutine saxpy(n, a, x, y)
  implicit none
  integer, value :: n
  real, value :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n), intent(inout) :: y
  !$acc kernels
  !$omp parallel do
  do i = 1, n
    y(i) = a * x(i) + y(i)
  enddo
  !$omp end parallel do
  !$acc end kernels
end subroutine saxpy

! saxpy OpenACC + OpenMP + MPI
subroutine saxpy(n, a, x, y)
  implicit none
  integer, value :: n
  real, value :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n), intent(inout) :: y
  !$acc kernels
  !$omp parallel do
  do i = 1, n
    y(i) = a * x(i) + y(i)
  enddo
  !$omp end parallel do
  !$acc end kernels
end subroutine saxpy

! saxpy OpenACC + MPI
subroutine saxpy(n, a, x, y)
  implicit none
  integer, value :: n
  real, value :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n),

=======

subroutine saxpy_acc(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n), intent(inout) :: y
  !$acc kernels
  do i = 1, n
    y(i) = a * x(i) + y(i)
  enddo
  !$acc end kernels
end subroutine saxpy_acc

! saxpy OpenMP
subroutine saxpy_omp(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n), intent(inout) :: y
  !$omp parallel do
  do i = 1, n
    y(i) = a * x(i) + y(i)
  enddo
  !$omp end parallel do
end subroutine saxpy_omp

end module saxpy
```

### 2.2.2. 生成されたコードの確認

```fortran
! saxpy OpenACC
subroutine saxpy_acc(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n), intent(inout) :: y
  !$acc kernels
  do i = 1, n
    y(i) = a * x(i) + y(i)
  enddo
  !$acc end kernels
end subroutine saxpy_acc

! saxpy OpenMP
subroutine saxpy_omp(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n), intent(inout) :: y
  !$omp parallel do
  do i = 1, n
    y(i) = a * x(i) + y(i)
  enddo
  !$omp end parallel do
end subroutine saxpy_omp
```

####

=======

subroutine saxpy(n,a,x,y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(1:n) :: x
  real, intent(inout), dimension(1:n) :: y
  integer :: i
  !$acc kernels
  do i=1,n
    y(i) = a*x(i) + y(i)
  end do
  !$acc end kernels
end subroutine

! saxpy OpenMP
subroutine saxpy(n,a,x,y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(1:n) :: x
  real, intent(inout), dimension(1:n) :: y
  integer :: i
  !$omp parallel do shared(n,a,x,y) private(i)
  do i=1,n
    y(i) = a*x(i) + y(i)
  end do
  !$omp end parallel do
end subroutine

! saxpy OpenMP with SIMD
subroutine saxpy(n,a,x,y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(1:n) :: x
  real, intent(inout), dimension(1:n) :: y
  integer :: i
  !$omp parallel do shared(n,a,x,y) private(i) simd
  do i=1,n
    y(i) = a*x(i) + y(i)
  end do
  !$omp end parallel do
end subroutine

! saxpy OpenMP with SIMD and aligned
subroutine saxpy(n,a,x,y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(1:n) :: x
  real, intent(inout), dimension(1:n) :: y
  integer :: i
  !$omp parallel do shared(n,a,x,y) private(i) simd aligned(x,y)
  do i=1,n
    y(i) = a*x(i) + y(i)
  end do
  !$omp end parallel do
end subroutine

! saxpy OpenMP with SIMD and aligned and linear
subroutine saxpy(n,a,x,y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in),

=======

subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real(4), intent(in) :: a
  real(4), intent(in) :: x(0:n-1)
  real(4), intent(inout) :: y(0:n-1)
  !$acc parallel loop
  do i = 0, n-1
    y(i) = a * x(i) + y(i)
  end do
end subroutine
```

###

! Prompt: saxpy OpenMP offload\nsubroutine
Synthesizing 10/10 solutions

=======

subroutine saxpy(n, a, x, incx, y, incy)
  implicit none
  integer, intent(in) :: n, incx, incy
  real, intent(in) :: a
  real, intent(in), dimension(0:n-1) :: x
  real, intent(inout), dimension(0:n-1) :: y
  !$acc parallel loop present(x, y) async
  do i = 0, n-1, 1
    y(i) = a*x(i) + y(i)
  end do
  !$acc wait
end subroutine saxpy

! sgemv OpenMP offload
subroutine sgemv(n, m, alpha, a, lda, x, incx, beta, y, incy)
  implicit none
  integer, intent(in) :: n, m, lda, incx, incy
  real, intent(in) :: alpha, beta
  real, intent(in), dimension(0:lda-1, 0:m-1) :: a
  real, intent(in), dimension(0:n-1) :: x
  real, intent(inout), dimension(0:m-1) :: y
  !$acc parallel loop present(a, x, y) async
  do i = 0, m-1, 1
    y(i) = 0.0
    do j = 0, n-1, 1
      y(i) = alpha*a(j,i)*x(j) + y(i)
    end do
    y(i) = beta*y(i)
  end do
  !$acc wait
end subroutine sgemv

! sgemm OpenMP offload
subroutine sgemm(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
  implicit none
  integer, intent(in) :: m, n, k, lda, ldb, ldc
  real, intent(in) :: alpha, beta
  real, intent(in), dimension(0:lda-1, 0:k-1) :: a
  real, intent(in), dimension(0:ldb-1, 0:n-1) :: b
  real, intent(inout), dimension(0:ldc-1, 0:n-1) :: c
  !$acc parallel loop present(a,

=======

subroutine saxpy(n, a, x, incx, y, incy)
  implicit none
  integer, intent(in) :: n, incx, incy
  real, intent(in) :: a
  real, intent(in), dimension(0:n-1) :: x
  real, intent(inout), dimension(0:n-1) :: y

  !$acc parallel loop
  do i = 0, n-1
    y(i) = a*x(i) + y(i)
  end do
end subroutine

! saxpy OpenACC
subroutine saxpy(n, a, x, incx, y, incy)
  implicit none
  integer, intent(in) :: n, incx, incy
  real, intent(in) :: a
  real, intent(in), dimension(0:n-1) :: x
  real, intent(inout), dimension(0:n-1) :: y

  !$acc parallel loop
  do i = 0, n-1
    y(i) = a*x(i) + y(i)
  end do
end subroutine

! saxpy OpenMP
subroutine saxpy(n, a, x, incx, y, incy)
  implicit none
  integer, intent(in) :: n, incx, incy
  real, intent(in) :: a
  real, intent(in), dimension(0:n-1) :: x
  real, intent(inout), dimension(0:n-1) :: y

  !$omp parallel do
  do i = 0, n-1
    y(i) = a*x(i) + y(i)
  end do
end subroutine

! saxpy OpenMP
subroutine saxpy(n, a, x, incx, y, incy)
  implicit none
  integer, intent(in) :: n, incx, incy
  real, intent(in) :: a
  real, intent(in), dimension(0:n-1) :: x
  real, intent(inout), dimension(0:n-1) :: y

  !$omp parallel do
  do i = 0, n-1
    y(i) = a*x(i) + y(i)
  end do
end subroutine

! saxpy OpenMP SIMD
subroutine saxpy(n, a, x, incx,

=======

subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)

  !$omp target teams distribute parallel for map(to:n, a, x) map(tofrom:y)
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine saxpy

! saxpy OpenMP offload with target data
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)

  !$omp target data map(to:n, a, x) map(tofrom:y)
  !$omp target teams distribute parallel for
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$omp end target data
end subroutine saxpy

! saxpy OpenMP offload with target data and declare target
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)

  !$omp declare target
  subroutine saxpy(n, a, x, y)
    implicit none
    integer, intent(in) :: n
    real, intent(in) :: a
    real, intent(in) :: x(n)
    real, intent(inout) :: y(n)

    !$omp target teams distribute parallel for
    do i = 1, n
      y(i) = a * x(i) + y(i)
    end do
  end subroutine saxpy
  !$omp end declare target

  !$omp target data map(to:n, a, x) map(tofrom:y)
  call saxpy(n, a, x, y)
  !$omp end target data
end subroutine saxpy

! saxpy OpenMP offload with target data and declare target and use
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n

=======

subroutine saxpy(n, a, x, incx, y, incy)
  implicit none
  integer, intent(in) :: n, incx, incy
  real, intent(in) :: a
  real, intent(in), dimension(1:*) :: x
  real, intent(inout), dimension(1:*) :: y
  integer :: i

  !$acc parallel loop vector_length(128)
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine saxpy

=======

subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)

  integer :: i

  !$acc kernels copyin(x) copy(y)
  do i = 1, n
    y(i) = a * x(i) + y(i)
  enddo
  !$acc end kernels
end subroutine saxpy

! saxpy OpenACC
subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)

  integer :: i

  !$acc kernels copyin(x) copy(y)
  do i = 1, n
    y(i) = a * x(i) + y(i)
  enddo
  !$acc end kernels
end subroutine saxpy

! saxpy OpenACC
subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)

  integer :: i

  !$acc kernels copyin(x) copy(y)
  do i = 1, n
    y(i) = a * x(i) + y(i)
  enddo
  !$acc end kernels
end subroutine saxpy

! saxpy OpenACC
subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)

  integer :: i

  !$acc kernels copyin(x) copy(y)
  do i = 1, n
    y(i) = a * x(i) + y(i)
  enddo
  !$acc end kernels
end subroutine saxpy

! saxpy OpenACC
subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)

  integer :: i

  !$

=======

subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(n) :: x
  real, intent(inout), dimension(n) :: y
  !$acc kernels loop gang vector
  do i = 1, n
    y(i) = a*x(i) + y(i)
  end do
end subroutine

! saxpy OpenMP offload
subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(n) :: x
  real, intent(inout), dimension(n) :: y
  !$acc kernels loop gang vector
  do i = 1, n
    y(i) = a*x(i) + y(i)
  end do
end subroutine

! saxpy OpenMP offload
subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(n) :: x
  real, intent(inout), dimension(n) :: y
  !$acc kernels loop gang vector
  do i = 1, n
    y(i) = a*x(i) + y(i)
  end do
end subroutine

! saxpy OpenMP offload
subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(n) :: x
  real, intent(inout), dimension(n) :: y
  !$acc kernels loop gang vector
  do i = 1, n
    y(i) = a*x(i) + y(i)
  end do
end subroutine

! saxpy OpenMP offload
subroutine saxpy(n, a, x, y)
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(n) :: x
  real, intent(inout), dimension(n) :: y
  !$acc kernels loop gang vector
  do i = 1, n
    y(i) = a*x(i) + y(i)
  end do
end subroutine

! saxpy OpenMP offload
subroutine

=======

subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(n) :: x
  real, intent(inout), dimension(n) :: y

  !$acc kernels copyin(x) copy(y)
  !$acc loop independent
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$acc end kernels
end subroutine saxpy

! saxpy OpenACC (not offload)
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(n) :: x
  real, intent(inout), dimension(n) :: y

  !$acc kernels copyin(x) copy(y)
  !$acc loop independent
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$acc end kernels
end subroutine saxpy

! saxpy OpenMP
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(n) :: x
  real, intent(inout), dimension(n) :: y

  !$omp parallel do
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$omp end parallel do
end subroutine saxpy

! saxpy OpenMP SIMD
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in), dimension(n) :: x
  real, intent(inout), dimension(n) :: y

  !$omp simd
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine saxpy

! saxpy OpenMP SIMD with OpenACC data
subroutine saxpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in),

=======

subroutine saxpy(n,a,x,y)
  implicit none
  integer, intent(in) :: n
  real*4, intent(in) :: a
  real*4, intent(in) :: x(n)
  real*4, intent(inout) :: y(n)
  integer :: i
  !$omp target teams distribute parallel for map(to: x, y) map(from: a)
  do i = 1, n
    y(i) = a*x(i) + y(i)
  enddo
end subroutine

! saxpy OpenMP offload with array sections
subroutine saxpy2(n,a,x,y)
  implicit none
  integer, intent(in) :: n
  real*4, intent(in) :: a
  real*4, intent(in) :: x(n)
  real*4, intent(inout) :: y(n)
  integer :: i
  !$omp target teams distribute parallel for map(to: x(1:n), y(1:n)) map(from: a)
  do i = 1, n
    y(i) = a*x(i) + y(i)
  enddo
end subroutine

! saxpy OpenMP offload with array sections
subroutine saxpy3(n,a,x,y)
  implicit none
  integer, intent(in) :: n
  real*4, intent(in) :: a
  real*4, intent(in) :: x(n)
  real*4, intent(inout) :: y(n)
  integer :: i
  !$omp target teams distribute parallel for map(to: x(1:n), y(1:n)) map(from: a)
  do i = 1, n
    y(i) = a*x(i) + y(i)
  enddo
end subroutine

! saxpy OpenMP offload with array sections
subroutine saxpy4(n,a,x,y)
  implicit none
  integer, intent(in) :: n
  real*4, intent(in) :: a
  real*4, intent(in) :: x(n)
  real*4, intent(inout) :: y(n)
  integer :: i
  !$omp target teams distribute parallel for map(to: x(1:n), y(1:n)) map(from: a)
  do i = 1, n
    y(i) = a*x(i) + y(i)
  enddo
end subroutine

=======

subroutine saxpy (n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n), intent(inout) :: y
  integer :: i
  !$omp target teams distribute parallel for map(tofrom: y) map(to: x) reduction(+: y)
  do i = 1, n
    y(i) = y(i) + a * x(i)
  end do
end subroutine saxpy

! saxpy OpenACC
subroutine saxpy (n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n), intent(inout) :: y
  integer :: i
  !$acc kernels loop gang vector reduction(+: y)
  do i = 1, n
    y(i) = y(i) + a * x(i)
  end do
end subroutine saxpy

! saxpy OpenACC with data directive
subroutine saxpy (n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n), intent(inout) :: y
  integer :: i
  !$acc data copyin(x) copy(y)
  !$acc kernels loop gang vector reduction(+: y)
  do i = 1, n
    y(i) = y(i) + a * x(i)
  end do
  !$acc end data
end subroutine saxpy

! saxpy OpenACC with data directive and present clause
subroutine saxpy (n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, dimension(n), intent(in) :: x
  real, dimension(n), intent(inout) :: y
  integer :: i
  !$acc data copyin(x) copy(y)
  !$acc kernels loop gang vector present(x, y) reduction(+: y)
  do i = 1, n
    y(i) = y(i) + a * x(i)
  end do

=======

subroutine saxpy_openmp_offload (n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  real, intent(inout) :: y(n)
  !$omp target teams distribute parallel do map(tofrom: y)
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine saxpy_openmp_offload

! Main program
program saxpy
  implicit none
  integer :: n
  real :: a
  real, allocatable :: x(:), y(:)
  real :: time1, time2
  integer :: i
  n = 100000000
  a = 2.0
  allocate (x(n), y(n))
  do i = 1, n
    x(i) = 1.0
    y(i) = 1.0
  end do
  print *, "Starting saxpy_serial"
  time1 = omp_get_wtime()
  call saxpy_serial(n, a, x, y)
  time2 = omp_get_wtime()
  print *, "Elapsed time (serial): ", time2 - time1
  print *, "Starting saxpy_openmp"
  time1 = omp_get_wtime()
  call saxpy_openmp(n, a, x, y)
  time2 = omp_get_wtime()
  print *, "Elapsed time (OpenMP): ", time2 - time1
  print *, "Starting saxpy_openmp_offload"
  time1 = omp_get_wtime()
  call saxpy_openmp_offload(n, a, x, y)
  time2 = omp_get_wtime()
  print *, "Elapsed time (OpenMP offload): ", time2 - time1
  deallocate (x, y)
end program saxpy

To compile the program, you will need to use the OpenMP Fortran compiler wrapper.

$ ifort -qopenmp -qoffload saxpy.f90 -o saxpy

To run the program, you will need to use the Intel® MPI Library command-line launcher.

$ mpiexec.hydra -n 1 ./saxpy

The output should be similar to the following:

Starting saxpy_serial
Elapsed time (serial):  1

