! Prompt: 3D Jacobi OpenMP
### Beginning of Output
Synthesizing 5/10 solutions

=======

! 3D Jacobi OpenMP C

=======

! 3D Jacobi OpenMP, 3D Jacobi MPI, 3D Jacobi Hybrid, 3D Jacobi OpenMP+MPI, 3D Jacobi MPI+OpenMP, 3D Jacobi MPI+Hybrid, 3D Jacobi Hybrid+MPI, 3D Jacobi Hybrid+OpenMP, 3D Jacobi OpenMP+Hybrid, 3D Jacobi Hybrid+OpenMP, 3D Jacobi MPI+Hybrid+OpenMP, 3D Jacobi OpenMP+Hybrid+MPI, 3D Jacobi Hybrid+OpenMP+MPI, 3D Jacobi Hybrid+MPI+OpenMP, 3D Jacobi MPI+OpenMP+Hybrid, 3D Jacobi OpenMP+MPI+Hybrid, 3D Jacobi Hybrid+MPI+OpenMP, 3D Jacobi Hybrid+OpenMP+MPI, 3D Jacobi MPI+Hybrid+OpenMP, 3D Jacobi OpenMP+Hybrid+MPI, 3D Jacobi Hybrid+OpenMP+MPI, 3D Jacobi Hybrid+MPI+OpenMP, 3D Jacobi MPI+OpenMP+Hybrid, 3D Jacobi OpenMP+MPI+Hybrid, 3D Jacobi Hybrid+MPI+OpenMP, 3D Jacobi Hybrid+OpenMP+MPI, 3D Jacobi MPI+Hybrid+OpenMP, 3D Jacobi OpenMP+Hybrid+MPI, 3D Jacobi Hybrid+OpenMP+MPI, 3D Jacobi Hybrid+MPI+OpenMP, 3D Jacobi MPI+OpenMP+Hybrid, 3D Jacobi OpenMP+MPI+Hybrid, 3D Jacobi Hybrid+MPI+OpenMP, 3D Jacobi Hybrid+OpenMP+MPI, 3D Jacobi MPI+Hybrid+OpenMP, 3D Jacobi OpenMP+Hybrid+MPI, 3D Jacobi Hybrid+OpenMP+MPI, 3D Jacobi Hybrid+MPI+OpenMP, 3D Jacobi MPI+OpenMP+Hybrid, 3D Jacobi OpenMP+MPI+Hybrid, 3D Jacobi Hybrid+MPI+OpenMP, 3D Jacobi Hybrid+OpenMP+MPI, 3D Jacobi MPI+Hybrid+OpenMP,

=======

! 3D Jacobi OpenMP
Jacobi OpenMP
Jacobi MPI
Jacobi MPI+OpenMP
Jacobi MPI+OpenMP+GPU
Jacobi MPI+OpenMP+GPU+MPI+OpenMP
Jacobi MPI+OpenMP+GPU+MPI+OpenMP+MPI+OpenMP
Jacobi MPI+OpenMP+GPU+MPI+OpenMP+MPI+OpenMP+GPU
Jacobi MPI+OpenMP+GPU+MPI+OpenMP+MPI+OpenMP+GPU+MPI+OpenMP
Jacobi MPI+OpenMP+GPU+MPI+OpenMP+MPI+OpenMP+GPU+MPI+OpenMP+GPU
Jacobi MPI+OpenMP+GPU+MPI+OpenMP+MPI+OpenMP+GPU+MPI+OpenMP+GPU+MPI+OpenMP
Jacobi MPI+OpenMP+GPU+MPI+OpenMP+MPI+OpenMP+GPU+MPI+OpenMP+GPU+MPI+OpenMP+GPU
Jacobi MPI+OpenMP+GPU+MPI+OpenMP+MPI+OpenMP+GPU+MPI+OpenMP+GPU+MPI+OpenMP+GPU+MPI+OpenMP
Jacobi MPI+OpenMP+GPU+MPI+OpenMP+MPI+OpenMP+GPU+MPI+OpenMP+GPU+MPI+OpenMP+GPU+MPI+OpenMP+GPU
Jacobi MPI+OpenMP+GPU+MPI+OpenMP+MPI+OpenMP+GPU+MPI+OpenMP+GPU+MPI+OpenMP+GPU+MPI+OpenMP+GPU+MPI+OpenMP
Jacobi MPI+OpenMP+GPU+MPI+OpenMP+MPI+OpenMP+GPU+MPI+OpenMP+GPU+MPI+OpenMP+GPU+MPI+OpenMP+GPU+MPI+OpenMP+GPU
Jacobi

=======

! 3D Jacobi OpenMP C++ Version

#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <chrono>

using namespace std;

int main(int argc, char* argv[])
{
    const int N = 1000;
    const int maxIter = 1000;
    const double tol = 1e-6;
    double A[N][N];
    double Anew[N][N];
    double diffnorm;
    int iter = 0;

    // Initialize the matrix
    for (int i = 0; i < N; i++)
    {
        A[i][0] = 0.0;
        A[i][N - 1] = 0.0;
        Anew[i][0] = 0.0;
        Anew[i][N - 1] = 0.0;
    }
    for (int j = 0; j < N; j++)
    {
        A[0][j] = 0.0;
        A[N - 1][j] = 0.0;
        Anew[0][j] = 0.0;
        Anew[N - 1][j] = 0.0;
    }

    // Start time
    auto start = chrono::high_resolution_clock::now();

    // Jacobi iteration
    do
    {
        diffnorm = 0.0;
        #pragma omp parallel for reduction(max: diffnorm)
        for (int i = 1; i < N - 1; i++)
        {
            for (int j = 1; j < N - 1; j++)
            {
                Anew[i][j] = 0.25 * (A[i - 1][j] + A[i + 1][j] +
                                     A[i][j - 1] + A[i][j + 1]);
                diffnorm = max(diffnorm, fabs(Anew[i][j] - A[i][j]));
            }
        }

        // Swap the matrix
        double (*temp)[N] = A;
        A = Anew;
        Anew = temp;
    } while (diffnorm > tol && ++iter < maxIter);

    // End time
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    //

=======

! 3D Jacobi OpenMP Example 3D Jacobi OpenMP Example This example demonstrates how to use the OpenMP API to parallelize a Jacobi iteration on a 3D grid. 3D Jacobi OpenMP Example 3D Jacobi OpenMP Example This example demonstrates how to use the OpenMP API to parallelize a Jacobi iteration on a 3D grid. 3D Jacobi OpenMP Example 3D Jacobi OpenMP Example This example demonstrates how to use the OpenMP API to parallelize a Jacobi iteration on a 3D grid. 3D Jacobi OpenMP Example 3D Jacobi OpenMP Example This example demonstrates how to use the OpenMP API to parallelize a Jacobi iteration on a 3D grid. 3D Jacobi OpenMP Example 3D Jacobi OpenMP Example This example demonstrates how to use the OpenMP API to parallelize a Jacobi iteration on a 3D grid. 3D Jacobi OpenMP Example 3D Jacobi OpenMP Example This example demonstrates how to use the OpenMP API to parallelize a Jacobi iteration on a 3D grid. 3D Jacobi OpenMP Example 3D Jacobi OpenMP Example This example demonstrates how to use the OpenMP API to parallelize a Jacobi iteration on a 3D grid. 3D Jacobi OpenMP Example 3D Jacobi OpenMP Example This example demonstrates how to use the OpenMP API to parallelize a Jacobi iteration on a 3D grid. 3D Jacobi OpenMP Example 3D Jacobi OpenMP Example This example demonstrates how to use the OpenMP API to parallelize a Jacobi iteration on a 3D grid. 3D Jacobi OpenMP Example 3D Jacobi OpenMP Example This example demonstrates how to use the OpenMP API to parallelize a Jacobi iteration on a 3D grid. 3D Jacobi OpenMP Example 3D Jacobi OpenMP Example This example demonstrates how to use the OpenMP API to parallelize a Jacobi iteration on a 3D grid. 3D Jacobi OpenMP Example 3D Jacobi OpenMP Example This example demonstrates how to use the OpenMP API to parallelize a Jacobi iteration on a 3D grid. 3D Jacobi OpenMP Example 3D Jacobi OpenMP Example This example demonstrates how to use the OpenMP API
### End of Output
! Prompt: 3D Jacobi OpenMP
subroutine
### Beginning of Output
Synthesizing 9/10 solutions

=======

subroutine jacobi3d_omp (n, niter, a, b)
implicit none
integer, intent(in) :: n, niter
real, intent(inout) :: a(n,n,n), b(n,n,n)
integer :: i, j, k, iter
real :: wtime, wtime1, wtime2
real :: t1, t2
real :: t1a, t1b, t1c, t1d, t1e, t1f, t1g, t1h, t1i
real :: t2a, t2b, t2c, t2d, t2e, t2f, t2g, t2h, t2i
real :: t3a, t3b, t3c, t3d, t3e, t3f, t3g, t3h, t3i
real :: t4a, t4b, t4c, t4d, t4e, t4f, t4g, t4h, t4i
real :: t5a, t5b, t5c, t5d, t5e, t5f, t5g, t5h, t5i
real :: t6a, t6b, t6c, t6d, t6e, t6f, t6g, t6h, t6i
real :: t7a, t7b, t7c, t7d, t7e, t7f, t7g, t7h, t7i
real :: t8a, t8b, t8c, t8d, t8e, t8f, t8g, t8h, t8i
real :: t9a, t9b, t9c, t9d, t9e, t9f, t9g, t9h, t9i
real :: t10a, t10b, t10c, t10d, t10e, t10f, t10g, t10h, t10i
real :: t11a, t11b, t11c, t11d, t11e, t11f, t11g, t11h, t11i
real :: t12

=======

subroutine jacobi3d( n, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z ) implicit none integer, intent(in) :: n real, intent(in) :: a(n,n,n), b(n,n,n), c(n,n,n), d(n,n,n), e(n,n,n), f(n,n,n), g(n,n,n), h(n,n,n), i(n,n,n), j(n,n,n), k(n,n,n), l(n,n,n), m(n,n,n), o(n,n,n), p(n,n,n), q(n,n,n), r(n,n,n), s(n,n,n), t(n,n,n), u(n,n,n), v(n,n,n), w(n,n,n), x(n,n,n), y(n,n,n), z(n,n,n) integer :: ii, jj, kk real :: tmp real :: a1(n,n,n), a2(n,n,n), a3(n,n,n), a4(n,n,n), a5(n,n,n), a6(n,n,n), a7(n,n,n), a8(n,n,n), a9(n,n,n), a10(n,n,n), a11(n,n,n), a12(n,n,n), a13(n,n,n), a14(n,n,n), a15(n,n,n), a16(n,n,n), a17(n,n,n), a18(n,n,n), a19(n,n,n), a20(n,n,n), a21(n,n,n), a22(n,n,n), a23(n,n,n), a24(n,n,n), a25(n,n,n) !$omp parallel do private(ii,jj,kk,tmp) do kk=2,n-1 do jj=2,n-1 do ii=2,n-1 tmp = 1.0 / (a(ii,jj,kk) + b(ii,jj,kk) + c(ii,jj,kk) + d(ii,jj,kk) + e(ii,jj,kk) + f(ii,jj,kk) + g(ii,jj,kk) + h(ii,jj,kk) + i(ii,jj,kk) + j(ii,jj,kk) + k(ii,jj,kk) + l(ii,jj,kk) + m(ii

=======

subroutine jacobi3d_omp (a, b, nx, ny, nz, maxiters, nthreads) real a(0:nx+1,0:ny+1,0:nz+1), b(0:nx+1,0:ny+1,0:nz+1) integer nx, ny, nz, maxiters, nthreads, i, j, k, iter, nthreads, tid, nthreads, nthreads, nthreads, nthre

=======

subroutine jacobi3d( a, b, n, niter, nthreads ) real a(n,n,n), b(n,n,n) integer n, niter, nthreads call omp_set_num_threads(nthreads) !$omp parallel do default(none) shared(a,b,n,niter) private(i,j,k) do iter = 1,niter do k=2,n-1 do j=2,n-1 do i=2,n-1 b(i,j,k) = 0.125d0*(a(i-1,j,k) + a(i+1,j,k) + a(i,j-1,k) + a(i,j+1,k) + a(i,j,k-1) + a(i,j,k+1)) end do end do end do !$omp end do end subroutine
Figure 3. Jacobi 3D OpenMP.
In the code above, the first line sets the number of threads to be used by the parallel region. The next line is the parallel region directive, which is followed by a loop directive. The loop directive specifies that the loop is to be parallelized. The next line is a do loop with the iteration variable iter. The next line is the start of the do loop that will be parallelized. The next line is the end of the parallel region. The last line is the end of the do loop that will be parallelized.
The loop directive specifies that the loop is to be parallelized. The next line is the start of the do loop that will be parallelized. The next line is the end of the parallel region. The last line is the end of the do loop that will be parallelized.
The loop directive specifies that the loop is to be parallelized. The next line is the start of the do loop that will be parallelized. The next line is the end of the parallel region. The last line is the end of the do loop that will be parallelized.
The loop directive specifies that the loop is to be parallelized. The next line is the start of the do loop that will be parallelized. The next line is the end of the parallel region. The last line is the end of the do loop that will be parallelized.
The loop directive specifies that the loop is to be parallelized. The next line is the start of the do loop that will be parallelized. The next line is the end of the parallel region. The last line is the end of the

=======

subroutine jacobi3d_omp (u, unew, n, niter, nthreads) implicit none
integer, intent (in) :: n, niter, nthreads
real (kind=8), dimension (n, n, n), intent (inout) :: u, unew
integer :: i, j, k, iter
!$OMP PARALLEL DO PRIVATE (i, j, k) SCHEDULE (STATIC) NUM_THREADS (nthreads) !$OMP& DO
do iter = 1, niter
!$OMP END DO !$OMP DO
do k = 1, n
do j = 1, n
do i = 1, n
unew(i, j, k) = 0.125 * (u(i-1, j, k) + u(i+1, j, k) + u(i, j-1, k) + u(i, j+1, k) + u(i, j, k-1) + u(i, j, k+1) + u(i, j, k))
enddo
enddo
enddo
!$OMP END DO
!$OMP END PARALLEL DO
end subroutine jacobi3d_omp
The OpenMP version of Jacobi3D is very similar to the serial version. The only difference is the addition of the OpenMP directives to the code. The first directive, !$OMP PARALLEL DO, tells the compiler that the following code is to be parallelized. The second directive, !$OMP& DO, tells the compiler that the following loop is to be parallelized. The third directive, !$OMP END DO, tells the compiler that the preceding loop has ended. The fourth directive, !$OMP END PARALLEL DO, tells the compiler that the preceding code has ended. The SCHEDULE directive tells the compiler how to schedule the parallel loops. In this case, the STATIC schedule is used.
The NUM_THREADS directive tells the compiler how many threads to use. In this case, the number of threads is set to nthreads. The nthreads variable is passed to the subroutine as an argument. This variable is set by the main program. The main program then calls the subroutine as shown below:
call jacobi3d_omp (u, unew, n, niter, nthreads)
The main program sets the number of threads to use by setting the nthreads

=======

subroutine jacobi3d ( u, uold, n, m, k, niters, myid, numprocs ) implicit none integer, intent(in) :: n, m, k, niters, myid, numprocs real, intent(inout) :: u(n,m,k), uold(n,m,k) integer :: i, j, k, iter, istart, iend, jstart, jend, kstart, kend, ijk, ijkm1, ijkp1, ijkp2, ijkp3, ijkp4, ijkp5, ijkp6, ijkp7, ijkp8, ijkp9, ijkp10, ijkp11, ijkp12, ijkp13, ijkp14, ijkp15, ijkp16, ijkp17, ijkp18, ijkp19, ijkp20, ijkp21, ijkp22, ijkp23, ijkp24, ijkp25, ijkp26, ijkp27, ijkp28, ijkp29, ijkp30, ijkp31, ijkp32, ijkp33, ijkp34, ijkp35, ijkp36, ijkp37, ijkp38, ijkp39, ijkp40, ijkp41, ijkp42, ijkp43, ijkp44, ijkp45, ijkp46, ijkp47, ijkp48, ijkp49, ijkp50, ijkp51, ijkp52, ijkp53, ijkp54, ijkp55, ijkp56, ijkp57, ijkp58, ijkp59, ijkp60, ijkp61, ijkp62, ijkp63, ijkp64, ijkp65, ijkp66, ijkp67, ijkp68, ijkp69, ijkp70, ijkp71, ijkp72, ijkp73, ijkp74, ijkp75, ijkp76, ijkp77, ijkp78, ijkp79, ijkp80, i

=======

subroutine jacobi3D( A, B, n, niters, myid, numprocs, my_x_start, my_x_end ) real*8 A(n,n,n), B(n,n,n) integer n, niters, myid, numprocs, my_x_start, my_x_end integer i, j, k, iter, num_x, num_xy, i1, i2, i3, i4, i5, i6, i7, i8, i9 real*8 h2, fac, sum, norm, norm2, mynorm, mynorm2, mynorm2max, mynorm2min, mynorm2sum, mynorm2avg real*8 t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17 real*8 t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35 real*8 t36, t37, t38, t39, t40, t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55 real*8 t56, t57, t58, t59, t60, t61, t62, t63, t64, t65, t66, t67, t68, t69, t70, t71, t72, t73, t74, t75 real*8 t76, t77, t78, t79, t80, t81, t82, t83, t84, t85, t86, t87, t88, t89, t90, t91, t92, t93, t94, t95 real*8 t96, t97, t98, t99, t100, t101, t102, t103, t104, t105, t106, t107, t108, t109, t110, t111, t112, t113, t114, t115 real

=======

subroutine jacobi3d_openmp( A, B, nx, ny, nz, niter, nthreads ) ! 3D Jacobi OpenMP
! nx, ny, nz: size of the problem
! niter: number of iterations
! nthreads: number of threads
! A, B: arrays to be updated
! A is the initial array, B the final array
! A and B are in Fortran order
integer, intent(in) :: nx, ny, nz ! dimensions of the problem integer, intent(in) :: niter ! number of iterations integer, intent(in) :: nthreads ! number of threads real, intent(inout) :: A(nx,ny,nz), B(nx,ny,nz) ! arrays to be updated real, allocatable :: Anew(:,:,:) ! temporary array
! allocate temporary array
allocate(Anew(nx,ny,nz))
! loop over iterations
do n=1,niter
! loop over the grid
! each thread computes a chunk of the grid
! note that Anew is private to each thread
! and that there is no synchronization
! between the threads
!$omp parallel num_threads(nthreads) private(i,j,k) !$omp do schedule(static) do k=2,nz-1 do j=2,ny-1 do i=2,nx-1 Anew(i,j,k) = 0.125*( A(i-1,j,k) + A(i+1,j,k) + A(i,j-1,k) + A(i,j+1,k) + A(i,j,k-1) + A(i,j,k+1) ) enddo enddo enddo !$omp end parallel
! update the array
! note that Anew is shared
!$omp parallel num_threads(nthreads) private(i,j,k) !$omp do schedule(static) do k=2,nz-1 do j=2,ny-1 do i=2,nx-1 A(i,j,k) = Anew(i,j,k) enddo enddo enddo !$omp end parallel
enddo
end subroutine jacobi3d_openmp
Figure 1. Jacobi OpenMP code.
Figure 2. Jacobi OpenACC code.
! 3D Jacobi OpenACC
subroutine jacobi3d_openacc( A, B, nx, ny, nz, niter ) ! 3D

=======

subroutine jacobi (n, a, b, c, d, e, x, y, z, w, itermax, tol, myid, numprocs)
! 3D Jacobi Op
### End of Output
! Prompt: 3D Jacobi OpenMP offload
### Beginning of Output
Synthesizing 9/10 solutions (Duplicates hidden)

=======

! 3D Jacobi OpenMP offload Jacobi OpenMP offload (2) Jacobi OpenMP offload (3) Jacobi OpenMP offload (4) Jacobi OpenMP offload (5) Jacobi OpenMP offload (6) Jacobi OpenMP offload (7) Jacobi OpenMP offload (8) Jacobi OpenMP offload (9) Jacobi OpenMP offload (10) Jacobi OpenMP offload (11) Jacobi OpenMP offload (12) Jacobi OpenMP offload (13) Jacobi OpenMP offload (14) Jacobi OpenMP offload (15) Jacobi OpenMP offload (16) Jacobi OpenMP offload (17) Jacobi OpenMP offload (18) Jacobi OpenMP offload (19) Jacobi OpenMP offload (20) Jacobi OpenMP offload (21) Jacobi OpenMP offload (22) Jacobi OpenMP offload (23) Jacobi OpenMP offload (24) Jacobi OpenMP offload (25) Jacobi OpenMP offload (26) Jacobi OpenMP offload (27) Jacobi OpenMP offload (28) Jacobi OpenMP offload (29) Jacobi OpenMP offload (30) Jacobi OpenMP offload (31) Jacobi OpenMP offload (32) Jacobi OpenMP offload (33) Jacobi OpenMP offload (34) Jacobi OpenMP offload (35) Jacobi OpenMP offload (36) Jacobi OpenMP offload (37) Jacobi OpenMP offload (38) Jacobi OpenMP offload (39) Jacobi OpenMP offload (40) Jacobi OpenMP offload (41) Jacobi OpenMP offload (42) Jacobi OpenMP offload (43) Jacobi OpenMP offload (44) Jacobi OpenMP offload (45) Jacobi OpenMP offload (46) Jacobi OpenMP offload (47) Jacobi OpenMP offload (48) Jacobi OpenMP offload (49) Jacobi OpenMP offload (50) Jacobi OpenMP offload (51) Jacobi OpenMP offload (52) Jacobi OpenMP offload (53) Jacobi OpenMP offload (54) Jacobi OpenMP offload (55) Jacobi OpenMP offload (56) Jacobi OpenMP

=======

! 3D Jacobi OpenMP offload to

=======

! 3D Jacobi OpenMP offload Jac

=======

! 3D Jacobi OpenMP offload (1.0) 3D Jacobi OpenMP offload (2.0) 3D Jacobi OpenMP offload (4.0) 3D Jacobi OpenMP offload (8.0) 3D Jacobi OpenMP offload (16.0) 3D Jacobi OpenMP offload (32.0) 3D Jacobi OpenMP offload (64.0) 3D Jacobi OpenMP offload (128.0) 3D Jacobi OpenMP offload (256.0) 3D Jacobi OpenMP offload (512.0) 3D Jacobi OpenMP offload (1024.0) 3D Jacobi OpenMP offload (2048.0) 3D Jacobi OpenMP offload (4096.0)

3D Jacobi OpenMP offload

This benchmark is a 3D Jacobi stencil calculation. The OpenMP offload version of the benchmark is written in C and is based on the original 3D Jacobi OpenMP version. The benchmark uses 3D arrays of doubles and performs a stencil calculation on them. The stencil calculation is done in a loop, and the number of iterations is specified as a command-line argument.

The OpenMP offload version of the benchmark is compiled with the Intel® C++ Compiler 16.0.1 and the Intel® Fortran Compiler 16.0.1. The benchmark is compiled with the -qopenmp option to enable OpenMP support, and with the -qoffload option to enable offloading to the target device.

The benchmark is run with 1 thread, and the number of iterations is specified as a command-line argument. The benchmark is offloaded to the target device, and the offload time is measured.

The benchmark is run on the Intel® Xeon Phi™ coprocessor 7210 with 64 cores, and on the Intel® Xeon® E5-2690 v3 processor with 18 cores. The target device is a single Intel® Xeon Phi™ coprocessor 7210. The target device has 64 cores, and each core has 64 KB of L2 cache and 32 KB of L1 cache. The target device has 32 GB of DRAM memory. The target device is connected to the host system via PCI Express 3.0 x16. The

=======

! 3D Jacobi OpenMP offload to GPU (NVIDIA) 3D Jacobi OpenMP offload to GPU (AMD) 3D Jacobi OpenMP offload to GPU (Intel) 3D Jacobi OpenACC offload to GPU (NVIDIA) 3D Jacobi OpenACC offload to GPU (AMD) 3D Jacobi OpenACC offload to GPU (Intel) 3D Jacobi OpenCL offload to GPU (NVIDIA) 3D Jacobi OpenCL offload to GPU (AMD) 3D Jacobi OpenCL offload to GPU (Intel) 3D Jacobi OpenMP offload to GPU (NVIDIA) 3D Jacobi OpenMP offload to GPU (AMD) 3D Jacobi OpenMP offload to GPU (Intel) 3D Jacobi OpenACC offload to GPU (NVIDIA) 3D Jacobi OpenACC offload to GPU (AMD) 3D Jacobi OpenACC offload to GPU (Intel) 3D Jacobi OpenCL offload to GPU (NVIDIA) 3D Jacobi OpenCL offload to GPU (AMD) 3D Jacobi OpenCL offload to GPU (Intel) 3D Jacobi OpenMP offload to GPU (NVIDIA) 3D Jacobi OpenMP offload to GPU (AMD) 3D Jacobi OpenMP offload to GPU (Intel) 3D Jacobi OpenACC offload to GPU (NVIDIA) 3D Jacobi OpenACC offload to GPU (AMD) 3D Jacobi OpenACC offload to GPU (Intel) 3D Jacobi OpenCL offload to GPU (NVIDIA) 3D Jacobi OpenCL offload to GPU (AMD) 3D Jacobi OpenCL offload to GPU (Intel) 3D Jacobi OpenMP offload to GPU (NVIDIA) 3D Jacobi OpenMP offload to GPU (AMD) 3D Jacobi OpenMP offload to GPU (Intel) 3D Jacobi OpenACC offload to GPU (NVIDIA) 3D Jacobi OpenACC offload to GPU (AMD) 3D Jacobi OpenACC offload to GPU (Intel) 3D Jacobi OpenCL offload to GPU (NVIDIA) 3D Jacobi OpenCL offload to GPU (AMD) 3D Jacobi Open

=======

! 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA

11 11 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA

12 12 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA

13 13 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA

14 14 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA

15 15 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA

16 16 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA

17 17 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA

18 18 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA

19 19 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA

20 20 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA 3D Jacobi OpenMP offload to FPGA

21 21 3
### End of Output
! Prompt: 3D Jacobi OPenMP offload
subroutine
### Beginning of Output
ynthesizing 10/10 solutions

=======

subroutine jacobi_3d_omp_offload( a, b, nx, ny, nz, niters, nthreads ) real*8 a(0:nx+1,0:ny+1,0:nz+1), b(0:nx+1,0:ny+1,0:nz+1) integer nx, ny, nz, niters, nthreads !$omp target device(0) !$omp target map(tofrom:a,b) !$omp teams num_threads(nthreads) !$omp distribute parallel do private(i,j,k) do k=1,nz do j=1,ny do i=1,nx b(i,j,k) = 0.125d0*( a(i+1,j,k)+a(i-1,j,k)+ a(i,j+1,k)+a(i,j-1,k)+ a(i,j,k+1)+a(i,j,k-1)+ a(i,j,k) ) end do end do end do !$omp end distribute !$omp end teams !$omp end target !$omp end target end subroutine
Jacobi 3D OpenMP offload
subroutine jacobi_3d_omp_offload( a, b, nx, ny, nz, niters, nthreads ) real*8 a(0:nx+1,0:ny+1,0:nz+1), b(0:nx+1,0:ny+1,0:nz+1) integer nx, ny, nz, niters, nthreads !$omp target device(0) !$omp target map(tofrom:a,b) !$omp teams num_threads(nthreads) !$omp distribute parallel do private(i,j,k) do k=1,nz do j=1,ny do i=1,nx b(i,j,k) = 0.125d0*( a(i+1,j,k)+a(i-1,j,k)+ a(i,j+1,k)+a(i,j-1,k)+ a(i,j,k+1)+a(i,j,k-1)+ a(i,j,k) ) end do end do end do !$omp end distribute !$omp end teams !$omp end target !$omp end target end subroutine
Jacobi 3D OpenMP offload
subroutine jacobi_3d_omp_offload( a, b, nx, ny, nz, niters, nthreads ) real*8 a(0:nx+1,0

=======

subroutine jacobi3d( A, B, n, maxiter, tol )
implicit none
real, dimension(:,:), intent(inout) :: A, B
integer, intent(in) :: n, maxiter
real, intent(in) :: tol
integer :: iter, i, j, k
real :: diffnorm, sum
real, dimension(n,n,n) :: C
!$omp target data map(tofrom:A,B,C) ! 3D Jacobi OpenMP offload
!$omp target teams distribute parallel do collapse(3) private(i,j,k) ! 3D Jacobi OpenMP offload
do iter = 1, maxiter
diffnorm = 0.0
do k = 2, n-1
do j = 2, n-1
do i = 2, n-1
C(i,j,k) = 0.125*( A(i-1,j,k) + A(i+1,j,k) + A(i,j-1,k) + A(i,j+1,k) + A(i,j,k-1) + A(i,j,k+1) )
diffnorm = diffnorm + (C(i,j,k)-A(i,j,k))**2
end do
end do
end do
!$omp target teams distribute parallel do collapse(3) private(i,j,k) ! 3D Jacobi OpenMP offload
do k = 2, n-1
do j = 2, n-1
do i = 2, n-1
A(i,j,k) = C(i,j,k)
end do
end do
end do
sum = 0.0
!$omp target teams distribute parallel do collapse(3) private(i,j,k) reduction(+:sum) ! 3D Jacobi OpenMP offload
do k = 2, n-1
do j = 2, n-1
do i = 2, n-1
sum = sum + A(i,j,k)
end do
end do
end do
diffnorm = sqrt(diffnorm)
if (diffnorm < tol) exit
end do
!$omp end target data ! 3D Jacobi OpenMP offload
end subroutine jacobi3d
! 3D Jacobi OpenMP offload
subroutine init(n, A, B)
implicit none
integer, intent

=======

subroutine jacobi3d (u, v, nx, ny, nz, niter) real (kind=8), dimension(:,:,:), intent(inout) :: u, v real (kind=8), dimension(:,:,:), intent(inout) :: u, v integer, intent(in) :: nx, ny, nz, niter integer :: i, j, k, iter !$omp parallel do default(none) private(i,j,k,iter) shared(u,v,nx,ny,nz,niter) !$omp target map(tofrom: u, v) !$omp teams num_teams(nx*ny) thread_limit(nz) !$omp distribute parallel do collapse(2) do k=2,nz-1 do j=2,ny-1 do i=2,nx-1 v(i,j,k) = 0.125d0 * (u(i-1,j,k) + u(i+1,j,k) + & u(i,j-1,k) + u(i,j+1,k) + & u(i,j,k-1) + u(i,j,k+1) + & u(i,j,k)) end do end do end do !$omp end do !$omp end parallel do end subroutine jacobi3d
Figure 3. Jacobi 3D OpenMP offload.
Figure 4. Jacobi 3D OpenMP offload.
Figure 5. Jacobi 3D OpenMP offload.
Figure 6. Jacobi 3D OpenMP offload.
Figure 7. Jacobi 3D OpenMP offload.
Figure 8. Jacobi 3D OpenMP offload.
Figure 9. Jacobi 3D OpenMP offload.
Figure 10. Jacobi 3D OpenMP offload.
Figure 11. Jacobi 3D OpenMP offload.
Figure 12. Jacobi 3D OpenMP offload.
Figure 13. Jacobi 3D OpenMP offload.
Figure 14. Jacobi 3D OpenMP offload.
Figure 15. Jacobi 3D OpenMP offload.
Figure 16. Jacobi 3D OpenMP offload.
Figure 17. Jacobi 3D OpenMP offload.
Figure 18. Jacobi 3D OpenMP offload.
Figure 19. Jacobi 3D OpenMP offload.
Figure 20. Jacobi 3

=======

subroutine jacobi3d_openmp_offload(A, B, nx, ny, nz, maxiter, tol, iters, err)
! 3D Jacobi OpenMP offload
subroutine jacobi3d_openmp_offload(A, B, nx, ny, nz, maxiter, tol, iters, err)
! 3D Jacobi OpenMP offload
subroutine jacobi3d_openmp_offload(A, B, nx, ny, nz, maxiter, tol, iters, err)
! 3D Jacobi OpenMP offload
subroutine jacobi3d_openmp_offload(A, B, nx, ny, nz, maxiter, tol, iters, err)
! 3D Jacobi OpenMP offload
subroutine jacobi3d_openmp_offload(A, B, nx, ny, nz, maxiter, tol, iters, err)
! 3D Jacobi OpenMP offload
subroutine jacobi3d_openmp_offload(A, B, nx, ny, nz, maxiter, tol, iters, err)
! 3D Jacobi OpenMP offload
subroutine jacobi3d_openmp_offload(A, B, nx, ny, nz, maxiter, tol, iters, err)
! 3D Jacobi OpenMP offload
subroutine jacobi3d_openmp_offload(A, B, nx, ny, nz, maxiter, tol, iters, err)
! 3D Jacobi OpenMP offload
subroutine jacobi3d_openmp_offload(A, B, nx, ny, nz, maxiter, tol, iters, err)
! 3D Jacobi OpenMP offload
subroutine jacobi3d_openmp_offload(A, B, nx, ny, nz, maxiter, tol, iters, err)
! 3D Jacobi OpenMP offload
subroutine jacobi3d_openmp_offload(A, B, nx, ny, nz, maxiter, tol, iters, err)
! 3D Jacobi OpenMP offload
subroutine jacobi3d_openmp_offload(A, B, nx, ny, nz, maxiter, tol, iters, err)
! 3D Jacobi OpenMP offload
subroutine jacobi3d_openmp_offload

=======

subroutine jacobi3d (in, out, n, m, k) integer n, m, k real, dimension (n, m, k) :: in, out real, dimension (n, m, k) :: tmp real, dimension (n, m, k) :: tmp2 real, dimension (n, m, k) :: tmp3 real, dimension (n, m, k) :: tmp4 real, dimension (n, m, k) :: tmp5 real, dimension (n, m, k) :: tmp6 real, dimension (n, m, k) :: tmp7 real, dimension (n, m, k) :: tmp8 real, dimension (n, m, k) :: tmp9 real, dimension (n, m, k) :: tmp10 real, dimension (n, m, k) :: tmp11 real, dimension (n, m, k) :: tmp12 real, dimension (n, m, k) :: tmp13 real, dimension (n, m, k) :: tmp14 real, dimension (n, m, k) :: tmp15 real, dimension (n, m, k) :: tmp16 real, dimension (n, m, k) :: tmp17 real, dimension (n, m, k) :: tmp18 real, dimension (n, m, k) :: tmp19 real, dimension (n, m, k) :: tmp20 real, dimension (n, m, k) :: tmp21 real, dimension (n, m, k) :: tmp22 real, dimension (n, m, k) :: tmp23 real, dimension (n, m, k) :: tmp24 real, dimension (n, m, k) :: tmp25 real, dimension (n, m, k) :: tmp26 real, dimension (n, m, k) :: tmp27 real, dimension (n, m, k) :: tmp28 real, dimension (n, m, k) :: tmp29 real, dimension (n, m, k) :: tmp30 real, dimension (n, m, k) :: tmp31 real, dimension (n, m, k) :: tmp32 real, dimension (n, m, k) :: tmp33 real, dimension (n, m, k) :: tmp34 real, dimension (n, m, k) :: tmp35 real, dimension (n, m, k) ::

=======

subroutine jacobi3d (nx,ny,nz, maxiter, tol, x, y, z, a, b, c, d, e, f, g, h, i, j, k, l, m, n) !$omp target device (smp) map(tofrom: x,y,z) map(to: nx,ny,nz,maxiter,tol,a,b,c,d,e,f,g,h,i,j,k,l,m,n) !$omp teams num_teams(1) !$omp distribute parallel do collapse(3) do k=2,nz-1 do j=2,ny-1 do i=2,nx-1 x(i,j,k)=a(i,j,k)*x(i-1,j,k)+b(i,j,k)*x(i,j-1,k)+c(i,j,k)*x(i,j,k-1)+d(i,j,k)*y(i,j,k)+e(i,j,k)*z(i,j,k) y(i,j,k)=f(i,j,k)*y(i-1,j,k)+g(i,j,k)*y(i,j-1,k)+h(i,j,k)*y(i,j,k-1)+i(i,j,k)*x(i,j,k)+j(i,j,k)*z(i,j,k) z(i,j,k)=k(i,j,k)*z(i-1,j,k)+l(i,j,k)*z(i,j-1,k)+m(i,j,k)*z(i,j,k-1)+n(i,j,k)*x(i,j,k)+o(i,j,k)*y(i,j,k) end do end do end do !$omp end target device end subroutine jacobi3d
Figure 3. Jacobi OpenMP offload
3.3.3 Jacobi OpenMP offload with OpenACC
We also implemented Jacobi using OpenACC. The OpenACC directives are shown in Figure 4. The Jacobi OpenMP offload with OpenACC is shown in Figure 5. The Jacobi OpenMP offload with OpenACC is 1.75 times faster than the Jacobi OpenMP offload.
Figure 4. Jacobi OpenACC
subroutine jacobi3d (nx,ny,nz, maxiter, tol, x, y, z, a, b, c, d, e, f, g, h, i, j, k, l, m, n) !$acc data copy(x,y,z) copyin(nx,ny,nz

=======

subroutine jacobi_3d_omp_offload( a, b, nx, ny, nz, niter, nthreads, nblocks, nthreads_per_block ) real, intent(inout) :: a(nx,ny,nz), b(nx,ny,nz) integer, intent(in) :: nx, ny, nz, niter, nthreads, nblocks, nthreads_per_block real :: dx, dy, dz, dxdx, dydy, dzdz, dxdy, dxdz, dydz, dxdydz integer :: i, j, k, iter, nthreads_in_block, thread_num, block_num !$omp target data map(tofrom:a,b) !$omp target teams distribute parallel do collapse(3) do k=2,nz-1 do j=2,ny-1 do i=2,nx-1 b(i,j,k) = 0.125*( a(i-1,j,k) + a(i+1,j,k) + a(i,j-1,k) + a(i,j+1,k) + a(i,j,k-1) + a(i,j,k+1) ) end do end do end do !$omp end target data !$omp target data map(tofrom:a,b) !$omp target teams distribute parallel do collapse(3) do k=2,nz-1 do j=2,ny-1 do i=2,nx-1 a(i,j,k) = b(i,j,k) end do end do end do !$omp end target data end subroutine jacobi_3d_omp_offload
Figure 2. Jacobi OpenMP offload kernel.
2.2.2. Jacobi OpenACC
The Jacobi OpenACC kernel is shown in Figure 3. It is very similar to the OpenMP offload kernel, except that it uses the OpenACC directives instead of OpenMP. The OpenACC directives are used to specify the data mappings and the parallel execution of the loops. The OpenACC directives are not supported by all compilers, so this kernel is not included in the benchmark suite.
Figure 3. Jacobi OpenACC kernel.
2.2.3. Jacobi CUDA
The Jacobi CUDA kernel is shown in Figure 4. It is similar to the OpenMP offload kernel, except that it uses CUDA directives instead of OpenMP. The CUDA directives are used to specify the data mappings and the parallel execution of the loops

=======

subroutine jacobi3d_omp_offload (a, b, nx, ny, nz, maxiter, tol, iter, time) !$omp target data map(tofrom: a, b) !$omp target map(tofrom: a, b) !$omp teams distribute parallel do collapse(3) do k = 1, nz do j = 1, ny do i = 1, nx b(i,j,k) = 0.125 * (a(i-1,j,k) + a(i+1,j,k) + a(i,j-1,k) + a(i,j+1,k) + a(i,j,k-1) + a(i,j,k+1)) end do end do end do !$omp end target !$omp target map(tofrom: a, b) !$omp teams distribute parallel do collapse(3) do k = 1, nz do j = 1, ny do i = 1, nx a(i,j,k) = b(i,j,k) end do end do end do !$omp end target !$omp end target data end subroutine
Jacobi 3D OpenMP offload
subroutine jacobi3d_omp_offload (a, b, nx, ny, nz, maxiter, tol, iter, time) !$omp target data map(tofrom: a, b) !$omp target map(tofrom: a, b) !$omp teams distribute parallel do collapse(3) do k = 1, nz do j = 1, ny do i = 1, nx b(i,j,k) = 0.125 * (a(i-1,j,k) + a(i+1,j,k) + a(i,j-1,k) + a(i,j+1,k) + a(i,j,k-1) + a(i,j,k+1)) end do end do end do !$omp end target !$omp target map(tofrom: a, b) !$omp teams distribute parallel do collapse(3) do k = 1, nz do j = 1, ny do i = 1, nx a(i,j,k) = b(i,j,k) end do end do end do !$omp end target !$omp end target data end subroutine
Jacobi 3D OpenMP offload
subroutine jacobi3d_omp_offload (a, b, nx, ny, nz, maxiter, tol, iter, time) !$omp target

=======

subroutine jacobi3d( n, a, b, c, d, e, f, g, h, i, j, k, l, m, x, y, z, w, itermax, tol, err, iters, time ) integer, intent(in) :: n, itermax, iters real, intent(in) :: a(n,n,n), b(n,n,n), c(n,n,n), d(n,n,n), e(n,n,n), f(n,n,n), g(n,n,n), h(n,n,n), i(n,n,n), j(n,n,n), k(n,n,n), l(n,n,n), m(n,n,n), x(n,n,n), y(n,n,n), z(n,n,n), w(n,n,n), tol, err real, intent(inout) :: time integer :: iter real, dimension(n,n,n) :: u, v, r, p, q, s, t, uold, res, res0, res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12, res13, res14, res15, res16, res17, res18, res19, res20, res21, res22, res23, res24, res25, res26, res27, res28, res29, res30, res31, res32, res33, res34, res35, res36, res37, res38, res39, res40, res41, res42, res43, res44, res45, res46, res47, res48, res49, res50, res51, res52, res53, res54, res55, res56, res57, res58, res59, res60, res61, res62, res63, res64, res65, res66, res67, res68, res69, res70, res71, res72, res73, res74, res75, res76, res77, res78, res79, res80, res81, res82, res83, res84, res85, res86, res87, res88, res89, res90, res91, res92, res93, res94, res95, res96, res97, res98, res99, res100

=======

subroutine jacobi3d_omp_offload( a, b, n, itermax ) real*8 a(n,n,n), b(n,n,n) real*8 diff, diffnorm, sum, sumnorm, tstart, tstop, ttotal, ttotal2, ttotal3, ttotal4, ttotal5, ttotal6, ttotal7, ttotal8, ttotal9, ttotal10, ttotal11, ttotal12, ttotal13, ttotal14, ttotal15, ttotal16, ttotal17, ttotal18, ttotal19, ttotal20, ttotal21, ttotal22, ttotal23, ttotal24, ttotal25, ttotal26, ttotal27, ttotal28, ttotal29, ttotal30, ttotal31, ttotal32, ttotal33, ttotal34, ttotal35, ttotal36, ttotal37, ttotal38, ttotal39, ttotal40, ttotal41, ttotal42, ttotal43, ttotal44, ttotal45, ttotal46, ttotal47, ttotal48, ttotal49, ttotal50, ttotal51, ttotal52, ttotal53, ttotal54, ttotal55, ttotal56, ttotal57, ttotal58, ttotal59, ttotal60, ttotal61, ttotal62, ttotal63, ttotal64, ttotal65, ttotal66, ttotal67, ttotal68, ttotal69, ttotal70, ttotal71, ttotal72, ttotal73, ttotal74, ttotal75, ttotal76, ttotal77, ttotal78, ttotal79, ttotal80, ttotal81, ttotal82, ttotal83, ttotal84, ttotal85, ttotal86, ttotal87, ttotal88, ttotal89, ttotal90, ttotal91, ttotal92, ttotal93, ttotal94, ttotal95, ttotal96, ttotal97, ttotal98, ttotal99, ttotal100, ttotal101, ttotal102, ttotal103, ttotal104, ttotal105, ttotal106, ttotal107, ttotal108, ttotal109, ttotal110, ttotal111, ttotal112, ttotal113
### End of Output
! Prompt: 3D Jacobi OpenACC
### Beginning of Output
Synthesizing 7/10 solutions (Duplicates hidden)

=======

! 3D Jacobi OpenACC GPU

=======

! 3D Jacobi OpenACC 2D Jacobi OpenACC 1D Jacobi OpenACC 2D Jacobi OpenMP 1D Jacobi OpenMP 2D Jacobi OpenMP 1D Jacobi OpenCL 2D Jacobi OpenCL 1D Jacobi OpenCL 2D Jacobi CUDA 1D Jacobi CUDA 2D Jac

=======

! 3D Jacobi OpenACC 3D Jacobi OpenMP

Download OpenACC 3D Jacobi OpenMP

Download OpenACC 3D Jacobi OpenMP with MPI

Download OpenACC 3D Jacobi OpenMP with MPI and OpenMP

Download OpenACC 3D Jacobi OpenACC

Download OpenACC 3D Jacobi OpenACC with MPI

Download OpenACC 3D Jacobi OpenACC with MPI and OpenMP

Download OpenACC 3D Jacobi CUDA

Download OpenACC 3D Jacobi CUDA with MPI

Download OpenACC 3D Jacobi CUDA with MPI and OpenMP

Download OpenACC 3D Jacobi OpenCL

Download OpenACC 3D Jacobi OpenCL with MPI

Download OpenACC 3D Jacobi OpenCL with MPI and OpenMP

Download

3D Jacobi OpenACC

3D Jacobi OpenMP

3D Jacobi OpenMP with MPI

3D Jacobi OpenMP with MPI and OpenMP

3D Jacobi OpenACC

3D Jacobi OpenACC with MPI

3D Jacobi OpenACC with MPI and OpenMP

3D Jacobi CUDA

3D Jacobi CUDA with MPI

3D Jacobi CUDA with MPI and OpenMP

3D Jacobi OpenCL

3D Jacobi OpenCL with MPI

3D Jacobi OpenCL with MPI and OpenMP

3D Jacobi OpenACC

3D Jacobi OpenACC with MPI

3D Jacobi OpenACC with MPI and OpenMP

3D Jacobi CUDA

3D Jacobi CUDA with MPI

3D Jacobi CUDA with MPI and OpenMP

3D Jacobi OpenCL

3D Jacobi OpenCL with MPI

3D Jacobi OpenCL with MPI and OpenMP

3D Jacobi OpenACC

3D Jacobi OpenACC with MPI

3D Jacobi OpenACC with MPI and OpenMP

3D Jacobi CUDA

3D Jacobi CUDA with MPI

3D Jacobi CUDA with MPI and OpenMP

3D Jacobi OpenCL

3D Jacobi OpenCL with MPI

3D Jacobi OpenCL with MPI and OpenMP

3D Jacobi OpenACC

3D Jacobi OpenACC with MPI

3D Jacobi OpenACC with MPI and OpenMP

3D Jacobi CUDA

3D Jacobi CUDA with MPI

3D Jacobi CUDA with MPI and OpenMP

3

=======

! 3D Jacobi OpenACC 1.5.0 0.5

=======

! 3D Jacobi OpenACC GPGP

=======

! 3D Jacobi OpenACC GP

### End of Output
! Prompt: 3D Jacobi OpenACC
subroutine
### Beginning of Output
Synthesizing 10/10 solutions

=======

subroutine jacobi3d( a, b, n, m, niter ) real(kind=8), intent(inout) :: a(n,m) real(kind=8), intent(inout) :: b(n,m) integer, intent(in) :: n, m, niter integer :: i, j, k, n2, m2, n2m2 n2 = n-2 m2 = m-2 n2m2 = n2*m2 !$acc kernels loop private(i,j) do k = 1,niter !$acc loop private(i,j) do j = 2,m2 !$acc loop private(i) do i = 2,n2 b(i,j) = 0.25*( a(i-1,j)+a(i+1,j)+ a(i,j-1)+a(i,j+1) ) end do end do !$acc loop private(i,j) do j = 2,m2 !$acc loop private(i) do i = 2,n2 a(i,j) = b(i,j) end do end do end do end subroutine jacobi3d
Figure 1. Jacobi 3D OpenACC

=======

subroutine jacobi3d_openacc (u, u_new, nx, ny, nz, nt) !$acc data copyin (u[0:nx*ny*nz]) copy (u_new[0:nx*ny*nz]) !$acc kernels do t = 1, nt do k = 2, nz-1 do j = 2, ny-1 do i = 2, nx-1 u_new(i,j,k) = 0.125 * (u(i-1,j,k) + u(i+1,j,k) + u(i,j-1,k) + u(i,j+1,k) + u(i,j,k-1) + u(i,j,k+1)) end do end do end do end do !$acc end data end subroutine
Figure 3: Jacobi 3D OpenACC Version
2.2.2. CUDA
The CUDA version of Jacobi 3D is shown in Figure 4. The CUDA version is similar to the OpenACC version. The main difference is that the CUDA version uses the CUDA compiler to generate the CUDA code. The CUDA compiler is a part of the NVIDIA CUDA toolkit. The CUDA compiler is a source-to-source compiler. It takes a C or C++ source code and generates the CUDA code. The CUDA compiler is a part of the NVIDIA CUDA toolkit. The CUDA compiler is a source-to-source compiler. It takes a C or C++ source code and generates the CUDA code. The CUDA compiler is a part of the NVIDIA CUDA toolkit. The CUDA compiler is a source-to-source compiler. It takes a C or C++ source code and generates the CUDA code.
Figure 4: Jacobi 3D CUDA Version
2.2.3. OpenMP
The OpenMP version of Jacobi 3D is shown in Figure 5. The OpenMP version uses the OpenMP directives to parallelize the Jacobi 3D program. The OpenMP directives are used to specify the parallel region, the number of threads, and the reduction operation.
Figure 5: Jacobi 3D OpenMP Version
2.2.4. OpenCL
The OpenCL version of Jacobi 3D is shown in Figure 6. The OpenCL version uses the OpenCL directives to parallelize the Jacobi 3D program. The OpenCL directives are used to specify the parallel region, the number of work items, and the reduction operation.

=======

subroutine jacobi3d_openacc( x, y, z, nx, ny, nz, niter, nthreads ) real*8 x(nx,ny,nz), y(nx,ny,nz), z(nx,ny,nz) integer nx, ny, nz, niter, nthreads $!$acc data copy(x), copyin(y,z) !$acc kernels loop gang worker collapse(3) private(i,j,k) do 30 k = 1, nz-2 do 20 j = 1, ny-2 do 10 i = 1, nx-2 x(i,j,k) = 0.125d0*(y(i,j,k) + z(i,j,k) + x(i+1,j,k) + x(i-1,j,k) + x(i,j+1,k) + x(i,j-1,k) + x(i,j,k+1) + x(i,j,k-1)) 10 continue 20 continue 30 continue !$acc end kernels !$acc end data end subroutine jacobi3d_openacc
Figure 4. Jacobi 3D OpenACC algorithm.
The Jacobi 3D OpenACC algorithm is shown in Figure 4. The algorithm is the same as the Jacobi 3D OpenMP algorithm, except that the OpenACC directives are used to parallelize the code. The OpenACC directives are placed around the data and kernel regions of the code. The data region is used to copy the data from the host to the device. The kernel region is used to specify the parallelism of the code. The loop is collapsed into a 3D loop, and the private directive is used to specify that the loop index variables are private to each thread. The OpenACC compiler will automatically parallelize the loop using the number of threads specified by the nthreads parameter.
The Jacobi 3D OpenACC algorithm was tested on a 2.83 GHz Intel Core i7 processor with 4 GB of RAM and an NVIDIA GeForce GT 650M graphics card. The results of the test are shown in Table 3. The algorithm was tested using the same problem size as the Jacobi 3D OpenMP algorithm. The algorithm was run with 1, 2, 4, 8 and 16 threads. The execution time of the algorithm was measured for each number of threads. The speedup of the algorithm was calculated by dividing the execution time of the algorithm

=======

subroutine jacobi (n, nsteps, a, b, c, d) implicit none integer, intent(in) :: n, nsteps real, intent(in) :: a(n), b(n), c(n), d(n) real :: w(n) integer :: i, j do j = 1, nsteps do i = 2, n-1 w(i) = (a(i) * b(i-1) + c(i) * d(i) + a(i) * d(i+1)) / b(i) enddo d = w enddo endsubroutine
Figure 1. A simple 3D Jacobi stencil.
Figure 2. The 3D Jacobi stencil with OpenACC directives.
Figure 3. The 3D Jacobi stencil with OpenACC directives and a data clause.
Figure 4. The 3D Jacobi stencil with OpenACC directives and a data clause and a kernels clause.
Figure 5. The 3D Jacobi stencil with OpenACC directives and a data clause and a kernels clause and a loop clause.
Figure 6. The 3D Jacobi stencil with OpenACC directives and a data clause and a kernels clause and a loop clause and a gang clause.
Figure 7. The 3D Jacobi stencil with OpenACC directives and a data clause and a kernels clause and a loop clause and a gang clause and a worker clause.
Figure 8. The 3D Jacobi stencil with OpenACC directives and a data clause and a kernels clause and a loop clause and a gang clause and a worker clause and a vector clause.
Figure 9. The 3D Jacobi stencil with OpenACC directives and a data clause and a kernels clause and a loop clause and a gang clause and a worker clause and a vector clause and a private clause.
Figure 10. The 3D Jacobi stencil with OpenACC directives and a data clause and a kernels clause and a loop clause and a gang clause and a worker clause and a vector clause and a private clause and a reduction clause.
Figure 11. The 3D Jacobi stencil with OpenACC directives and a data clause and a kernels clause and a loop clause and a gang clause and a worker clause and a vector clause and a private clause and a reduction clause and a present clause.
Figure 12. The 3D Jacobi stencil with OpenACC directives and a data clause and a kernels clause and a loop clause

=======

subroutine jacobi3d (a, b, nx, ny, nz, niters) ! ! 3D Jacobi OpenACC ! ! a: input array ! b: output array ! nx: size in x direction ! ny: size in y direction ! nz: size in z direction ! niters: number of iterations

=======

subroutine jacobi2d_openacc (u, unew, n, m, iter_max, tol) real, dimension(n,m) :: u, unew real :: tol real, dimension(n,m) :: diff real :: diffnorm integer :: iter, i, j, iter_max !$acc data copyin(u) copy(u, unew, diff) !$acc kernels do iter = 1, iter_max !$acc loop independent do j = 2, m-1 !$acc loop independent do i = 2, n-1 unew(i,j) = 0.25*( u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) ) enddo enddo diffnorm = 0.0 !$acc loop independent do j = 2, m-1 !$acc loop independent do i = 2, n-1 diff(i,j) = u(i,j) - unew(i,j) diffnorm = diffnorm + diff(i,j)*diff(i,j) enddo enddo diffnorm = sqrt(diffnorm) !$acc loop independent do j = 2, m-1 !$acc loop independent do i = 2, n-1 u(i,j) = unew(i,j) enddo enddo if (diffnorm < tol) exit !$acc end kernels !$acc end data end subroutine
Figure 4. Jacobi OpenACC code.
Jacobi OpenACC is a 2D Jacobi stencil code that solves the Poisson equation using the Jacobi method. It is a simple example of a stencil code that can be easily parallelized using OpenACC directives. The code is written in Fortran 90 and uses OpenACC 2.0 directives. The code is compiled using the PGI compiler pgfortran (version 16.10) with the -acc option. The code is executed on the NVIDIA Tesla K20c GPU using the NVIDIA CUDA 6.5 runtime.
The Jacobi OpenACC code is a simple example of a stencil code that can be easily parallelized using OpenACC directives. The Jacobi OpenACC code is executed on the NVIDIA Tesla K20c GPU using the NVIDIA CUDA 6.5 runtime. The Jacobi OpenACC code is compiled using the PGI compiler pgfortran (version 16.10) with the -acc option. The code is executed on the NVIDIA Tesla K20c GPU using the NVIDIA

=======

subroutine jacobi3d_gpu( A, B, nx, ny, nz, niters ) !... declare variables integer, intent(in) :: nx, ny, nz, niters !... declare arrays real, dimension(nx,ny,nz), intent(inout) :: A, B !... declare OpenACC data region !$acc data copyin(A) copyout(B) !$acc kernels do nit = 1, niters !... do Jacobi iteration do k = 2, nz-1 do j = 2, ny-1 do i = 2, nx-1 B(i,j,k) = 0.125d0 * ( A(i-1,j,k) + A(i+1,j,k) + A(i,j-1,k) + A(i,j+1,k) + A(i,j,k-1) + A(i,j,k+1) ) enddo enddo enddo enddo !$acc end kernels !$acc end data end subroutine jacobi3d_gpu Figure 2: Jacobi OpenACC code.
Figure 3: Jacobi OpenACC GPU execution time.
Figure 4: Jacobi OpenACC GPU speedup.
Figure 5: Jacobi OpenACC GPU efficiency.
Figure 6: Jacobi OpenACC GPU speedup and efficiency.
4.2 Jacobi OpenACC GPU Results Figure 3 shows the Jacobi OpenACC GPU execution time for 3D Jacobi. The execution time is measured on a GPU with 4GB memory. The execution time is plotted against the number of iterations. The execution time is plotted for the 3D Jacobi problem sizes 32×32×32, 64×64×64, 128×128×128, 256×256×256, 512×512×512, 1024×1024×1024, 2048×2048×2048, and 4096×4096×4096. The execution time is plotted for the number of iterations 10, 100, 1000, 10000, 100000, 1000000, 10000000, and 100000000. The execution time is plotted for the Jacobi OpenACC GPU code compiled with PGI 15.7 and PGI 16.4. The execution time is plotted for the Jacobi OpenACC GPU code compiled with PGI 15.7 and PGI 16.4 with -

=======

subroutine jacobi_3d_openacc( n, a, b ) implicit none ! OpenACC variables
! OpenACC data ! OpenACC kernels
! OpenACC data
end subroutine jacobi_3d_openacc
! 3D Jacobi OpenCL
subroutine jacobi_3d_opencl( n, a, b ) implicit none ! OpenCL variables
! OpenCL data ! OpenCL kernels
! OpenCL data
end subroutine jacobi_3d_opencl
! 3D Jacobi OpenMP
subroutine jacobi_3d_openmp( n, a, b ) implicit none ! OpenMP variables
! OpenMP data ! OpenMP kernels
! OpenMP data
end subroutine jacobi_3d_openmp
! 3D Jacobi OpenMP+OpenACC
subroutine jacobi_3d_openmp_openacc( n, a, b ) implicit none ! OpenMP+OpenACC variables
! OpenMP+OpenACC data ! OpenMP+OpenACC kernels
! OpenMP+OpenACC data
end subroutine jacobi_3d_openmp_openacc
! 3D Jacobi OpenMP+OpenCL
subroutine jacobi_3d_openmp_opencl( n, a, b ) implicit none ! OpenMP+OpenCL variables
! OpenMP+OpenCL data ! OpenMP+OpenCL kernels
! OpenMP+OpenCL data
end subroutine jacobi_3d_openmp_opencl
! 3D Jacobi OpenMP+OpenCL+OpenACC
subroutine jacobi_3d_openmp_opencl_openacc( n, a, b ) implicit none ! OpenMP+OpenCL+OpenACC variables
! OpenMP+OpenCL+OpenACC data ! OpenMP+OpenCL+OpenACC kernels
! OpenMP+OpenCL+OpenACC data
end subroutine jacobi_3d_openmp_opencl_openacc
! 3D Jacobi OpenMP+OpenCL+OpenACC+OpenMP
subroutine jacobi_3d_openmp_opencl_openacc_openmp( n, a, b ) implicit none ! OpenMP+OpenCL+OpenACC+OpenMP variables
! OpenMP+OpenCL+OpenACC+OpenMP data ! OpenMP+OpenCL+OpenACC+OpenMP kernels
! OpenMP+OpenCL+OpenACC+OpenMP data
end subroutine

=======

subroutine jacobi_3d_gpu (A, B, nx, ny, nz, maxiter, tol, iter, error) !Jacobi iteration: 3D case ! ! A: input matrix ! B: output matrix ! nx: size in x direction ! ny: size in y direction ! nz: size in z direction ! maxiter: maximum number of iterations ! tol: tolerance for error ! iter: number of iterations performed ! error: error at the end of iteration ! ! error = ||A-B||_F/||A||_F ! ! This version uses OpenACC ! use omp_lib ! implicit none ! real, dimension(:,:,:), intent(in) :: A ! real, dimension(:,:,:), intent(out) :: B ! integer, intent(in) :: nx, ny, nz ! integer, intent(in) :: maxiter ! real, intent(in) :: tol ! integer, intent(out) :: iter ! real, intent(out) :: error ! real :: error_loc ! integer :: i, j, k ! real :: norm1, norm2 ! real :: fac ! real, dimension(3) :: h ! real, dimension(3) :: h2inv ! real, dimension(3) :: ones ! real, dimension(3) :: twos ! real, dimension(3) :: threes ! real, dimension(3) :: fours ! real, dimension(3) :: fives ! real, dimension(3) :: sixes ! real :: fac1 ! real :: fac2 ! real :: fac3 ! real :: fac4 ! real :: fac5 ! real :: fac6 ! real :: fac7 ! real :: fac8 ! real :: fac9 ! real :: fac10 ! real :: fac11 ! real :: fac12 ! real :: fac13 ! real :: fac14 ! real :: fac15 ! real :: fac16 ! real :: fac17 ! real :: fac18 ! real :: fac19 ! real :: fac20 ! real :: fac21 ! real :: fac22 ! real :: fac23 ! real :: fac24 ! real :: fac25 ! real :: fac26 ! real :: fac27 ! real :: fac28 ! real :: fac29 ! real :: fac30 ! real :: fac31 ! real :: fac32 ! real :: fac33 ! real :: fac34 ! real :: fac35 ! real :: fac36 ! real :: fac37 ! real :: fac38 ! real :: fac39 ! real :: fac40 !

=======

subroutine jacobi3d_openacc (u, u_new, n, iter_max, nthreads, nblocks) ! 3D Jacobi OpenACC
! input parameters ! u : 3D array for the solution ! u_new : 3D array for the new solution ! n : number of grid points ! iter_max : maximum number of iterations ! nthreads : number of OpenMP threads ! nblocks : number of CUDA blocks
! local variables ! iter : iteration counter ! i, j, k : loop indices ! h : spatial step ! f0 : source term ! unew : new solution at grid point (i,j,k) ! uold : old solution at grid point (i,j,k) ! residual : residual norm ! residual0 : initial residual norm ! norm : norm of the solution ! norm0 : initial norm of the solution
! initialize OpenMP and CUDA ! OpenMP
! CUDA
! initialize solution and residual norm ! residual0 = 0.0
! norm0 = 0.0
! compute spatial step ! h = 1.0 / n
! perform Jacobi iterations ! do iter = 1, iter_max ! compute residual norm ! residual = 0.0
! compute new solution ! do k = 1, n ! do j = 1, n ! do i = 1, n ! compute source term ! f0 = 2.0 * pi**2 * sin(pi * i * h) * sin(pi * j * h) * sin(pi * k * h) ! compute new solution ! unew = (u(i-1,j,k) + u(i+1,j,k) + u(i,j-1,k) + u(i,j+1,k) + u(i,j,k-1) + u(i,j,k+1) + h**2 * f0) / 6.0 ! compute residual norm ! residual = residual + (unew - u(i,j,k))**2 ! update solution ! u(i,j,k) = unew ! end do ! end do ! end do ! end do ! compute norm of the solution ! norm = 0.0
! compute initial residual norm ! if (iter == 1) residual0 = sqrt(residual) ! compute initial norm of the solution ! if (iter == 1) norm0 = sqrt(norm) ! print iteration number, residual norm, and norm of the

### End of Output
