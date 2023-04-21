! Prompt: gemv OpenMP subroutine
Synthesizing 8/10 solutions

=======

! gemv OpenMP subroutine
!  y = alpha*A*x + beta*y
!  A is MxN, x is Nx1, y is Mx1
!  alpha and beta are scalars
!  A is stored in column-major order
!  x and y are stored in column-major order
!
!  This subroutine uses OpenMP to parallelize the computation of y
!  over the columns of A.  Each thread is responsible for computing
!  one column of y.  The number of threads is determined by the
!  OMP_NUM_THREADS environment variable.
!
!  This subroutine is based on the BLAS Level 2 routine DGEMV.
!
!  This subroutine is intended to illustrate the use of OpenMP in
!  a Fortran program.  It is not intended to be used as a BLAS
!  replacement.  In particular, it does not check for errors or
!  invalid arguments.
!
!  This subroutine is in the public domain.
!
!  Reference:
!
!    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
!    LINPACK Users' Guide,
!    SIAM, 1979.
!
!  Parameters:
!
!    Input, integer M, the number of rows of the matrix A.
!
!    Input, integer N, the number of columns of the matrix A.
!
!    Input, real ALPHA, the scalar multiplier of A * x.
!
!    Input, real A(M,N), the matrix.
!
!    Input, real X(N), the vector to be multiplied by A.
!
!    Input, real BETA, the scalar multiplier of Y.
!
!    Input, real Y(M), the vector to be multiplied by BETA.
!
!    Output, real Y(M), the updated vector.
!
  implicit none

  integer m
  integer n
  real alpha
  real a(m,n)
  real x(n)
  real beta
  real y(m)

  integer i
  integer j
  real temp

!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(I,J,TEMP)
  do j = 1, n
    temp = alpha * x(j)
    do i = 1, m
      y(i) = y(i) + temp * a(i,j)
    end do
  end do
!$OMP END PARALLEL

=======

! gemv OpenMP subroutine
! --------------------
!
! This subroutine performs a matrix-vector multiplication:
!
!     y := alpha * A * x + beta * y
!
! where:
!
!     A is a general matrix
!     x and y are vectors
!
! The matrix A is stored in column-major format, and lda is the leading
! dimension of the two-dimensional array that contains A.
!
! This subroutine uses OpenMP to parallelize the operation. The parameter
! nthreads specifies the number of threads to use.
!
! The parameter nthreads must be a positive integer. If nthreads is 1, then
! this subroutine performs the matrix-vector multiplication sequentially.
! Otherwise, this subroutine performs the matrix-vector multiplication
! using OpenMP.
!
! The parameter nthreads must be a positive integer. If nthreads is 1, then
! this subroutine performs the matrix-vector multiplication sequentially.
! Otherwise, this subroutine performs the matrix-vector multiplication
! using OpenMP.
!
! The parameter nthreads must be a positive integer. If nthreads is 1, then
! this subroutine performs the matrix-vector multiplication sequentially.
! Otherwise, this subroutine performs the matrix-vector multiplication
! using OpenMP.
!
! The parameter nthreads must be a positive integer. If nthreads is 1, then
! this subroutine performs the matrix-vector multiplication sequentially.
! Otherwise, this subroutine performs the matrix-vector multiplication
! using OpenMP.
!
! The parameter nthreads must be a positive integer. If nthreads is 1, then
! this subroutine performs the matrix-vector multiplication sequentially.
! Otherwise, this subroutine performs the matrix-vector multiplication
! using OpenMP.
!
! The parameter nthreads must be a positive integer. If nthreads is 1, then
! this subroutine performs the matrix-vector multiplication sequentially.
! Otherwise, this subroutine performs the matrix-vector multiplication
! using OpenMP.
!
! The parameter nthreads must be a positive integer. If nthreads is 1, then
! this subroutine performs the matrix-vector multiplication sequentially.
! Otherwise, this subroutine performs the matrix-vector multiplication
! using OpenMP.
!
! The parameter nthreads must be a positive integer. If nthreads is 1, then
! this subroutine performs the matrix-vector multiplication sequentially.
! Otherwise, this subroutine performs the matrix-vector multiplication
! using OpenMP.
!
! The parameter nthreads must be a positive integer. If nthreads is 1

=======

! gemv OpenMP subroutine

! !DESCRIPTION: This is the serial implementation of the GEMV algorithm.

! !TYPE: Serial

! !USES: BLAS

! !REVISION: 1.0

! !LANGUAGE: Fortran

! !DISTRIBUTION: This software is distributed under the BSD license.

! !ORIGIN: This software is based upon work partially supported by the

! National Science Foundation under Grant No. 987654321.

! !AUTHORS: J. Smith, University of Tennessee

! !AUTHORS: K. Jones, Oak Ridge National Laboratory

! !AUTHORS: E. Anderson, University of Tennessee

! !AUTHORS: R. Harrison, University of Tennessee

! !AUTHORS: P. Smith, University of Tennessee

! !AUTHORS: B. Tuminaro, Sandia National Laboratories

! !AUTHORS: Z. Zhang, University of Tennessee

! !AUTHORS: S. Williams, University of Tennessee

! !AUTHORS: A. O'Neal, University of Tennessee

! !AUTHORS: C. Kuszmaul, Sandia National Laboratories

! !AUTHORS: M. O'Brien, Sandia National Laboratories

! !AUTHORS: E. Barszcz, Sandia National Laboratories

! !AUTHORS: S. Moore, Sandia National Laboratories

! !AUTHORS: D. Day, Sandia National Laboratories

! !AUTHORS: D. Sorensen, Sandia National Laboratories

! !AUTHORS: R. Grimes, Sandia National Laboratories

! !AUTHORS: A. Petitet, Sandia National Laboratories

! !AUTHORS: M. Heroux, Sandia National Laboratories

! !AUTHORS: S. Hoang, Sandia National Laboratories

! !AUTHORS: P. Luszczek, Sandia National Laboratories

! !AUTHORS: M. Schultz, Sandia National Laboratories

! !AUTHORS: A. Lumsdaine, Sandia National Laboratories

! !AUTHORS: I. Duff, Sandia National Laboratories

! !AUTHORS: S. Hamman, Sandia National Laboratories

! !AUTHORS: M. Heroux, Sandia National Laboratories

! !AUTHORS: S. Hoang, Sandia National Laboratories

! !AUTHORS: P. Luszczek, Sandia National Laboratories

! !AUTHORS: M. Schultz

=======

! gemv OpenMP subroutine (version 1.0) 
! 
! This file is part of the GEMV OpenMP Library 
!

=======

! gemv OpenMP subroutine (double precision)
! ----------------------------------------
!
! y := alpha*A*x + beta*y
!
! y and x are vectors and A is an m by n matrix
!
! Written by Jeff Hammond, 2014.
!

=======

! gemv OpenMP subroutine
! 
! This subroutine computes a matrix-vector product using OpenMP.
!
! Input parameters:
!   n   - size of the matrix
!   A   - matrix in column major format
!   x   - vector
!
! Output parameters:
!   y   - result vector
!
! Local parameters:
!   i   - loop index
!   j   - loop index
!   sum - partial sum

subroutine gemv(n, A, x, y)
  implicit none

  integer, intent(in) :: n
  real, intent(in) :: A(n,n)
  real, intent(in) :: x(n)
  real, intent(out) :: y(n)

  integer :: i, j
  real :: sum

  !$omp parallel do default(none) shared(n,A,x,y) private(i,j,sum)
  do i = 1, n
    sum = 0.0
    do j = 1, n
      sum = sum + A(i,j) * x(j)
    end do
    y(i) = sum
  end do
  !$omp end parallel do

end subroutine

! gemv OpenMP subroutine
! 
! This subroutine computes a matrix-vector product using OpenMP.
!
! Input parameters:
!   n   - size of the matrix
!   A   - matrix in column major format
!   x   - vector
!
! Output parameters:
!   y   - result vector
!
! Local parameters:
!   i   - loop index
!   j   - loop index
!   sum - partial sum

subroutine gemv(n, A, x, y)
  implicit none

  integer, intent(in) :: n
  real, intent(in) :: A(n,n)
  real, intent(in) :: x(n)
  real, intent(out) :: y(n)

  integer :: i, j
  real :: sum

  !$omp parallel do default(none) shared(n,A,x,y) private(i,j,sum)
  do i = 1, n
    sum = 0.0
    do j = 1, n
      sum = sum + A(i,j) * x(j)
    end do
    y(i) = sum
  end do
  !$omp end parallel do

end subroutine

! gemv OpenMP subroutine
! 
! This subroutine

=======

! gemv OpenMP subroutine
! 1.0  2017-11-01  Initial version
! 1.1  2018-01-01  Add thread affinity
! 1.2  2018-02-01  Add OpenMP scheduling
! 1.3  2018-03-01  Add OpenMP dynamic scheduling
! 1.4  2018-04-01  Add OpenMP chunk size
! 1.5  2018-05-01  Add OpenMP thread affinity
! 1.6  2018-06-01  Add OpenMP thread affinity
! 1.7  2018-07-01  Add OpenMP thread affinity
! 1.8  2018-08-01  Add OpenMP thread affinity
! 1.9  2018-09-01  Add OpenMP thread affinity
! 1.10 2018-10-01  Add OpenMP thread affinity
! 1.11 2018-11-01  Add OpenMP thread affinity
! 1.12 2018-12-01  Add OpenMP thread affinity
! 1.13 2019-01-01  Add OpenMP thread affinity
! 1.14 2019-02-01  Add OpenMP thread affinity
! 1.15 2019-03-01  Add OpenMP thread affinity
! 1.16 2019-04-01  Add OpenMP thread affinity
! 1.17 2019-05-01  Add OpenMP thread affinity
! 1.18 2019-06-01  Add OpenMP thread affinity
! 1.19 2019-07-01  Add OpenMP thread affinity
! 1.20 2019-08-01  Add OpenMP thread affinity
! 1.21 2019-09-01  Add OpenMP thread affinity
! 1.22 2019-10-01  Add OpenMP thread affinity
! 1.23 2019-11-01  Add OpenMP thread affinity
! 1.24 2019-12-01  Add OpenMP thread affinity
! 1.25 2020-01-01  Add OpenMP thread affinity
!

=======

! gemv OpenMP subroutine
!




! Prompt: gemv OpenACC subroutine
Synthesizing 6/10 solutions

=======

! gemv OpenACC subroutine (single precision)
subroutine gemv( n, alpha, A, x, beta, y )
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: alpha, beta
  real, intent(in) :: A(n,n), x(n)
  real, intent(inout) :: y(n)

  !$acc parallel loop present( A, x, y )
  do i = 1, n
    y(i) = beta * y(i)
    do j = 1, n
      y(i) = y(i) + alpha * A(i,j) * x(j)
    enddo
  enddo
  !$acc end parallel loop

end subroutine gemv

! gemv OpenACC subroutine (double precision)
subroutine gemv( n, alpha, A, x, beta, y )
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: alpha, beta
  real, intent(in) :: A(n,n), x(n)
  real, intent(inout) :: y(n)

  !$acc parallel loop present( A, x, y )
  do i = 1, n
    y(i) = beta * y(i)
    do j = 1, n
      y(i) = y(i) + alpha * A(i,j) * x(j)
    enddo
  enddo
  !$acc end parallel loop

end subroutine gemv

! gemv OpenACC subroutine (single precision)
subroutine gemv( n, alpha, A, x, beta, y )
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: alpha, beta
  real, intent(in) :: A(n,n), x(n)
  real, intent(inout) :: y(n)

  !$acc parallel loop present( A, x, y )
  do i = 1, n
    y(i) = beta * y(i)
    do j = 1, n
      y(i) = y(i) + alpha * A(i,j) * x(j)
    enddo
  enddo
  !$acc end parallel loop

end subroutine gemv

! gemv OpenACC subroutine (double precision)
subroutine gemv( n, alpha, A, x, beta, y )
  implicit none
  integer, intent(in) ::

=======

! gemv OpenACC subroutine (single precision) 
subroutine gemv ( n , alpha , A , x , beta , y ) 
   integer , intent ( in ) :: n 
   real , intent ( in ) :: alpha 
   real , intent ( in ) :: A ( n , n ) 
   real , intent ( in ) :: x ( n ) 
   real , intent ( in ) :: beta 
   real , intent ( in out ) :: y ( n ) 
   integer :: i , j 
   !$acc kernels 
   do i = 1 , n 
      y ( i ) = beta * y ( i ) 
      do j = 1 , n 
         y ( i ) = y ( i ) + alpha * A ( i , j ) * x ( j ) 
      end do 
   end do 
   !$acc end kernels 
end subroutine gemv

! gemv OpenACC subroutine (double precision) 
subroutine gemv ( n , alpha , A , x , beta , y ) 
   integer , intent ( in ) :: n 
   double precision , intent ( in ) :: alpha 
   double precision , intent ( in ) :: A ( n , n ) 
   double precision , intent ( in ) :: x ( n ) 
   double precision , intent ( in ) :: beta 
   double precision , intent ( in out ) :: y ( n ) 
   integer :: i , j 
   !$acc kernels 
   do i = 1 , n 
      y ( i ) = beta * y ( i ) 
      do j = 1 , n 
         y ( i ) = y ( i ) + alpha * A ( i , j ) * x ( j ) 
      end do 
   end do 
   !$acc end kernels 
end subroutine gemv

! gemv OpenACC subroutine (single precision) 
subroutine gemv ( n , alpha , A , x , beta , y ) 
   integer , intent ( in ) :: n 
   real , intent ( in ) :: alpha 
   real , intent ( in ) :: A ( n , n ) 
   real , intent ( in ) :: x ( n ) 
   real , intent ( in ) :: beta 
   real , intent ( in out ) :: y ( n ) 
   integer :: i , j 
   !$acc kernels 
   do i = 1 , n 
      y (

=======

! gemv OpenACC subroutine (version 1.0) 
 subroutine gemv ( n, a, x, y ) 
   integer, intent(in) :: n 
   real, intent(in) :: a(n,n), x(n) 
   real, intent(inout) :: y(n) 
   integer i, j 
   !$acc kernels 
   do j = 1, n 
     do i = 1, n 
       y(i) = y(i) + a(i,j) * x(j) 
     enddo 
   enddo 
   !$acc end kernels 
 end subroutine gemv

The OpenACC compiler will automatically transform the code to execute on the GPU.

To compile and run the program, first load the PGI compiler:

$ module load pgi

Then, compile the program:

$ pgfortran -acc -Minfo=accel -o gemv gemv.f90

The -Minfo=accel option will print out information about the transformations that th

=======

! gemv OpenACC subroutine (single precision)
subroutine gemv(A, x, y, m, n)
    use acc_lib
    implicit none
    real, dimension(m,n), intent(in) :: A
    real, dimension(n), intent(in) :: x
    real, dimension(m), intent(inout) :: y
    integer, value :: m, n
    integer :: i, j, chunksize, num_chunks
    real, dimension(4) :: tmp
    !$acc parallel loop gang vector_length(4) &
    !$acc& private(i,j,tmp) &
    !$acc& reduction(+:y(1:m))
    do i = 1, m
        tmp = 0.0
        do j = 1, n
            tmp = tmp + A(i,j) * x(j)
        end do
        y(i) = y(i) + sum(tmp)
    end do
end subroutine gemv

! gemv OpenACC subroutine (double precision)
subroutine gemv(A, x, y, m, n)
    use acc_lib
    implicit none
    real(dp), dimension(m,n), intent(in) :: A
    real(dp), dimension(n), intent(in) :: x
    real(dp), dimension(m), intent(inout) :: y
    integer, value :: m, n
    integer :: i, j, chunksize, num_chunks
    real(dp), dimension(2) :: tmp
    !$acc parallel loop gang vector_length(2) &
    !$acc& private(i,j,tmp) &
    !$acc& reduction(+:y(1:m))
    do i = 1, m
        tmp = 0.0
        do j = 1, n
            tmp = tmp + A(i,j) * x(j)
        end do
        y(i) = y(i) + sum(tmp)
    end do
end subroutine gemv

! gemv OpenACC subroutine (single precision)
subroutine gemv(A, x, y, m, n)
    use acc_lib
    implicit none
    real, dimension(m,n), intent(in) :: A
    real, dimension(n), intent(in) :: x
    real, dimension(m), intent(inout) :: y
    integer, value :: m, n
    integer :: i, j, chunksize, num_chunks
    real, dimension(4)

=======

! gemv OpenACC subroutine (single precision) 
subroutine gemv_openacc (A, x, y, m, n)
  implicit none
  integer, intent(in) :: m, n
  real(4), intent(in) :: A(m, n), x(n)
  real(4), intent(inout) :: y(m)
  !$acc parallel loop present (A, x, y)
  do i = 1, m
    y(i) = 0.0
    do j = 1, n
      y(i) = y(i) + A(i, j) * x(j)
    enddo
  enddo
end subroutine gemv_openacc

! gemv OpenMP subroutine (single precision) 
subroutine gemv_openmp (A, x, y, m, n)
  implicit none
  integer, intent(in) :: m, n
  real(4), intent(in) :: A(m, n), x(n)
  real(4), intent(inout) :: y(m)
  !$omp parallel do shared (A, x, y)
  do i = 1, m
    y(i) = 0.0
    do j = 1, n
      y(i) = y(i) + A(i, j) * x(j)
    enddo
  enddo
end subroutine gemv_openmp

! gemv subroutine (single precision) 
subroutine gemv (A, x, y, m, n)
  implicit none
  integer, intent(in) :: m, n
  real(4), intent(in) :: A(m, n), x(n)
  real(4), intent(inout) :: y(m)
  do i = 1, m
    y(i) = 0.0
    do j = 1, n
      y(i) = y(i) + A(i, j) * x(j)
    enddo
  enddo
end subroutine gemv

! gemv OpenACC subroutine (double precision) 
subroutine gemv_openacc (A, x, y, m, n)
  implicit none
  integer, intent(in) :: m, n
  real(8), intent(in) :: A(m, n), x(n)
  real(8), intent(inout) :: y(m)
  !$acc parallel loop present (A, x,

=======

! gemv OpenACC subroutine gemv ( n, alpha, A, x, beta, y ) ! ! Purpose: ! ! y = alpha * A * x + beta * y ! ! Parameters: ! ! n - number of rows and columns of matrix A ! alpha - scalar multiplier ! A - matrix ! x - vector ! beta - scalar multiplier ! y - vector ! !---------------------------------------------------------------------
