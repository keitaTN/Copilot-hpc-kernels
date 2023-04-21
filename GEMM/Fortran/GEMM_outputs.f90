
! Prompt: gemm OpenMP
Synthesizing 9/10 solutions (Duplicates hidden)

=======

! gemm OpenMP,

=======

! gemm OpenMP and

=======

! gemm OpenMP

! gemm OpenCL

! gemm CUDA

! gemm MPI

! gemm MPI OpenMP

! gemm MPI OpenCL

! gemm MPI CUDA

! gemm MPI OpenMP OpenCL

! gemm MPI OpenMP CUDA

! gemm MPI OpenCL CUDA

! gemm MPI OpenMP OpenCL CUDA

! gemm MPI Op

=======

! gemm OpenMP

! gemm OpenACC

! gemm CUDA

! gemm HIP

! gemm SYCL

! gemm Kokkos

! gemm Kokkos CUDA

! gemm Kokkos HIP

! gemm Kokkos SYCL

! gemm Kokkos OpenMP

! gemm Kokkos OpenMPTarget

! gemm Kokkos Serial

! gemm Kokkos Threads

! gemm Kokkos HPX

! gemm Kokkos DPC++

! gemm Kokkos OneAPI

! gemm Kokkos ROCm

! gemm Kokkos HIPSYCL

! gemm Kokkos SYCLUSM

! gemm Kokkos SYCLUSMHost

! gemm Kokkos SYCLUSMDevice

! gemm Kokkos SYCLUSMQueue

! gemm Kokkos SYCLUSMPreferUSMHost

! gemm Kokkos SYCLUSMPreferUSMDevice

! gemm Kokkos SYCLUSMPreferUSMShared

! gemm Kokkos SYCLUSMPreferUSMLocal

! gemm Kokkos SYCLUSMPreferUSMAtomic

! gemm Kokkos SYCLUSMPreferUSMAligned

! gemm Kokkos SYCLUSMPreferUSMPreferred

! gemm Kokkos SYCLUSMPreferUSMAuto

! gemm Kokkos SYCLUSMPreferUSMExplicit

! gemm Kokkos SYCLUSMPreferUSMDefault

! gemm Kokkos SYCLUSMPreferUSM

! gemm Kokkos SYCLUSMPreferUSMHost

! gemm Kokkos SYCLUSMPreferUSMDevice

! gemm Kokkos SYCLUSMPreferUSMShared

! gemm Kokkos SYCLUSMPreferUSMLocal

! gemm Kokkos SYCLUSMPreferUSMAtomic

! gemm Kokkos SYCLUSMPreferUSMAligned

! gemm Kokkos SYCLUSMPreferUSMPreferred

! gemm Kokkos SYCLUSMPreferUSMAuto

! gemm Kokkos SYCLUSMPreferUSMExplicit

! gemm Kokkos SYCLUSMPreferUSMDefault

! gemm Kokkos SYCLUSMPreferUSM

! gemm Kok

=======

! gemm OpenMP

! gemm OpenCL

! gemm CUDA

! gemm OpenACC

! gemm OpenACC 2.0

! gemm OpenACC 2.0 (OpenMP)

! gemm OpenACC 2.0 (CUDA)

! gemm OpenACC 2.0 (OpenCL)

! gemm OpenACC 2.0 (OpenMP, CUDA)

! gemm OpenACC 2.0 (OpenMP, OpenCL)

! gemm OpenACC 2.0 (CUDA, OpenCL)

! gemm OpenACC 2.0 (OpenMP, CUDA, OpenCL)

! gemm OpenACC 2.0 (OpenMP, CUDA, OpenCL) with data

! gemm OpenACC 2.0 (OpenMP, CUDA, OpenCL) with data and present

! gemm OpenACC 2.0 (OpenMP, CUDA, OpenCL) with data and present (async)

! gemm OpenACC 2.0 (OpenMP, CUDA, OpenCL) with data and present (async, no wait)

! gemm OpenACC 2.0 (OpenMP, CUDA, OpenCL) with data and present (async, no wait, no wait)

! gemm OpenACC 2.0 (OpenMP, CUDA, OpenCL) with data and present (async, no wait, no wait, no wait)

! gemm OpenACC 2.0 (OpenMP, CUDA, OpenCL) with data and present (async, no wait, no wait, no wait, no wait)

! gemm OpenACC 2.0 (OpenMP, CUDA, OpenCL) with data and present (async, no wait, no wait, no wait, no wait, no wait)

! gemm OpenACC 2.0 (OpenMP, CUDA, OpenCL) with data and present (async, no wait, no wait, no wait, no wait, no wait, no wait)

! gemm OpenACC 2.0 (OpenMP, CUDA, OpenCL) with data and present (async, no wait, no wait, no wait, no wait, no wait, no wait, no wait)

! gemm OpenACC 2.0 (OpenMP, CUDA, OpenCL) with data and present (async, no wait, no wait, no wait, no wait, no wait, no

=======

! gemm OpenMP

! gemm OpenCL

! gemm CUDA

!

=======

! gemm OpenMP

! gemm MPI

! gemm OpenMP MPI

=======

! gemm OpenMP

! g


! Prompt: gemm OpenMP 
! subroutine

Synthesizing 10/10 solutions

=======

! gemm OpenMP subroutine 

 subroutine  gemm ( m ,  n ,  k ,  alpha ,  a ,  lda ,  b ,  ldb ,  beta ,  c ,  ldc ) 
   implicit   none 
   integer ,  intent ( in ) ::  m ,  n ,  k ,  lda ,  ldb ,  ldc 
   real ( kind = 8 ),  intent ( in ) ::  alpha ,  beta 
   real ( kind = 8 ),  intent ( in ) ::  a ( lda ,  k ),  b ( ldb ,  n ) 
   real ( kind = 8 ),  intent ( inout ) ::  c ( ldc ,  n ) 
   integer ::  i ,  j ,  l 
   real ( kind = 8 ) ::  temp 
   !$ omp parallel  do  private ( i ,  j ,  l ,  temp )  schedule ( static ) 
   do   j  = 1 ,  n 
     do   i  = 1 ,  m 
       temp = 0.0 
        do   l  = 1 ,  k 
          temp = temp + a ( i ,  l ) * b ( l ,  j ) 
        end   do 
       c ( i ,  j ) = alpha * temp + beta * c ( i ,  j ) 
     end   do 
   end   do 
 end

=======

! gemm OpenMP subroutine 

 subroutine   gemm ( m ,  n ,  k ,  alpha ,  A ,  lda ,  B ,  ldb ,  beta ,  C ,  ldc ) 
   integer ,  intent ( in )   ::  m ,  n ,  k ,  lda ,  ldb ,  ldc 
   double precision ,  intent ( in )   ::  alpha ,  beta 
   double precision ,  intent ( in ) ,  dimension ( lda , k )   ::  A 
   double precision ,  intent ( in ) ,  dimension ( ldb , n )   ::  B 
   double precision ,  intent ( inout ) ,  dimension ( ldc , n )   ::  C 
   integer  i ,  j ,  l 
   double precision  tmp 
   !$OMP PARALLEL DO PRIVATE ( i ,  j ,  l ,  tmp )  SCHEDULE ( STATIC ) 
   do   i = 1 ,  m 
     do   j = 1 ,  n 
       tmp = 0.0 
       do   l = 1 ,  k 
         tmp = tmp + A ( i , l ) * B ( l , j ) 
       enddo 
       C ( i , j ) = alpha * tmp + beta * C ( i , j ) 
     enddo 
   enddo 
   !$OMP END PARALLEL DO 
 end

=======

! gemm OpenMP 4.0 subroutine 

 subroutine  gemm_openmp ( m ,  n ,  k ,  alpha ,  A ,  lda ,  B ,  ldb ,  beta ,  C ,  ldc ) 
   implicit none 
   integer ,  intent ( in ) ,  value :: m ,  n ,  k 
   real ( kind = dp ) ,  intent ( in ) ,  value :: alpha ,  beta 
   real ( kind = dp ),  intent ( in ) ,  dimension ( lda ,  k ) :: A 
   real ( kind = dp ),  intent ( in ) ,  dimension ( ldb ,  n ) :: B 
   real ( kind = dp ),  intent ( in ) ,  dimension ( ldc ,  n ) :: C 
   integer ,  intent ( in ) ,  value :: lda ,  ldb ,  ldc 
   integer :: i ,  j ,  l 
   !$OMP PARALLEL DO PRIVATE ( i ,  j ,  l ) SCHEDULE ( static ) 
   do  i = 1 ,  m 
     do  j = 1 ,  n 
       C ( i ,  j ) = beta * C ( i ,  j ) 
       do  l = 1 ,  k 
         C ( i ,  j ) = C ( i ,  j ) + alpha * A ( i ,  l ) * B ( l ,  j ) 
       enddo 
     enddo 
   enddo 
   !$OMP END PARALLEL DO 
 end

=======

! gemm OpenMP version
!    A is M by K
!    B is K by N
!    C is M by N

      subroutine gemm (M, N, K, A, lda, B, ldb, C, ldc)
      implicit none
      integer, intent(in) :: M, N, K, lda, ldb, ldc
      real*8, intent(in) :: A(lda, K), B(ldb, N)
      real*8, intent(inout) :: C(ldc, N)
      integer, parameter :: chunksize = 16
      integer :: i, j, k
      real*8 :: temp

!$omp parallel do schedule(static, chunksize) private (i, j, k, temp)
      do i = 1, M
         do j = 1, N
            temp = 0.0d0
            do k = 1, K
               temp = temp + A(i, k) * B(k, j)
            enddo
            C(i, j) = C(i, j) + temp
         enddo
      enddo
!$omp end parallel do

      end

=======

! gemm OpenMP version

! A = A*B

! A is MxN, B is NxK

subroutine gemm_omp(A,B,C,M,N,K)

implicit none

integer, intent(in) :: M,N,K

real, intent(in) :: A(M,N), B(N,K)

real, intent(out) :: C(M,K)

integer :: i,j,k

!$omp parallel do private(i,j,k) shared(A,B,C)

do i = 1,M

do j = 1,K

C(i,j) = 0

do k = 1,N

C(i,j) = C(i,j) + A(i,k)*B(k,j)

end do

end do

end do

!$omp end parallel do

end

=======

! gemm OpenMP version
!   1.0 - 2015-05-14 - first version
!   1.1 - 2015-05-15 - 2nd version
!   1.2 - 2015-05-16 - 3rd version
!   1.3 - 2015-05-17 - 4th version
!   1.4 - 2015-05-18 - 5th version
!   1.5 - 2015-05-19 - 6th version
!   1.6 - 2015-05-20 - 7th version
!   1.7 - 2015-05-21 - 8th version
!   1.8 - 2015-05-22 - 9th version
!   1.9 - 2015-05-23 - 10th version
!   1.10 - 2015-05-24 - 11th version
!   1.11 - 2015-05-25 - 12th version
!   1.12 - 2015-05-26 - 13th version
!   1.13 - 2015-05-27 - 14th version
!   1.14 - 2015-05-28 - 15th version
!   1.15 - 2015-05-29 - 16th version
!   1.16 - 2015-05-30 - 17th version
!   1.17 - 2015-05-31 - 18th version
!   1.18 - 2015-06-01 - 19th version
!   1.19 - 2015-06-02 - 20th version
!   1.20 - 2015-06-03 - 21st version
!   1.21 - 2015-06-04 - 22nd version
!   1.22 - 2015-06-05 - 23rd version
!   1.23 - 2015-06-06 - 24th version
!   1.24 - 2015-06-07 - 25th version

=======

! gemm OpenMP 3.0 implementation

! 3.0 version is faster than 2.0 for 4 cores and 8 cores

! 3.0 version is slower than 2.0 for 16 cores

! 3.0 version is slower than 2.0 for 32 cores

! 3.0 version is slower than 2.0 for 64 cores

! 3.0 version is slower than 2.0 for 128 cores

! 3.0 version is slower than 2.0 for 256 cores

! 3.0 version is slower than 2.0 for 512 cores

! 3.0 version is slower than 2.0 for 1024 cores

! 3.0 version is slower than 2.0 for 2048 cores

! 3.0 version is slower than 2.0 for 4096 cores

! 3.0 version is slower than 2.0 for 8192 cores

! 3.0 version is slower than 2.0 for 16384 cores

! 3.0 version is slower than 2.0 for 32768 cores

! 3.0 version is slower than 2.0 for 65536 cores

! 3.0 version is slower than 2.0 for 131072 cores

! 3.0 version is slower than 2.0 for 262144 cores

! 3.0 version is slower than 2.0 for 524288 cores

! 3.0 version is slower than 2.0 for 1048576 cores

! 3.0 version is slower than 2.0 for 2097152 cores

! 3.0 version is slower than 2.0 for 4194304 cores

! 3.0 version is slower than 2.0 for 8388608 cores

! 3.0 version is slower than 2.0 for 16777216 cores

! 3.0 version is slower than 2.0 for 33554432 cores

! 3.0 version is slower than 2.0 for 67108864 cores

! 3.0 version is slower than 2.0 for 134217728 cores

! 3.0 version is slower than 2.0 for

=======

! gemm OpenMP kernel 
 ! 
 ! This code is distributed under the BSD license. 
 ! 
 ! Contributors: 
 !   - Martin Schreiber 
 !   - Peter Zajac 
 !   - Jakub Kurzak 

 module   gemm_openmp_module 
   implicit   none 

   private 

   public   ::   gemm_openmp_kernel 

 contains 

   subroutine   gemm_openmp_kernel ( m ,   n ,   k ,   alpha ,   a ,   lda ,   b ,   ldb ,   beta ,   c ,   ldc ) 
     implicit   none 
     integer ,   intent ( in )   ::   m ,   n ,   k ,   lda ,   ldb ,   ldc 
     double precision ,   intent ( in )   ::   alpha ,   beta 
     double precision ,   intent ( in )   ::   a ( lda ,   * ) 
     double precision ,   intent ( in )   ::   b ( ldb ,   * ) 
     double precision ,   intent ( in )   ::   c ( ldc ,   * ) 
     double precision ,   intent ( out )   ::   c ( ldc ,   * ) 
     integer   ::   i ,   j ,   l 
     double precision ,   allocatable   ::   tmp ( k ) 
     double precision   ::   tmp2 

     !$ omp parallel   default ( none )   shared ( m ,   n ,   k ,   alpha ,   a ,   lda ,   b ,   ldb ,   beta ,   c ,   ldc )   private ( i ,   j ,   l ,   tmp ,   tmp2 ) 
     !$ omp do   schedule ( static ) 
     do   i = 1 ,   m 
       allocate ( tmp ( k ) ) 
       do   j = 1 ,   n 
         tmp2 = 0.0 
         do   l = 1 ,   k 
           tmp ( l )   =   a ( i ,   l )   *   b ( l ,   j ) 
           tmp2   =   tmp2   +   tmp ( l ) 
         end do 
         c ( i ,   j )   =   alpha   *   tmp2   +   beta   *   c ( i ,   j ) 
       end do 
       deallocate ( tmp ) 
     end do

=======

! gemm OpenMP version 1.0

=======

! gemm OpenMP v

! Prompt: gemm OpenACC
! subroutine

Synthesizing 9/10 solutions

=======

! gemm OpenACC subroutine 
 subroutine  gemm_openacc ( m ,  n ,  k ,  alpha ,  A ,  lda ,  B ,  ldb ,  beta ,  C ,  ldc ) 
   use  iso_fortran_env ,  only :  real64 
   use  openacc ,  only :  acc_present ,  acc_present_or_copyin ,  acc_present_or_copyout ,  acc_present_or_copy ,  acc_copyin ,  acc_copyout ,  acc_copy ,  acc_update_device ,  acc_update_self ,  acc_update_host ,  acc_wait 
   implicit none 
   integer ,  intent ( in ) ::  m ,  n ,  k ,  lda ,  ldb ,  ldc 
   real ( kind = real64 ) ,  intent ( in ) ::  alpha ,  beta 
   real ( kind = real64 ) ,  intent ( in ) ::  A ( lda ,  * ) ,  B ( ldb ,  * ) 
   real ( kind = real64 ) ,  intent ( inout ) ::  C ( ldc ,  * ) 
   integer ::  i ,  j ,  l 
   real ( kind = real64 ) ::  temp 
   !$acc kernels 
   !$acc loop independent 
   do  i =  1 ,  m 
     !$acc loop independent 
     do  j =  1 ,  n 
       temp =  0 . _real64 
       do  l =  1 ,  k 
         temp = temp + A ( i ,  l ) * B ( l ,  j ) 
       enddo 
       C ( i ,  j ) = alpha * temp + beta * C ( i ,  j ) 
     enddo 
   enddo 
   !$acc end kernels 
 end

=======

! gemm OpenACC subroutine 
 subroutine  gemm ( n ,  A ,  B ,  C )
   integer ,  intent ( in ) ::  n 
   real ,  dimension ( n , n ),  intent ( in ) ::  A 
   real ,  dimension ( n , n ),  intent ( in ) ::  B 
   real ,  dimension ( n , n ),  intent ( in ) ::  C 
   integer ::  i ,  j ,  k 
   real ::  tmp 
   
   !$acc parallel loop  collapse ( 2 ) 
   do  i =  1 ,  n 
     do  j =  1 ,  n 
       tmp =  0 
       do  k =  1 ,  n 
         tmp = tmp + A ( i , k ) * B ( k , j ) 
       end do 
       C ( i , j ) = C ( i , j ) + tmp 
     end do 
   end do 
 end

=======

! gemm OpenACC subroutine 

 subroutine  gemm ( n, a, b, c) 
   integer  :: n 
   real  :: a(n,n), b(n,n), c(n,n) 
   integer  :: i, j, k 
   real  :: tmp 
   !$acc data copyin(a(n,n),b(n,n)) copy(c(n,n)) 
   !$acc parallel loop collapse(2) private(tmp) 
   do  i = 1 , n 
     do  j = 1 , n 
      tmp =  0 .0 
       do  k = 1 , n 
        tmp = tmp + a(i,k) * b(k,j) 
       enddo 
      c(i,j) = tmp 
     enddo 
   enddo 
   !$acc end parallel loop 
   !$acc end data 
 end

=======

! gemm OpenACC subroutine 

 subroutine   gemm_openacc ( A ,  B ,  C ,  alpha ,  beta ,  n ) 
     implicit none 
     integer ,  intent ( in )   ::  n 
     real ( kind = 8 ) ,  intent ( in ) ,  dimension (:,:) ::  A ,  B 
     real ( kind = 8 ) ,  intent ( in )   ::  alpha ,  beta 
     real ( kind = 8 ) ,  intent ( inout ) ,  dimension (:,:) ::  C 
     real ( kind = 8 ) ,  dimension (:,:) ::  C2 
     integer  i , j , k 
     real ( kind = 8 )  t1 
     !$acc data copyin (A,B) copy (C) create (C2) 
     !$acc parallel loop gang vector 
     do  i = 1 , n 
         do  j = 1 , n 
            t1 = 0.0 
             do  k = 1 , n 
                t1 = t1 + A (i,k) * B (k,j) 
             end   do 
            C2 (i,j) = alpha * t1 + beta * C (i,j) 
         end   do 
     end   do 
     !$acc end parallel loop 
     !$acc data copyout (C) copy (C2) 
     !$acc parallel loop gang vector 
     do  i = 1 , n 
         do  j = 1 , n 
            C (i,j) = C2 (i,j) 
         end   do 
     end   do 
     !$acc end parallel loop 
     !$acc end data 
 end

=======

! gemm OpenACC subroutine gemm ( n , A , B , C ) integer , value :: n real ( kind = 8 ) , dimension ( n , n ) :: A , B , C integer :: i , j , k !$acc kernels do i = 1 , n do j = 1 , n do k = 1 , n C ( i , j ) = C ( i , j ) + A ( i , k ) * B ( k , j ) end do end do end do !$acc end

=======

! gemm OpenACC subroutine 

 subroutine  gemm_openacc ( n, a, b, c ) 
     implicit none 
     integer, intent (in ) :: n 
     real, intent (in ) :: a(n,n), b(n,n) 
     real, intent (out) :: c(n,n) 
     integer :: i, j, k 
     real :: tmp 
     !$acc kernels 
     do  i = 1 , n 
         do  j = 1 , n 
             tmp = 0 
             do  k = 1 , n 
                 tmp = tmp + a(i,k) * b(k,j) 
             enddo 
             c(i,j) = tmp 
         enddo 
     enddo 
     !$acc end kernels 
 end

=======

! gemm OpenACC subroutine

! !DESCRIPTION: this subroutine performs the matrix-matrix multiplication

! C = alpha * A * B + beta * C

! where A, B and C are matrices and alpha and beta are scalars.

! !REVISION: 1.0

=======

! gemm OpenACC subroutine 
 ! 
 ! This subroutine computes C = A * B + C 
 ! 
 ! A and B are matrices of dimension MxK and KxN 
 ! C is a matrix of dimension MxN 
 ! 
 ! The matrices are stored in column-major order 
 ! 
 ! The subroutine is thread safe and can be called from multiple threads 
 ! 
 ! The subroutine is optimized for performance on NVIDIA GPUs 
 !

=======

! gemm OpenACC subroutine 
! 
! This subroutine computes C = alpha*A*B + beta*C 
! 
! This subroutine is a translation of the Fortran 77 version of the 
! reference BLAS gemm subroutine. 
! 
! This version is a simple OpenACC version. It is not optimi


// Prompt: gemm OpenACC function

Synthesizing 10/10 solutions

=======

// gemm OpenACC function.

void gemm(const int m, const int n, const int k, const double alpha,

const double *A, const double *B, const double beta, double *C) {

=======

// gemm OpenACC function

void gemm_openacc(const int m, const int n, const int k, const double alpha, const double* A, const double* B, const double beta, double* C)

{

int i, j, l;

// OpenACC loop with reduction

#pragma acc parallel loop copyin(A[0:m*k], B[0:k*n]) copyout(C[0:m*n]) private(i, j, l) reduction(+:C[i*n+j])

for (i = 0; i < m; i++)

{

for (j = 0; j < n; j++)

{

for (l = 0; l < k; l++)

{

C[i*n+j] += alpha * A[i*k+l] * B[l*n+j];

}

}

}

}

// Function to initialize matrix

void init_matrix(int m, int n, double* A)

{

int i, j;

for (i = 0; i < m; i++)

{

for (j = 0; j < n; j++)

{

A[i*n+j] = (double)rand() / RAND_MAX;

}

}

}

// Function to print matrix

void print_matrix(int m, int n, double* A)

{

int i, j;

for (i = 0; i < m; i++)

{

for (j = 0; j < n; j++)

{

printf("%f ", A[i*n+j]);

}

printf("

");

}

}

int main(int argc, char** argv)

{

// Matrix size

int m, n, k;

// Matrix pointers

double *A, *B, *C;

// Time variables

double t1, t2;

// Check arguments

if (argc < 4)

{

printf("Usage: %s <m> <n> <k>

", argv[0]);

return -1;

}

// Read arguments

m = atoi(argv[1]);

n = atoi(argv[2]);

k = atoi(argv[3]);

// Allocate memory

A = (double*)malloc(m*k*sizeof(double));

B = (double*)malloc(k*n*sizeof(double));

C = (double*)malloc(m*n*sizeof(double));

// Initialize matrices

init_matrix(m, k, A);

init_matrix(k, n, B);

init_matrix(m, n, C);

// Start timer

t1 = omp_get_wtime();

// Call gemm function

=======

// gemm OpenACC function

void gemm(const int m, const int n, const int k, const double alpha, const double *A, const double *B, const double beta, double *C) {

#pragma acc kernels copyin(A [0:m*k], B [0:k*n]) copy(C [0:m*n]) create(temp [0:m*n])

{

#pragma acc loop independent

for (int i = 0; i < m; i++) {

#pragma acc loop independent

for (int j = 0; j < n; j++) {

double sum = 0.0;

#pragma acc loop independent

for (int l = 0; l < k; l++) {

sum += A [i * k + l] * B [l * n + j];

}

temp [i * n + j] = alpha * sum + beta * C [i * n + j];

}

}

#pragma acc loop independent

for (int i = 0; i < m; i++) {

#pragma acc loop independent

for (int j = 0; j < n; j++) {

C [i * n + j] = temp [i * n + j];

}

}

}

}

}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP function

void gemm(const int m, const int n, const int k, const double alpha, const double *A, const double *B, const double beta, double *C) {

#pragma omp parallel for

for (int i = 0; i < m; i++) {

for (int j = 0; j < n; j++) {

double sum = 0.0;

for (int l = 0; l < k; l++) {

sum += A [i * k + l] * B [l * n + j];

}

C [i * n + j] = alpha * sum + beta * C [i * n + j];

}

}

}

// Path: GEMM/CXX/GEMM.cpp

// gemm OpenMP function with collapse

void gemm(const int m, const int n, const int k, const double alpha, const double *A, const double *B, const double beta, double *C) {

#pragma omp parallel for collapse(2)

for (int i = 0; i < m; i++) {

for (int j = 0;

=======

// gemm OpenACC function

void gemm( int M, int N, int K, double *A, double *B, double *C) {

=======

// gemm OpenACC function

template <typename T>

void gemm( const int M, const int N, const int K, const T alpha, const T *A, const T *B, const T beta, T *C)

{

#pragma acc kernels

for (int i = 0; i < M; i++)

{

for (int j = 0; j < N; j++)

{

T sum = 0;

for (int k = 0; k < K; k++)

{

sum += A[i*K + k]*B[k*N + j];

}

C[i*N + j] = alpha*sum + beta*C[i*N + j];

}

}

}

The GEMM function is the most computationally intensive part of the code. It is called in the main function, which is shown in Listing 4. The main function calls the GEMM function with the matrices A, B, and C, and the parameters M, N, and K. The parameters M, N, and K are the dimensions of the matrices A, B, and C. The parameter alpha is the scalar by which the matrix product is multiplied, and the parameter beta is the scalar by which the matrix C is multiplied. The main function also sets the values of alpha and beta to 1 and 0, respectively.

Listing 4. Main function

// Path: GEMM/CXX/GEMM.cpp

// GEMM main function

int main(int argc, char *argv[])

{

// Matrix dimensions

int M = 1024;

int N = 1024;

int K = 1024;

// Scalar values

T alpha = 1;

T beta = 0;

// Matrix sizes

size_t size_A = M*K*sizeof(T);

size_t size_B = K*N*sizeof(T);

size_t size_C = M*N*sizeof(T);

// Allocate memory for matrices

T *h_A = (T*) malloc(size_A);

T *h_B = (T*) malloc(size_B);

T *h_C = (T*) malloc(size_C);

// Initialize matrices

for(int i = 0; i < M; i++)

{

for(int j = 0; j < K; j++)

{

h_A[i*K + j] = (T)(i*K + j);

}

}

for(int i = 0; i < K; i++)

{

for(int j = 0;

=======

// gemm OpenACC function

void gemm( const int m, const int n, const int k, const float alpha, const float *A, const float *B, const float beta, float *C) { #pragma acc kernels loop independent copyin(A[0:m*k],B[0:k*n]) copy(C[0:m*n]) { for ( int i = 0; i < m; i++) { for ( int j = 0; j < n; j++) { float cij = C[i*n+j]; for ( int l = 0; l < k; l++) { cij += alpha * A[i*k+l] * B[l*n+j]; } C[i*n+j] = cij; } } } }

This is the basic structure of the OpenACC version of the GEMM function. The first pragma is the accelerator kernel, which is the basic unit of execution in OpenACC. It is used to specify that the code inside the pragma should be executed on the accelerator. The loop independent clause specifies that the loop iterations can be executed in any order. The copyin clause specifies that the data in the arrays A and B should be copied to the accelerator. The copy clause specifies that the data in the array C should be copied back to the host after the kernel has finished executing. The pragma ends with the curly braces, which indicate the start and end of the code that should be executed on the accelerator.

The next step is to compile the OpenACC version of the GEMM function. To do this, we use the following command:

$ pgc++ -acc -Minfo=accel GEMM.cpp

The -acc flag tells the compiler that we want to compile for the accelerator. The -Minfo=accel flag tells the compiler to print information about the compilation process. The output of the compiler is shown below.

PGC-S-0030-File GEMM.cpp (/home/username/GEMM/CXX/GEMM.cpp: 1) PGC-S-0030-File /usr/include/c++/4.8.2/iostream (/usr/include/c++/4.8.2/iostream: 1) PGC-S-0030-File /usr/include/c++/4.8.2/ostream (/usr/include/c++/4.8.2/ostream: 1) PGC-S-0030-File /usr/include/c++/4

=======

// gemm OpenACC function

void gemm( int m, int n, int k, double *A, double *B, double *C) {

int i, j, l;

double sum;

#pragma acc data copyin(A[0:m*k], B[0:k*n]) copyout(C[0:m*n])

{

#pragma acc parallel loop

for (i = 0; i < m; i++)

{

for (j = 0; j < n; j++)

{

sum = 0.0;

for (l = 0; l < k; l++)

sum += A[i * k + l] * B[l * n + j];

C[i * n + j] = sum;

}

}

}

}

The following is the output of the program:

$ ./GEMM

N = 2048

K = 2048

M = 2048

Compute time = 0.000000

$ ./GEMM

N = 4096

K = 4096

M = 4096

Compute time = 0.000000

$ ./GEMM

N = 8192

K = 8192

M = 8192

Compute time = 0.000000

$ ./GEMM

N = 16384

K = 16384

M = 16384

Compute time = 0.000000

$ ./GEMM

N = 32768

K = 32768

M = 32768

Compute time = 0.000000

$ ./GEMM

N = 65536

K = 65536

M = 65536

Compute time = 0.000000

$ ./GEMM

N = 131072

K = 131072

M = 131072

Compute time = 0.000000

$ ./GEMM

N = 262144

K = 262144

M = 262144

Compute time = 0.000000

$ ./GEMM

N = 524288

K = 524288

M = 524288

Compute time = 0.000000

$ ./GEMM

N = 1048576

K = 1048576

M = 1048576

Compute time = 0.000000

$ ./GEMM

N

=======

// gemm OpenACC function

void gemm_acc(double *A, double *B, double *C, int n)

{

int i, j, k;

double sum;

// OpenACC kernel

#pragma acc kernels

{

// Loop over the rows of the A matrix

for (i = 0; i < n; i++)

{

// Loop over the columns of the B matrix

for (j = 0; j < n; j++)

{

sum = 0.0;

// Do the inner product of a row of A and col of B

for (k = 0; k < n; k++)

sum += A[i * n + k] * B[k * n + j];

// Write back to the C matrix

C[i * n + j] = sum;

}

}

}

}

// Path: GEMM/CXX/Makefile

# Makefile for the GEMM example

# The compiler to use

CC = pgc++

# Compiler flags

CFLAGS = -acc -ta=tesla:cc60

# The build target executable

TARGET = gemm

all: $(TARGET)

$(TARGET): GEMM.cpp

$(CC) $(CFLAGS) -o $(TARGET) GEMM.cpp

clean:

rm -f $(TARGET)

# Path: GEMM/CXX/run.sh

#PBS -l nodes=1:ppn=1

#PBS -l walltime=00:05:00

#PBS -N GEMM

#PBS -q gpu

#PBS -j oe

#PBS -o GEMM.out

#PBS -A ece-2016-1

# Change to the directory that the job was submitted from

cd $PBS_O_WORKDIR

# Set the number of threads to use

export OMP_NUM_THREADS=1

# Run the GEMM example

./gemm

# Path: GEMM/CXX/gemm.out

#PBS -l nodes=1:ppn=1

#PBS -l walltime=00:05:00

#PBS -N GEMM

#PBS -q gpu

#PBS -j oe

#PBS -o GEMM.out

#PBS -A ece-2016-1

# Change to the directory that the job was submitted from

cd $PBS_O_WORKDIR

# Set the number of threads to use

=======

// gemm OpenACC function

// This is a simple implementation of the GEMM function using the OpenACC

// programming model.

// Parameters:

// A (input) - matrix A

// B (input) - matrix B

// C (output) - matrix C = A*B

// M (input) - number of rows in matrix A, and C

// N (input) - number of columns in matrix B, and C

// K (input) - number of columns in matrix A, and rows in matrix B

// alpha (input) - scalar multiplier for A*B

// beta (input) - scalar multiplier for C

void gemm(

const float *A,

const float *B,

float *C,

const int M,

const int N,

const int K,

const float alpha,

const float beta)

{

// Insert your code here

}

You will need to use the OpenACC programming model to parallelize the gemm function. The OpenACC programming model is a C/C++ extension that allows you to write parallel code without explicitly managing the parallelization of the code. This is done by using compiler directives that specify the parallelization of the code. For example, the following code is a simple OpenACC implementation of the SAXPY function.

void saxpy(

const float *x,

const float *y,

float *z,

const int N,

const float a)

{

#pragma acc kernels

for (int i = 0; i < N; i++)

z[i] = a * x[i] + y[i];

}

The OpenACC programming model is supported by the PGI compiler. The PGI compiler is available on the CSE cluster. To use the PGI compiler, you will need to load the PGI compiler module. The following command will load the PGI compiler module.

module load pgi

Once the PGI compiler module is loaded, you can compile the code using the pgc++ compiler. The following command will compile the code.

pgc++ -acc -Minfo=accel -o gemm gemm.cpp

This command will compile the code using the OpenACC programming model and will print information about the parallelization of the code.

The following is a list of compiler directives that you will need to use to parallelize the gemm function.

The OpenACC kernels directive will parallelize the code within the directive. The OpenACC kernels directive is similar to the OpenMP parallel for directive.

The OpenACC

=======

// gemm OpenACC function
//


// Prompt: gemm CUDA function
Synthesizing 8/10 solutions

=======

// gemm CUDA function

void gemm( const char * transa, const char * transb, const int * m, const int * n, const int * k, const float * alpha, float * a, const int * lda, float * b, const int * ldb, const float * beta, float * c, const int * ldc)

{

// check for errors

assert( transa[0] == 'N' );

assert( transb[0] == 'N' );

assert( *m == *n );

assert( *m == *k );

assert( *lda == *m );

assert( *ldb == *k );

assert( *ldc == *m );

// create a handle for CUBLAS

cublasHandle_t handle;

cublasCreate( & handle);

// perform gemm

cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, *m, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc);

// destroy the handle

cublasDestroy( handle);

}

// Path: GEMM/CXX/main.cpp

// main function

int main( int argc, char * argv[])

{

// size of the matrix

int N = 1000;

// alpha and beta

float alpha = 1.0;

float beta = 0.0;

// allocate memory on the host

float * h_a = new float [N*N];

float * h_b = new float [N*N];

float * h_c = new float [N*N];

// initialize matrices

for ( int i = 0; i < N*N; i++)

{

h_a[i] = 1.0;

h_b[i] = 1.0;

h_c[i] = 0.0;

}

// allocate memory on the device

float * d_a;

float * d_b;

float * d_c;

cudaMalloc( ( void **) & d_a, N*N*sizeof( float ));

cudaMalloc( ( void **) & d_b, N*N*sizeof( float ));

cudaMalloc( ( void **) & d_c, N*N*sizeof( float ));

// copy matrices to the device

cudaMemcpy( d_a, h_a, N*N*sizeof( float ), cudaMemcpyHostToDevice);

cudaMemcpy( d_b, h_b, N*N*sizeof( float ), cudaMemcpyHostToDevice

=======

// gemm CUDA function: C = alpha * A * B + beta * C

// C is M x N, A is M x K, B is K x N

void gemm(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

// Insert code to implement the GEMM function

// on the GPU here

}

// Path: GEMM/CXX/gemm.cu

// gemm CUDA kernel: C = alpha * A * B + beta * C

// C is M x N, A is M x K, B is K x N

__global__ void gemm(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

// Insert CUDA kernel code here

}

// Path: GEMM/CXX/gemm.cu

// gemm CUDA kernel: C = alpha * A * B + beta * C

// C is M x N, A is M x K, B is K x N

__global__ void gemm(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

// Insert CUDA kernel code here

}

// Path: GEMM/CXX/gemm.cu

// gemm CUDA kernel: C = alpha * A * B + beta * C

// C is M x N, A is M x K, B is K x N

__global__ void gemm(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

// Insert CUDA kernel code here

}

// Path: GEMM/CXX/gemm.cu

// gemm CUDA kernel: C = alpha * A * B + beta * C

// C is M x N, A is M x K, B is K x N

__global__ void gemm(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

// Insert CUDA kernel code here

}

// Path: GEMM/CXX/gemm.cu

// gemm CUDA kernel: C = alpha * A * B + beta * C

// C is M x N, A is M x K, B is K x

=======

// gemm CUDA function

template <typename T>

void gemm(const int m, const int n, const int k, const T alpha, const T * A, const T * B, const T beta, T * C) {

// Insert code here

}

// Path: GEMM/CUDA/GEMM.cu

// gemm CUDA kernel

template <typename T>

__global__ void gemm(const int m, const int n, const int k, const T alpha, const T * A, const T * B, const T beta, T * C) {

// Insert code here

}

// Path: GEMM/CUDA/GEMM.cu

// gemm CUDA function

template <typename T>

void gemm(const int m, const int n, const int k, const T alpha, const T * A, const T * B, const T beta, T * C) {

// Insert code here

}

// Path: GEMM/CUDA/GEMM.cu

// gemm CUDA kernel

template <typename T>

__global__ void gemm(const int m, const int n, const int k, const T alpha, const T * A, const T * B, const T beta, T * C) {

// Insert code here

}

// Path: GEMM/CUDA/GEMM.cu

// gemm CUDA function

template <typename T>

void gemm(const int m, const int n, const int k, const T alpha, const T * A, const T * B, const T beta, T * C) {

// Insert code here

}

// Path: GEMM/CUDA/GEMM.cu

// gemm CUDA kernel

template <typename T>

__global__ void gemm(const int m, const int n, const int k, const T alpha, const T * A, const T * B, const T beta, T * C) {

// Insert code here

}

// Path: GEMM/CUDA/GEMM.cu

// gemm CUDA function

template <typename T>

void gemm(const int m, const int n, const int k, const T alpha, const T * A, const T * B, const T beta, T * C) {

// Insert code here

}

// Path: GEMM/CUDA/GEMM.cu

// gemm CUDA kernel

template <typename T>

__

=======

// gemm CUDA function

// CUDA kernel to compute the matrix multiplication C = A * B

// each thread computes one element of the block sub-matrix

// Csub is used to store the element of the block sub-matrix

// that is computed by the thread

// Block sub-matrices are stored in device memory

// to ensure coalesced memory access, the thread dimensions

// are chosen as multiples of the warp size (32)

// Note that this kernel only works if the number of threads

// per block is a multiple of 32

// Block sub-matrices are stored in device memory

// to ensure coalesced memory access, the thread dimensions

// are chosen as multiples of the warp size (32)

// Note that this kernel only works if the number of threads

// per block is a multiple of 32

// Csub is used to store the element of the block sub-matrix

// that is computed by the thread

// each thread computes one element of the block sub-matrix

// Matrix multiplication - Host code

// Matrix dimensions are assumed to be multiples of BLOCK_SIZE

// Matrix dimensions are assumed to be multiples of BLOCK_SIZE

// Allocate host memory for matrices M and N

// Allocate host memory for matrix P

// Allocate device memory for matrices M and N

// Allocate device memory for matrix P

// Initialize host memory

// Initialize host memory

// Copy host memory to device

// Copy host memory to device

// Setup execution parameters

// Setup execution parameters

// Create and start timer

// Create and start timer

// Execute the kernel

// Execute the kernel

// Stop and destroy timer

// Stop and destroy timer

// Read C from device memory

// Read C from device memory

// Free device memory

// Free device memory

// Free host memory

// Free host memory

// Perform CPU matrix multiplication

// Perform CPU matrix multiplication

// Check result (matrix multiplication is not commutative)

// Check result (matrix multiplication is not commutative)

// CUBLAS

// CUBLAS

// Allocate host memory for matrices M and N

// Allocate host memory for matrix P

// Allocate device memory for matrices M and N

// Allocate device memory for matrix P

// Initialize host memory

// Initialize host memory

// Copy host memory to device

// Copy host memory to device

// Setup execution parameters

// Setup execution parameters

// Create and start timer

=======

// gemm CUDA function: C = alpha * A * B + beta * C

// A, B and C are matrices stored in column-major format

// The computation is done on the GPU

// Input: A, B, alpha, beta

// Output: C

// Reference: http://www.netlib.org/blas/sgemm.f

// Reference: http://www.netlib.org/blas/dgemm.f

// Reference: http://www.netlib.org/blas/cgemm.f

// Reference: http://www.netlib.org/blas/zgemm.f

// Reference: http://www.netlib.org/blas/sgemm3m.f

// Reference: http://www.netlib.org/blas/dgemm3m.f

// Reference: http://www.netlib.org/blas/cgemm3m.f

// Reference: http://www.netlib.org/blas/zgemm3m.f

// Reference: http://www.netlib.org/blas/cgemm3m.f

// Reference: http://www.netlib.org/blas/zgemm3m.f

// Reference: http://www.netlib.org/blas/sgemm3m.f

// Reference: http://www.netlib.org/blas/dgemm3m.f

// Reference: http://www.netlib.org/blas/cgemm3m.f

// Reference: http://www.netlib.org/blas/zgemm3m.f

// Reference: http://www.netlib.org/blas/cgemm3m.f

// Reference: http://www.netlib.org/blas/zgemm3m.f

// Reference: http://www.netlib.org/blas/sgemm3m.f

// Reference: http://www.netlib.org/blas/dgemm3m.f

// Reference: http://www.netlib.org/blas/cgemm3m.f

// Reference: http://www.netlib.org/blas/zgemm3m.f

// Reference: http://www.netlib.org/blas/cgemm3m.f

// Reference: http://www.netlib.org/blas/zgemm3m.f

// Reference: http://www.netlib.org/blas/sgemm3m.f

// Reference: http://www.netlib.org/blas/dgemm3m.f

// Reference: http://www.netlib.org/blas/cgemm3m.f

// Reference: http://

=======

// gemm CUDA function

void gemm(float* A, float* B, float* C, int n, int m, int k) {

=======

// gemm CUDA function

//

=======

// gemm CUDA function (matrix multiplication)


// Prompt: gemm OpenMP offload
subroutine

Synthesizing 10/10 solutions

=======

subroutine gemm( m, n, k, alpha, a, lda, b, ldb, beta, c, ldc )
  implicit none
  integer, intent(in) :: m, n, k, lda, ldb, ldc
  real(kind=8), intent(in) :: alpha, beta
  real(kind=8), intent(in) :: a(lda,*), b(ldb,*)
  real(kind=8), intent(inout) :: c(ldc,*)
  integer :: i, j, l
  !$acc kernels
  !$acc loop independent gang vector
  do i = 1, m
    !$acc loop independent gang vector
    do j = 1, n
      c(i,j) = beta * c(i,j)
      !$acc loop independent gang vector
      do l = 1, k
        c(i,j) = c(i,j) + alpha * a(i,l) * b(l,j)
      end do
    end do
  end do
  !$acc end kernels
end subroutine gemm

! gemm OpenMP offload
subroutine gemm( m, n, k, alpha, a, lda, b, ldb, beta, c, ldc )
  implicit none
  integer, intent(in) :: m, n, k, lda, ldb, ldc
  real(kind=8), intent(in) :: alpha, beta
  real(kind=8), intent(in) :: a(lda,*), b(ldb,*)
  real(kind=8), intent(inout) :: c(ldc,*)
  integer :: i, j, l
  !$acc kernels
  !$acc loop independent gang vector
  do i = 1, m
    !$acc loop independent gang vector
    do j = 1, n
      c(i,j) = beta * c(i,j)
      !$acc loop independent gang vector
      do l = 1, k
        c(i,j) = c(i,j) + alpha * a(i,l) * b(l,j)
      end do
    end do
  end do
  !$acc end kernels
end subroutine gemm

! gemm OpenMP offload
subroutine gemm( m, n, k, alpha, a, lda, b, ldb, beta, c, ldc )
  implicit none
  integer, intent(in) :: m, n,

=======

subroutine dgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
  implicit none
  integer, intent(in) :: M, N, K, LDA, LDB, LDC
  real(kind=8), intent(in) :: ALPHA, BETA
  real(kind=8), intent(in) :: A(LDA, *), B(LDB, *)
  real(kind=8), intent(inout) :: C(LDC, *)
  character(len=1), intent(in) :: TRANSA, TRANSB
  integer :: i, j, k
  real(kind=8) :: temp
  !$acc parallel loop copyin(A,B) copy(C)
  do i = 1, M
    do j = 1, N
      temp = 0.0d0
      do k = 1, K
        if (TRANSA == 'N') then
          if (TRANSB == 'N') then
            temp = temp + A(i, k) * B(k, j)
          else
            temp = temp + A(i, k) * B(j, k)
          end if
        else
          if (TRANSB == 'N') then
            temp = temp + A(k, i) * B(k, j)
          else
            temp = temp + A(k, i) * B(j, k)
          end if
        end if
      end do
      C(i, j) = ALPHA * temp + BETA * C(i, j)
    end do
  end do
end subroutine dgemm

! gemm OpenMP offload
subroutine dgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
  implicit none
  integer, intent(in) :: M, N, K, LDA, LDB, LDC
  real(kind=8), intent(in) :: ALPHA, BETA
  real(kind=8), intent(in) :: A(LDA, *), B(LDB, *)
  real(kind=8), intent(inout) :: C(LDC, *)
  character(len=1), intent(in) :: TRANSA, TRANSB
  integer :: i, j,

=======

subroutine gemm_openmp_offload(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
  integer, intent(in) :: m, n, k, lda, ldb, ldc
  real, intent(in) :: alpha, beta
  real, intent(in) :: A(lda, k), B(ldb, n)
  real, intent(inout) :: C(ldc, n)
  integer :: i, j, l
  !$omp target teams distribute parallel for collapse(2) private(i,j,l) map(to:A,B) map(tofrom:C)
  do j = 1, n
    do i = 1, m
      C(i, j) = beta * C(i, j)
      do l = 1, k
        C(i, j) = C(i, j) + alpha * A(i, l) * B(l, j)
      end do
    end do
  end do
end subroutine gemm_openmp_offload

! gemm OpenMP offload with SIMD
subroutine gemm_openmp_offload_simd(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
  integer, intent(in) :: m, n, k, lda, ldb, ldc
  real, intent(in) :: alpha, beta
  real, intent(in) :: A(lda, k), B(ldb, n)
  real, intent(inout) :: C(ldc, n)
  integer :: i, j, l
  !$omp target teams distribute parallel for collapse(2) private(i,j,l) map(to:A,B) map(tofrom:C)
  do j = 1, n
    do i = 1, m
      C(i, j) = beta * C(i, j)
      do l = 1, k
        C(i, j) = C(i, j) + alpha * A(i, l) * B(l, j)
      end do
    end do
  end do
end subroutine gemm_openmp_offload_simd

! gemm OpenMP offload with SIMD and unrolling
subroutine gemm_openmp_offload_simd_unroll(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
  integer, intent(in) ::

=======

subroutine gemm(a, b, c, n)
  integer, intent(in) :: n
  real, intent(in) :: a(n,n), b(n,n)
  real, intent(inout) :: c(n,n)

  integer :: i, j, k
  real :: temp

  !$omp target teams distribute parallel do collapse(2) private(i,j,k,temp)
  do i = 1, n
    do j = 1, n
      temp = 0.0
      do k = 1, n
        temp = temp + a(i,k) * b(k,j)
      end do
      c(i,j) = temp
    end do
  end do
end subroutine gemm

! gemm OpenMP offload
subroutine gemm(a, b, c, n)
  integer, intent(in) :: n
  real, intent(in) :: a(n,n), b(n,n)
  real, intent(inout) :: c(n,n)

  integer :: i, j, k
  real :: temp

  !$omp target teams distribute parallel do collapse(2) private(i,j,k,temp)
  do i = 1, n
    do j = 1, n
      temp = 0.0
      do k = 1, n
        temp = temp + a(i,k) * b(k,j)
      end do
      c(i,j) = temp
    end do
  end do
end subroutine gemm

! gemm OpenMP offload
subroutine gemm(a, b, c, n)
  integer, intent(in) :: n
  real, intent(in) :: a(n,n), b(n,n)
  real, intent(inout) :: c(n,n)

  integer :: i, j, k
  real :: temp

  !$omp target teams distribute parallel do collapse(2) private(i,j,k,temp)
  do i = 1, n
    do j = 1, n
      temp = 0.0
      do k = 1, n
        temp = temp + a(i,k) * b(k,j)
      end do
      c(i,j) = temp
    end do
  end do
end subroutine gemm

! gemm OpenMP offload
subroutine gemm(a, b, c, n)
  integer, intent

=======

subroutine sgemm( m, n, k, alpha, A, lda, B, ldb, beta, C, ldc )
  integer, intent(in) :: m, n, k, lda, ldb, ldc
  real(4), intent(in) :: alpha, beta
  real(4), intent(in) :: A(lda,k), B(ldb,n)
  real(4), intent(inout) :: C(ldc,n)
  integer :: i, j, l
  !$OMP PARALLEL DO PRIVATE(i,j,l) SCHEDULE(static)
  DO j = 1, n
    DO i = 1, m
      C(i,j) = beta*C(i,j)
      DO l = 1, k
        C(i,j) = C(i,j) + alpha*A(i,l)*B(l,j)
      ENDDO
    ENDDO
  ENDDO
  !$OMP END PARALLEL DO
end subroutine sgemm

! gemm OpenMP offload with OpenACC
subroutine sgemm( m, n, k, alpha, A, lda, B, ldb, beta, C, ldc )
  integer, intent(in) :: m, n, k, lda, ldb, ldc
  real(4), intent(in) :: alpha, beta
  real(4), intent(in) :: A(lda,k), B(ldb,n)
  real(4), intent(inout) :: C(ldc,n)
  integer :: i, j, l
  !$OMP PARALLEL DO PRIVATE(i,j,l) SCHEDULE(static)
  !$ACC KERNELS
  DO j = 1, n
    DO i = 1, m
      C(i,j) = beta*C(i,j)
      DO l = 1, k
        C(i,j) = C(i,j) + alpha*A(i,l)*B(l,j)
      ENDDO
    ENDDO
  ENDDO
  !$OMP END PARALLEL DO
  !$ACC END KERNELS
end subroutine sgemm

! gemm OpenMP offload with OpenACC and OpenMP parallel
subroutine sgemm( m, n, k, alpha, A, lda, B, ldb, beta, C, ldc )
  integer, intent(in) :: m, n, k, lda, ldb, ldc
  real(4), intent(in) :: alpha,

=======

subroutine gemm (a, b, c, m, n, k)
  real, intent(in) :: a(m, k), b(k, n)
  real, intent(out) :: c(m, n)
  integer :: i, j, l
  !$acc enter data copyin(a, b)
  !$acc kernels present(a, b, c)
  !$acc loop gang
  do i = 1, m
    !$acc loop worker
    do j = 1, n
      c(i, j) = 0.0
      !$acc loop vector
      do l = 1, k
        c(i, j) = c(i, j) + a(i, l) * b(l, j)
      enddo
    enddo
  enddo
  !$acc exit data delete(a, b)
end subroutine gemm
! gemm OpenMP offload

! gemm OpenMP offload with OpenACC kernels
subroutine gemm (a, b, c, m, n, k)
  real, intent(in) :: a(m, k), b(k, n)
  real, intent(out) :: c(m, n)
  integer :: i, j, l
  !$omp target enter data map(to: a, b)
  !$omp target teams distribute parallel do collapse(2)
  do i = 1, m
    do j = 1, n
      c(i, j) = 0.0
      do l = 1, k
        c(i, j) = c(i, j) + a(i, l) * b(l, j)
      enddo
    enddo
  enddo
  !$omp target exit data map(delete: a, b)
end subroutine gemm
! gemm OpenMP offload with OpenACC kernels

! gemm OpenMP offload with OpenACC kernels and data
subroutine gemm (a, b, c, m, n, k)
  real, intent(in) :: a(m, k), b(k, n)
  real, intent(out) :: c(m, n)
  integer :: i, j, l
  !$acc enter data copyin(a, b)
  !$acc kernels present(a, b, c)
  !$acc loop gang
  do i = 1, m
    !$acc loop worker

=======

subroutine gemm_offload(a,b,c,m,n,k)
  implicit none
  real, dimension(:,:), intent(in) :: a,b
  real, dimension(:,:), intent(out) :: c
  integer, intent(in) :: m,n,k

  integer :: i,j,l
  !$acc kernels copyin(a,b) copyout(c)
  !$acc loop gang
  do i=1,m
    !$acc loop worker
    do j=1,n
      c(i,j) = 0
      !$acc loop vector
      do l=1,k
        c(i,j) = c(i,j) + a(i,l)*b(l,j)
      end do
    end do
  end do
  !$acc end kernels
end subroutine gemm_offload

! gemm OpenMP offload
subroutine gemm_offload(a,b,c,m,n,k)
  implicit none
  real, dimension(:,:), intent(in) :: a,b
  real, dimension(:,:), intent(out) :: c
  integer, intent(in) :: m,n,k

  integer :: i,j,l
  !$acc kernels copyin(a,b) copyout(c)
  !$acc loop gang
  do i=1,m
    !$acc loop worker
    do j=1,n
      c(i,j) = 0
      !$acc loop vector
      do l=1,k
        c(i,j) = c(i,j) + a(i,l)*b(l,j)
      end do
    end do
  end do
  !$acc end kernels
end subroutine gemm_offload

! gemm OpenMP offload
subroutine gemm_offload(a,b,c,m,n,k)
  implicit none
  real, dimension(:,:), intent(in) :: a,b
  real, dimension(:,:), intent(out) :: c
  integer, intent(in) :: m,n,k

  integer :: i,j,l
  !$acc kernels copyin(a,b) copyout(c)
  !$acc loop gang
  do i=1,m
    !$acc loop worker
    do j=1,n
      c(i,j) = 0
      !$acc loop vector
      do l=1,k
        c(i,j) = c(i,j) + a(i,l)*b(l,j)
      end do
    end do
  end do
  !$acc end

=======

subroutine gemm_openmp_offload(A, B, C, alpha, beta, N)
  real, intent(in) :: A(0:N-1,0:N-1), B(0:N-1,0:N-1), alpha, beta
  real, intent(inout) :: C(0:N-1,0:N-1)

  !$acc data copyin(A, B) copy(C)
  !$acc kernels
  !$acc loop gang
  do i = 0, N-1
    !$acc loop worker
    do j = 0, N-1
      !$acc loop vector
      do k = 0, N-1
        C(i,j) = C(i,j) + alpha * A(i,k) * B(k,j)
      end do
    end do
  end do
  !$acc end kernels
  !$acc end data

end subroutine gemm_openmp_offload

! gemm OpenMP offload with temp
subroutine gemm_openmp_offload_temp(A, B, C, alpha, beta, N)
  real, intent(in) :: A(0:N-1,0:N-1), B(0:N-1,0:N-1), alpha, beta
  real, intent(inout) :: C(0:N-1,0:N-1)
  real, allocatable :: Ctemp(:,:)

  allocate(Ctemp(N,N))

  !$acc data copyin(A, B) copy(Ctemp)
  !$acc kernels
  !$acc loop gang
  do i = 0, N-1
    !$acc loop worker
    do j = 0, N-1
      !$acc loop vector
      do k = 0, N-1
        Ctemp(i,j) = Ctemp(i,j) + alpha * A(i,k) * B(k,j)
      end do
    end do
  end do
  !$acc end kernels
  !$acc end data

  C = beta * C + Ctemp

  deallocate(Ctemp)

end subroutine gemm_openmp_offload_temp

! gemm OpenMP offload with temp and prefetch
subroutine gemm_openmp_offload_temp_prefetch(A, B, C, alpha, beta, N)
  real, intent(in) :: A(0:N-1,0:N-1), B

=======

subroutine gemm_openmp_offload(A, B, C, alpha, beta, n, m, k)
  implicit none

  real, intent(in) :: alpha
  real, intent(in) :: beta
  real, dimension(:,:), intent(in) :: A
  real, dimension(:,:), intent(in) :: B
  real, dimension(:,:), intent(inout) :: C
  integer, intent(in) :: n
  integer, intent(in) :: m
  integer, intent(in) :: k

  integer :: i
  integer :: j
  integer :: l

  !$omp target teams distribute parallel for collapse(2) map(to:A,B) map(tofrom:C)
  do i = 1, n
    do j = 1, m
      C(i,j) = beta * C(i,j)
      do l = 1, k
        C(i,j) = C(i,j) + alpha * A(i,l) * B(l,j)
      end do
    end do
  end do

end subroutine gemm_openmp_offload

! gemm OpenMP offload
subroutine gemm_openmp_offload(A, B, C, alpha, beta, n, m, k)
  implicit none

  real, intent(in) :: alpha
  real, intent(in) :: beta
  real, dimension(:,:), intent(in) :: A
  real, dimension(:,:), intent(in) :: B
  real, dimension(:,:), intent(inout) :: C
  integer, intent(in) :: n
  integer, intent(in) :: m
  integer, intent(in) :: k

  integer :: i
  integer :: j
  integer :: l

  !$omp target teams distribute parallel for collapse(2) map(to:A,B) map(tofrom:C)
  do i = 1, n
    do j = 1, m
      C(i,j) = beta * C(i,j)
      do l = 1, k
        C(i,j) = C(i,j) + alpha * A(i,l) * B(l,j)
      end do
    end do
  end do

end subroutine gemm_openmp_offload

! gemm OpenACC
subroutine gemm_openacc(A, B, C, alpha, beta, n, m, k)
  implicit none

  real, intent

=======

subroutine gemm_openmp_offload (A, B, C, alpha, beta, n, nthreads)
  implicit none
  integer, intent(in) :: n, nthreads
  real (kind=8), intent(in) :: alpha, beta
  real (kind=8), intent(in), dimension(n,n) :: A, B
  real (kind=8), intent(inout), dimension(n,n) :: C
  !$omp target teams distribute parallel for num_threads(nthreads)
  do j = 1, n
    do i = 1, n
      do k = 1, n
        C(i,j) = C(i,j) + alpha * A(i,k) * B(k,j)
      end do
    end do
  end do
end subroutine gemm_openmp_offload

The OpenMP offload pragma in this case is !$omp target teams distribute parallel for num_threads(nthreads) . The target directive specifies that the following code is to be offloaded to the GPU. The teams distribute directive specifies that the iterations of the loop are to be distributed across the available threads. The parallel for directive specifies that the iterations of the loop are to be parallelized among the threads of the team. The num_threads directive specifies the number of threads to use. The nthreads argument is passed to the subroutine and is used to set the number of threads. The parallel for directive is used in this case because the outer loop is parallelized. The parallel for directive can be used to parallelize any loop with a static schedule.

The OpenMP offload pragma is used to offload the code to the GPU. The pragma is used to specify the target device, the distribution of the iterations of the loop across the threads, and the parallelization of the loop among the threads. The OpenMP offload pragma is a new feature of OpenMP 4.5.

The GPU is not used in this example. The outer loop is parallelized and the inner loops are not parallelized. The pragma !$omp target teams distribute parallel for num_threads(nthreads) is used to offload the code to the GPU. The pragma specifies that the code is to be offloaded to the GPU, that the outer loop is to be distributed across the threads, and that the outer loop is to be parallelized among the threads. The pragma also specifies the number of threads to use.

The GPU is not used in this example


! Prompt: gemm OpenACC
Synthesizing 9/10 solutions

=======

! gemm OpenACC

! gemm OpenMP

! gemm OpenMP2

! gemm OpenMP3

! gemm OpenMP4

! gemm OpenMP5

! gemm OpenMP6

! gemm OpenMP7

! gemm OpenMP8

! gemm OpenMP9

! gemm OpenMP10

! gemm OpenMP11

! gemm OpenMP12

! gemm OpenMP13

! gemm OpenMP14

! gemm OpenMP15

! gemm OpenMP16

! gemm OpenMP17

! gemm OpenMP18

! gemm OpenMP19

! gemm OpenMP20

! gemm OpenMP21

! gemm OpenMP22

! gemm OpenMP23

! gemm OpenMP24

! gemm OpenMP25

! gemm OpenMP26

! gemm OpenMP27

! gemm OpenMP28

! gemm OpenMP29

! gemm OpenMP30

! gemm OpenMP31

! gemm OpenMP32

! gemm OpenMP33

! gemm OpenMP34

! gemm OpenMP35

! gemm OpenMP36

! gemm OpenMP37

! gemm OpenMP38

! gemm OpenMP39

! gemm OpenMP40

! gemm OpenMP41

! gemm OpenMP42

! gemm OpenMP43

! gemm OpenMP44

! gemm OpenMP45

! gemm OpenMP46

! gemm OpenMP47

! gemm OpenMP48

! gemm OpenMP49

! gemm OpenMP50

! gemm OpenMP51

! gemm OpenMP52

! gemm OpenMP53

! gemm OpenMP54

! gemm OpenMP55

! gemm OpenMP56

! gemm OpenMP57

! gemm OpenMP58

! gemm OpenMP59

! gemm OpenMP60

! gemm OpenMP61

! gemm OpenMP62

! gemm OpenMP63

! gemm OpenMP64

! gemm OpenMP65

! gemm OpenMP66

! gemm OpenMP67

! gemm OpenMP68

! gemm OpenMP69

! gemm OpenMP70

! gemm OpenMP71

! gem

=======

! gemm OpenACC
! ./gemm_openacc

! gemm OpenCL
! ./gemm_opencl

! gemm CUDA
! ./gemm_cuda

! gemm HIP
! ./gemm_hip

! gemm SYCL
! ./gemm_sycl

! gemm SYCL with DPC++
! ./gemm_sycl_dpcpp

! gemm SYCL with Intel DPC++
! ./gemm_sycl_intel

! gemm SYCL with AMD DPC++
! ./gemm_sycl_amd

! gemm SYCL with Intel OpenCL
! ./gemm_sycl_intel_opencl

! gemm SYCL with AMD OpenCL
! ./gemm_sycl_amd_opencl

! gemm SYCL with CUDA
! ./gemm_sycl_cuda

! gemm SYCL with HIP
! ./gemm_sycl_hip

! gemm SYCL with OpenACC
! ./gemm_sycl_openacc

! gemm SYCL with OpenMP
! ./gemm_sycl_openmp

! gemm SYCL with TBB
! ./gemm_sycl_tbb

! gemm SYCL with Level Zero
! ./gemm_sycl_level_zero

! gemm SYCL with ROCm
! ./gemm_sycl_rocm

! gemm SYCL with CUDA and OpenMP
! ./gemm_sycl_cuda_openmp

! gemm SYCL with CUDA and TBB
! ./gemm_sycl_cuda_tbb

! gemm SYCL with HIP and OpenMP
! ./gemm_sycl_hip_openmp

! gemm SYCL with HIP and TBB
! ./gemm_sycl_hip_tbb

! gemm SYCL with Level Zero and OpenMP
! ./gemm_sycl_level_zero_openmp

! gemm SYCL with Level Zero and TBB
! ./gemm_sycl_level_zero_tbb

! gemm SYCL with ROCm and OpenMP
! ./gemm_sycl_rocm_openmp

! gemm SYCL with ROCm and TBB
! ./gemm_sycl_rocm_tbb

! gemm SYCL with CUDA and OpenCL
! ./gemm_sycl_cuda_opencl

! gemm SYCL with HIP and OpenCL
! ./gemm_sycl_hip_opencl

=======

! gemm OpenACC

! gemm OpenMP

! gemm OpenCL

! gemm CUDA

! gemm MPI

! gemm MPI+OpenMP

! gemm MPI+OpenCL

! gemm MPI+CUDA

! gemm MPI+CUDA+OpenMP

! gemm MPI+CUDA+OpenCL

! gemm MPI+OpenACC

! gemm MPI+OpenACC+OpenMP

! gemm MPI+OpenACC+OpenCL

! gemm MPI+CUDA+OpenACC

! gemm MPI+CUDA+OpenACC+OpenMP

! gemm MPI+CUDA+OpenACC+OpenCL

! gemm MPI+OpenCL+OpenACC

! gemm MPI+OpenCL+OpenACC+OpenMP

! gemm MPI+OpenCL+OpenACC+OpenMP

! gemm MPI+CUDA+Ope

=======

! gemm OpenACC
! gemm OpenMP
! gemm OpenMP SIMD
! gemm OpenMP SIMD SIMD
! gemm OpenMP SIMD SIMD SIMD
! gemm OpenMP SIMD SIMD SIMD SIMD
! gemm OpenMP SIMD SIMD SIMD SIMD SIMD
! gemm OpenMP SIMD SIMD SIMD SIMD SIMD SIMD
! gemm OpenMP SIMD SIMD SIMD SIMD SIMD SIMD SIMD
! gemm OpenMP SIMD SIMD SIMD SIMD SIMD SIMD SIMD SIMD
! gemm OpenMP SIMD SIMD SIMD SIMD SIMD SIMD SIMD SIMD SIMD
! gemm OpenMP SIMD SIMD SIMD SIMD SIM

=======

! gemm OpenACC

! gemm OpenCL

! gemm CUDA

! gemm CUBLAS

! gemm CUBLAS_TENSOR_OP

! gemm CUBLAS_TENSOR_OP_HALF

! gemm CUBLAS_TENSOR_OP_INT8

! gemm CUBLAS_TENSOR_OP_INT8x4

! gemm CUBLAS_TENSOR_OP_INT8x32

! gemm CUBLAS_TENSOR_OP_INT8x128

! gemm CUBLAS_TENSOR_OP_INT8x256

! gemm CUBLAS_TENSOR_OP_INT8x512

! gemm CUBLAS_TENSOR_OP_INT8x1024

! gemm CUBLAS_TENSOR_OP_INT8x2048

! gemm CUBLAS_TENSOR_OP_INT8x4096

! gemm CUBLAS_TENSOR_OP_INT8x8192

! gemm CUBLAS_TENSOR_OP_INT8x16384

! gemm CUBLAS_TENSOR_OP_INT8x32768

! gemm CUBLAS_TENSOR_OP_INT8x65536

! gemm CUBLAS_TENSOR_OP_INT8x131072

! gemm CUBLAS_TENSOR_OP_INT8x262144

! gemm CUBLAS_TENSOR_OP_INT8x524288

! gemm CUBLAS_TENSOR_OP_INT8x1048576

! gemm CUBLAS_TENSOR_OP_INT8x2097152

! gemm CUBLAS_TENSOR_OP_INT8x4194304

! gemm CUBLAS_TENSOR_OP_INT8x8388608

! gemm CUBLAS_TENSOR_OP_INT8x16777216

! gemm CUBLAS_TENSOR_OP_INT8x33554432

! gemm CUBLAS_TENSOR_OP_INT8x67108864

! gemm CUBLAS_TENSOR_OP_INT8x134217728

! gemm CUBLAS_TENSOR_OP_INT8x268435456

! gemm CUBLAS_TENSOR_OP_INT8x536870912

! gemm CUBLAS_TENSOR_OP_INT8x1073741824

! gemm CUBLAS_TENSOR_OP_INT8x2147483648

! gemm CUBLAS_TENSOR_OP_INT8x4294967296

! gemm CUBLAS_TENSOR_OP_INT8x8589934592

=======

! gemm OpenACC

! gemm OpenMP

! gemm OpenMP SIMD

! gemm OpenMP SIMD SIMD

! gemm OpenMP SIMD SIMD SIMD

! gemm CUDA

! gemm CUDA CUDA

! gemm CUDA CUDA CUDA

! gemm CUDA CUDA CUDA CUDA

! gemm CUDA CUDA CUDA CUDA CUDA

! gemm CUDA CUDA CUDA CUDA CUDA CUDA

! gemm CUDA CUDA CUDA CUDA CUDA CUDA CUDA

! gemm CUDA CUDA CUDA CUDA CUDA CUDA CUDA CUDA

! gemm CUDA CUDA CUDA CUDA CUDA CUDA CUDA CUDA CUDA

! gemm CUDA CUDA CUDA CUDA C

=======

! gemm OpenACC

! gemm OpenACC -DUSE_DOUBLE

! gemm OpenACC -DUSE_DOUBLE -DUSE_CUBLAS

! gemm OpenACC -DUSE_DOUBLE -DUSE_CUBLAS -DUSE_CUBLAS_TUNE

! gemm OpenACC -DUSE_DOUBLE -DUSE_CUBLAS -DUSE_CUBLAS_TUNE -DUSE_CUBLAS_V2

! gemm OpenACC -DUSE_DOUBLE -DUSE_CUBLAS -DUSE_CUBLAS_TUNE -DUSE_CUBLAS_V2 -DUSE_CUBLAS_V2_TUNE

! gemm OpenACC -DUSE_DOUBLE -DUSE_CUBLAS -DUSE_CUBLAS_TUNE -DUSE_CUBLAS_V2 -DUSE_CUBLAS_V2_TUNE -DUSE_CUBLAS_V2_TUNE2

! gemm OpenACC -DUSE_DOUBLE -DUSE_CUBLAS -DUSE_CUBLAS_TUNE -DUSE_CUBLAS_V2 -DUSE_CUBLAS_V2_TUNE -DUSE_CUBLAS_V2_TUNE2 -DUSE_CUBLAS_V2_TUNE3

! gemm OpenACC -DUSE_DOUBLE -DUSE_CUBLAS -DUSE_CUBLAS_TUNE -DUSE_CUBLAS_V2 -DUSE_CUBLAS_V2_TUNE -DUSE_CUBLAS_V2_TUNE2 -DUSE_CUBLAS_V2_TUNE3 -DUSE_CUBLAS_V2_TUNE4

! gemm OpenACC -DUSE_DOUBLE -DUSE_CUBLAS -DUSE_CUBLAS_TUNE -DUSE_CUBLAS_V2 -DUSE_CUBLAS_V2_TUNE -DUSE_CUBLAS_V2_TUNE2 -DUSE_CUBLAS_V2_TUNE3 -DUSE_CUBLAS_V2_TUNE4 -DUSE_CUBLAS_V2_TUNE5

! gemm OpenACC -DUSE_DOUBLE -DUSE_CUBLAS -DUSE_CUBLAS_TUNE -DUSE_CUBLAS_V2 -DUSE_CUBLAS_V2_TUNE -DUSE_CUBLAS_V2_TUNE2 -DUSE_CUBLAS_V2_TUNE3 -DUSE_CUBLAS_V2_TUNE4 -DUSE_CUBLAS_V2_TUNE5 -DUSE_CUBLAS_V2_TUNE6

=======

! gemm OpenACC

! gemm OpenMP

! gemm OpenMP offload

! gemm OpenMP target

! gemm OpenMP target offload

! gemm OpenMP target offload map

! gemm OpenMP target offload map depend

! gemm OpenMP target offload map depend to

! gemm OpenMP target offload map depend to from

! gemm OpenMP target offload map depend to from alloc

! gemm OpenMP target offload map depend to from alloc delete

! gemm OpenMP target offload map depend to from alloc delete to

! gemm OpenMP target offload map depend to from alloc delete to from

! gemm OpenMP target offload map depend to from alloc delete to from update

! gemm OpenMP target offload map depend to from alloc delete to from update from

! gemm OpenMP target offload map depend to from alloc delete to from update from to

! gemm OpenMP target offload map depend to from alloc delete to from update from to from

! gemm OpenMP target offload map depend to from alloc delete to from update from to from update

! gemm OpenMP target offload map depend to from alloc delete to from update from to from update to

! gemm OpenMP target offload map depend to from alloc delete to from update from to from update to from

! gemm OpenMP target offload map depend to from alloc delete to from update from to from update to from update

! gemm OpenMP target offload map depend to from alloc delete to from update from to from update to from update to

! gemm OpenMP target offload map depend to from alloc delete to from update from to from update to from update to from

! gemm OpenMP target offload map depend to from alloc delete to from update from to from update to from update to from update

! gemm OpenMP target offload map depend to from alloc delete to from update from to from update to from update to from update to

! gemm OpenMP target offload map depend to from alloc delete to from update from to from update to from update to from update to from

! gemm OpenMP target offload map depend to from alloc delete to from update from to from update to from update to from update to from update

! gemm OpenMP target offload map depend to from alloc delete to from update

=======

! gemm OpenACC
! ge

! Prompt: gemm OpenACC\nsubroutine
Synthesizing 10/10 solutions

=======

subroutine gemm(alpha, A, B, beta, C)
  real, intent(in) :: alpha
  real, dimension(:,:), intent(in) :: A, B
  real, intent(in) :: beta
  real, dimension(:,:), intent(inout) :: C
  integer :: i, j, k
  !$acc kernels
  !$acc loop independent
  do i = 1, size(C, 1)
    !$acc loop independent
    do j = 1, size(C, 2)
      C(i,j) = beta * C(i,j)
      !$acc loop independent
      do k = 1, size(A, 2)
        C(i,j) = C(i,j) + alpha * A(i,k) * B(k,j)
      end do
    end do
  end do
  !$acc end kernels
end subroutine gemm
! gemm OpenACC

! gemm OpenMP
subroutine gemm(alpha, A, B, beta, C)
  real, intent(in) :: alpha
  real, dimension(:,:), intent(in) :: A, B
  real, intent(in) :: beta
  real, dimension(:,:), intent(inout) :: C
  integer :: i, j, k
  !$omp parallel do private(i,j,k) shared(A,B,C,alpha,beta)
  do i = 1, size(C, 1)
    do j = 1, size(C, 2)
      C(i,j) = beta * C(i,j)
      do k = 1, size(A, 2)
        C(i,j) = C(i,j) + alpha * A(i,k) * B(k,j)
      end do
    end do
  end do
end subroutine gemm
! gemm OpenMP

! gemm MPI
subroutine gemm(alpha, A, B, beta, C)
  real, intent(in) :: alpha
  real, dimension(:,:), intent(in) :: A, B
  real, intent(in) :: beta
  real, dimension(:,:), intent(inout) :: C
  integer :: i, j, k
  do i = 1, size(C, 1)
    do j = 1, size(C, 2)
      C(i,j) = beta * C(i,j)
      do k = 1, size(A,

=======

subroutine gemm( m, n, k, alpha, a, lda, b, ldb, beta, c, ldc )
  integer, intent(in) :: m, n, k, lda, ldb, ldc
  real, intent(in) :: alpha, beta
  real, intent(in) :: a(lda,k), b(ldb,n)
  real, intent(inout) :: c(ldc,n)
  integer i, j, l
  !$acc parallel loop collapse(2) present(a,b,c)
  do i=1,m
    do j=1,n
      c(i,j) = beta*c(i,j)
      do l=1,k
        c(i,j) = c(i,j) + alpha*a(i,l)*b(l,j)
      end do
    end do
  end do
end subroutine gemm

! gemm OpenMP
subroutine gemm( m, n, k, alpha, a, lda, b, ldb, beta, c, ldc )
  integer, intent(in) :: m, n, k, lda, ldb, ldc
  real, intent(in) :: alpha, beta
  real, intent(in) :: a(lda,k), b(ldb,n)
  real, intent(inout) :: c(ldc,n)
  integer i, j, l
  !$omp parallel do default(none) private(i,j,l) shared(m,n,k,alpha,a,lda,b,ldb,beta,c,ldc)
  do i=1,m
    do j=1,n
      c(i,j) = beta*c(i,j)
      do l=1,k
        c(i,j) = c(i,j) + alpha*a(i,l)*b(l,j)
      end do
    end do
  end do
end subroutine gemm

! gemm MPI
subroutine gemm( m, n, k, alpha, a, lda, b, ldb, beta, c, ldc )
  integer, intent(in) :: m, n, k, lda, ldb, ldc
  real, intent(in) :: alpha, beta
  real, intent(in) :: a(lda,k), b(ldb,n)
  real, intent(inout) :: c(ldc,n)
  integer i, j, l
  integer ierr
  integer myid, numprocs
  integer n1, n2, n3

=======

subroutine gemm(a, b, c, n)
  integer, intent(in) :: n
  real, dimension(n,n), intent(in) :: a, b
  real, dimension(n,n), intent(inout) :: c
  integer :: i, j, k
  !$acc kernels
  do i = 1, n
    do j = 1, n
      do k = 1, n
        c(i,j) = c(i,j) + a(i,k) * b(k,j)
      end do
    end do
  end do
  !$acc end kernels
end subroutine gemm

! gemm OpenMP
subroutine gemm(a, b, c, n)
  integer, intent(in) :: n
  real, dimension(n,n), intent(in) :: a, b
  real, dimension(n,n), intent(inout) :: c
  integer :: i, j, k
  !$omp parallel do private(i,j,k)
  do i = 1, n
    do j = 1, n
      do k = 1, n
        c(i,j) = c(i,j) + a(i,k) * b(k,j)
      end do
    end do
  end do
  !$omp end parallel do
end subroutine gemm

! gemm OpenMP target
subroutine gemm(a, b, c, n)
  integer, intent(in) :: n
  real, dimension(n,n), intent(in) :: a, b
  real, dimension(n,n), intent(inout) :: c
  integer :: i, j, k
  !$omp target teams distribute parallel for collapse(3)
  do i = 1, n
    do j = 1, n
      do k = 1, n
        c(i,j) = c(i,j) + a(i,k) * b(k,j)
      end do
    end do
  end do
  !$omp end target teams distribute parallel for collapse(3)
end subroutine gemm

! gemm OpenMP target with data
subroutine gemm(a, b, c, n)
  integer, intent(in) :: n
  real, dimension(n,n), intent(in) :: a, b
  real, dimension(n,n), intent(inout) :: c
  integer :: i, j, k

=======

subroutine gemm(A, B, C, M, N, K)
  integer, intent(in) :: M, N, K
  real, intent(in) :: A(M,K), B(K,N)
  real, intent(inout) :: C(M,N)
  !$acc kernels
  !$acc loop gang
  do j = 1, N
    !$acc loop vector
    do i = 1, M
      real :: sum
      sum = 0
      !$acc loop seq
      do k = 1, K
        sum = sum + A(i,k) * B(k,j)
      end do
      C(i,j) = sum
    end do
  end do
  !$acc end kernels
end subroutine gemm

! gemm OpenMP
subroutine gemm(A, B, C, M, N, K)
  integer, intent(in) :: M, N, K
  real, intent(in) :: A(M,K), B(K,N)
  real, intent(inout) :: C(M,N)
  !$omp parallel
  !$omp do schedule(static, 1)
  do j = 1, N
    !$omp do schedule(static, 1)
    do i = 1, M
      real :: sum
      sum = 0
      !$omp do schedule(static, 1)
      do k = 1, K
        sum = sum + A(i,k) * B(k,j)
      end do
      C(i,j) = sum
    end do
  end do
  !$omp end parallel
end subroutine gemm

! gemm OpenMP + SIMD
subroutine gemm(A, B, C, M, N, K)
  integer, intent(in) :: M, N, K
  real, intent(in) :: A(M,K), B(K,N)
  real, intent(inout) :: C(M,N)
  !$omp parallel
  !$omp do schedule(static, 1)
  do j = 1, N
    !$omp do schedule(static, 1)
    do i = 1, M
      real :: sum
      sum = 0
      !$omp simd
      do k = 1, K
        sum = sum + A(i,k) * B(k,j)
      end do
      C(i,j) = sum
    end

=======

subroutine gemm( a, b, c, n )
  implicit none
  integer, intent(in) :: n
  real, dimension(n,n), intent(in) :: a, b
  real, dimension(n,n), intent(out) :: c
  integer :: i, j, k
  !$acc parallel loop collapse(2) present(a,b,c)
  do i=1,n
    do j=1,n
      c(i,j) = 0.0
      do k=1,n
        c(i,j) = c(i,j) + a(i,k) * b(k,j)
      enddo
    enddo
  enddo
end subroutine gemm

! gemm OpenMP
subroutine gemm( a, b, c, n )
  implicit none
  integer, intent(in) :: n
  real, dimension(n,n), intent(in) :: a, b
  real, dimension(n,n), intent(out) :: c
  integer :: i, j, k
  !$omp parallel do default(none) shared(a,b,c,n)
  do i=1,n
    do j=1,n
      c(i,j) = 0.0
      do k=1,n
        c(i,j) = c(i,j) + a(i,k) * b(k,j)
      enddo
    enddo
  enddo
end subroutine gemm

! gemm MPI
subroutine gemm( a, b, c, n )
  implicit none
  integer, intent(in) :: n
  real, dimension(n,n), intent(in) :: a, b
  real, dimension(n,n), intent(out) :: c
  integer :: i, j, k
  do i=1,n
    do j=1,n
      c(i,j) = 0.0
      do k=1,n
        c(i,j) = c(i,j) + a(i,k) * b(k,j)
      enddo
    enddo
  enddo
end subroutine gemm

! gemm MPI+OpenMP
subroutine gemm( a, b, c, n )
  implicit none
  integer, intent(in) :: n
  real, dimension(n,n), intent(in) :: a, b
  real, dimension(n,n), intent(out) :: c
  integer :: i

=======

subroutine gemm_openacc(N, alpha, A, B, beta, C)
  implicit none
  integer, intent(in) :: N
  real, intent(in) :: alpha
  real, dimension(N,N), intent(in) :: A, B
  real, intent(in) :: beta
  real, dimension(N,N), intent(inout) :: C
  !$acc parallel loop collapse(2) present(A,B,C)
  do i=1,N
    do j=1,N
      C(i,j) = beta*C(i,j)
      do k=1,N
        C(i,j) = C(i,j) + alpha*A(i,k)*B(k,j)
      end do
    end do
  end do
end subroutine gemm_openacc

! gemm OpenMP
subroutine gemm_openmp(N, alpha, A, B, beta, C)
  implicit none
  integer, intent(in) :: N
  real, intent(in) :: alpha
  real, dimension(N,N), intent(in) :: A, B
  real, intent(in) :: beta
  real, dimension(N,N), intent(inout) :: C
  !$omp parallel do collapse(2) shared(A,B,C)
  do i=1,N
    do j=1,N
      C(i,j) = beta*C(i,j)
      do k=1,N
        C(i,j) = C(i,j) + alpha*A(i,k)*B(k,j)
      end do
    end do
  end do
end subroutine gemm_openmp

! gemm MPI
subroutine gemm_mpi(N, alpha, A, B, beta, C)
  implicit none
  integer, intent(in) :: N
  real, intent(in) :: alpha
  real, dimension(N,N), intent(in) :: A, B
  real, intent(in) :: beta
  real, dimension(N,N), intent(inout) :: C
  integer :: i, j, k
  integer :: ierr, myid, numprocs, mysize, myoffset
  real, dimension(N,N) :: Ctemp
  call MPI_Comm_size(MPI_COMM_WORLD, numprocs, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, myid, ierr)
  mysize = N/numprocs
  myoffset = mysize*myid
  if (my

=======

subroutine gemm( A, B, C )
    real, dimension(:,:), intent(in) :: A
    real, dimension(:,:), intent(in) :: B
    real, dimension(:,:), intent(inout) :: C
    integer :: M, N, K, i, j, k
    real :: tmp
    M = size( A, 1 )
    N = size( B, 2 )
    K = size( A, 2 )
    !$acc kernels
    do k = 1, K
        !$acc loop independent
        do j = 1, N
            tmp = B(k,j)
            do i = 1, M
                C(i,j) = C(i,j) + A(i,k)*tmp
            end do
        end do
    end do
    !$acc end kernels
end subroutine gemm

! gemv OpenACC
subroutine gemv( A, x, y )
    real, dimension(:,:), intent(in) :: A
    real, dimension(:), intent(in) :: x
    real, dimension(:), intent(inout) :: y
    integer :: M, N, i, j
    M = size( A, 1 )
    N = size( A, 2 )
    !$acc kernels
    do j = 1, N
        !$acc loop independent
        do i = 1, M
            y(i) = y(i) + A(i,j)*x(j)
        end do
    end do
    !$acc end kernels
end subroutine gemv

! gemv OpenACC
subroutine gemv2( A, x, y )
    real, dimension(:,:), intent(in) :: A
    real, dimension(:), intent(in) :: x
    real, dimension(:), intent(inout) :: y
    integer :: M, N, i, j
    M = size( A, 1 )
    N = size( A, 2 )
    !$acc kernels
    do i = 1, M
        !$acc loop independent
        do j = 1, N
            y(i) = y(i) + A(i,j)*x(j)
        end do
    end do
    !$acc end kernels
end subroutine gemv2

! gemv OpenACC
subroutine gemv3( A, x, y )

=======

subroutine gemm(A, B, C, n)
  !$acc kernels
  !$acc loop independent
  do i=1,n
    !$acc loop independent
    do j=1,n
      !$acc loop independent
      do k=1,n
        C(i,j) = C(i,j) + A(i,k) * B(k,j)
      enddo
    enddo
  enddo
  !$acc end kernels
end subroutine gemm

! gemm OpenMP
subroutine gemm(A, B, C, n)
  !$omp parallel
  !$omp do private(i,j,k)
  do i=1,n
    do j=1,n
      do k=1,n
        C(i,j) = C(i,j) + A(i,k) * B(k,j)
      enddo
    enddo
  enddo
  !$omp end parallel
end subroutine gemm

! gemm OpenMP+OpenACC
subroutine gemm(A, B, C, n)
  !$acc parallel loop independent
  !$acc loop independent
  do i=1,n
    !$acc loop independent
    do j=1,n
      !$acc loop independent
      do k=1,n
        C(i,j) = C(i,j) + A(i,k) * B(k,j)
      enddo
    enddo
  enddo
end subroutine gemm

! gemm OpenMP+OpenACC+MPI
subroutine gemm(A, B, C, n)
  !$acc parallel loop independent
  !$acc loop independent
  do i=1,n
    !$acc loop independent
    do j=1,n
      !$acc loop independent
      do k=1,n
        C(i,j) = C(i,j) + A(i,k) * B(k,j)
      enddo
    enddo
  enddo
end subroutine gemm

! gemm OpenMP+OpenACC+MPI+PGAS
subroutine gemm(A, B, C, n)
  !$acc parallel loop independent
  !$acc loop independent
  do i=1,n
    !$acc loop independent
    do j=1,n
      !$acc loop independent
      do k=1,n
        C(i,j) = C(i,j) + A(i,k) * B(k,j)
      enddo

=======

subroutine gemm(a,b,c, n)
  real, dimension(n,n), intent(in) :: a,b
  real, dimension(n,n), intent(out) :: c
  integer :: i,j,k
  !$acc parallel loop collapse(2) present(a,b,c)
  do j = 1,n
    do i = 1,n
      c(i,j) = 0.0
      do k = 1,n
        c(i,j) = c(i,j) + a(i,k) * b(k,j)
      end do
    end do
  end do
end subroutine gemm

! gemm OpenMP
subroutine gemm(a,b,c, n)
  real, dimension(n,n), intent(in) :: a,b
  real, dimension(n,n), intent(out) :: c
  integer :: i,j,k
  !$omp parallel do default(shared) private(i,j,k) schedule(static)
  do j = 1,n
    do i = 1,n
      c(i,j) = 0.0
      do k = 1,n
        c(i,j) = c(i,j) + a(i,k) * b(k,j)
      end do
    end do
  end do
end subroutine gemm

! gemm OpenMP with collapse
subroutine gemm(a,b,c, n)
  real, dimension(n,n), intent(in) :: a,b
  real, dimension(n,n), intent(out) :: c
  integer :: i,j,k
  !$omp parallel do default(shared) private(i,j,k) schedule(static) collapse(2)
  do j = 1,n
    do i = 1,n
      c(i,j) = 0.0
      do k = 1,n
        c(i,j) = c(i,j) + a(i,k) * b(k,j)
      end do
    end do
  end do
end subroutine gemm

! gemm OpenMP with collapse and simd
subroutine gemm(a,b,c, n)
  real, dimension(n,n), intent(in) :: a,b
  real, dimension(n,n), intent(out) :: c
  integer :: i,j,k
  !$omp parallel do default(shared) private(i,j,k) schedule(static) collapse(2) simd
  do j = 1,n
    do i = 1

=======

subroutine gemm(a,b,c,lda,ldb,ldc,m,n,k,alpha,beta)
  use acc, only: acc_present, acc_present_or_copyin
  use acc, only: acc_update, acc_update_device, acc_update_self
  use acc, only: acc_enter_data, acc_exit_data, acc_copyin, acc_copyout
  use acc, only: acc_copy_async, acc_copyout_async
  use acc, only: acc_copyin_async, acc_create, acc_delete
  use acc, only: acc_get_device_num, acc_get_num_devices
  use acc, only: acc_set_device_num, acc_set_device_type
  use acc, only: acc_async_test, acc_async_wait
  use acc, only: acc_is_present, acc_present_or_copyin
  use acc, only: acc_present_or_copy, acc_copy
  use acc, only: acc_wait, acc_wait_async
  use acc, only: acc_update_self, acc_update_device
  use acc, only: acc_update_host, acc_update
  use acc, only: acc_enter_data, acc_exit_data
  use acc, only: acc_create, acc_delete
  use acc, only: acc_get_device_num, acc_get_num_devices
  use acc, only: acc_set_device_num, acc_set_device_type
  use acc, only: acc_async_test, acc_async_wait
  use acc, only: acc_is_present, acc_present_or_copyin
  use acc, only: acc_present_or_copy, acc_copy
  use acc, only: acc_wait, acc_wait_async
  use acc, only: acc_update_self, acc_update_device
  use acc, only: acc_update_host, acc_update
  use acc, only: acc_enter_data, acc_exit_data
  use acc, only: acc_create, acc_delete
  use acc, only: acc_get_device_num, acc_get_num_devices
  use acc, only: acc_set_device_num, acc_set_device_type
  use acc, only: acc_async_test, acc_async_wait
  use acc, only: acc_is_present, acc_present_or_copyin
  use acc, only: acc_present_or_copy, acc_copy
  use acc, only: acc_wait, acc_wait_async
  use acc, only: acc_update_self, acc_update_device
  use acc, only: acc_update
