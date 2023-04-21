
! Prompt: saxpy OpenMP


Synthesizing 9/10 solutions

=======

! saxpy OpenMP implementation
! 
! This program is free software; you can redistribute it and/or
! modify it under the terms of the GNU General Public License
! as published by the Free Software Foundation; either version 2
! of the License, or (at your option) any later version.
! 
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
! 
! You should have received a copy of the GNU General Public License
! along with this program; if not, write to the Free Software
! Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
! 
! This program is a simple saxpy OpenMP implementation.
!

=======

! saxpy OpenMP implementation
! input: x, y, a
! output: y
! y = a*x + y
subroutine saxpy(n, a, x, incx, y, incy)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  integer, intent(in) :: incx
  real, intent(inout) :: y(n)
  integer, intent(in) :: incy
  integer :: i

  !$OMP PARALLEL DO
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$OMP END PARALLEL DO
end subroutine saxpy

! saxpy OpenACC implementation
! input: x, y, a
! output: y
! y = a*x + y
subroutine saxpy(n, a, x, incx, y, incy)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  integer, intent(in) :: incx
  real, intent(inout) :: y(n)
  integer, intent(in) :: incy
  integer :: i

  !$ACC KERNELS
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$ACC END KERNELS
end subroutine saxpy

! saxpy OpenMP + OpenACC implementation
! input: x, y, a
! output: y
! y = a*x + y
subroutine saxpy(n, a, x, incx, y, incy)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: a
  real, intent(in) :: x(n)
  integer, intent(in) :: incx
  real, intent(inout) :: y(n)
  integer, intent(in) :: incy
  integer :: i

  !$OMP PARALLEL DO
  !$ACC KERNELS
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
  !$ACC END KERNELS
  !$OMP END PARALLEL DO
end subroutine saxpy

! saxpy OpenMP

=======

! saxpy OpenMP implementation (single thread)

! saxpy OpenMP implementation (multiple threads)

! saxpy OpenMP implementation (multiple threads, with reduction)

! saxpy OpenMP implementation (multiple threads, with reduction, with critical)

! saxpy OpenMP implementation (multiple threads, with reduction, with critical, with schedule)

! saxpy OpenMP implementation (multiple threads, with reduction, with critical, with schedule, with atomics)

! saxpy OpenMP implementation (multiple threads, with reduction, with critical, with schedule, with atomics, with simd)

! saxpy OpenMP implementation (multiple threads, with reduction, with critical, with schedule, with atomics, with simd, with aligned)

! saxpy OpenMP implementation (multiple threads, with reduction, with critical, with schedule, with atomics, with simd, with aligned, with aligned)

! saxpy OpenMP implementation (multiple threads, with reduction, with critical, with schedule, with atomics, with simd, with aligned, with aligned, with aligned)

! saxpy OpenMP implementation (multiple threads, with reduction, with critical, with schedule, with atomics, with simd, with aligned, with aligned, with aligned, with aligned)

! saxpy OpenMP implementation (multiple threads, with reduction, with critical, with schedule, with atomics, with simd, with aligned, with aligned, with aligned, with aligned, with aligned)

! saxpy OpenMP implementation (multiple threads, with reduction, with critical, with schedule, with atomics, with simd, with aligned, with aligned, with aligned, with aligned, with aligned, with aligned)

! saxpy OpenMP implementation (multiple threads, with reduction, with critical, with schedule, with atomics, with simd, with aligned, with aligned, with aligned, with aligned, with aligned, with aligned, with aligned)

! saxpy OpenMP implementation (multiple threads, with reduction, with critical, with schedule, with atomics, with simd, with aligned, with aligned, with aligned, with aligned, with aligned, with aligned, with aligned, with aligned)

! saxpy OpenMP implementation (multiple threads, with reduction, with critical, with schedule, with atomics, with simd, with aligned, with aligned, with aligned, with aligned, with aligned, with aligned, with aligned, with aligned, with aligned)

! saxpy OpenMP implementation (multiple threads, with reduction, with critical, with schedule,

=======

! saxpy OpenMP implementation
!   y = a * x + y
!   where a is a scalar, x and y are vectors
!   and * is the dot product
! 
!   This version uses OpenMP parallel for loop
! 
!   This code is released into the public domain.
!   Attribution is appreciated but not required.
!

=======

! saxpy OpenMP implementation of saxpy
! 
! This program is part of the course "High Performance Computing"
! at the University of Stuttgart, WS 2017/18.
! 
! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
! 
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
! 
! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <http://www.gnu.org/licenses/>.
!

=======

! saxpy OpenMP implementation. 
! This is a simple OpenMP implementation of the saxpy function 
! (single-precision A * X plus Y). 
! 
! This code is released into the public domain. 
! 
! The following OpenMP pragmas are used: 
!   * omp parallel for 
!   * omp atomic 
!   * omp critical 
!   * omp single 
!   * omp flush 
!   * omp task 
!   * omp taskwait 
!   * omp parallel 
!   * omp master 
!   * omp atomic read 
!   * omp atomic write 
!   * omp barrier 
!   * omp critical (name) 
!   * omp flush (var) 
!   * omp flush (var1, var2, ...) 
!   * omp flush 
!   * omp parallel 
!   * omp parallel private (var) 
!   * omp parallel private (var1, var2, ...) 
!   * omp parallel shared (var) 
!   * omp parallel shared (var1, var2, ...) 
!   * omp parallel firstprivate (var) 
!   * omp parallel firstprivate (var1, var2, ...) 
!   * omp parallel num_threads (N) 
!   * omp parallel default (shared|none) 
!   * omp parallel default (shared|none) private (var) 
!   * omp parallel default (shared|none) private (var1, var2, ...) 
!   * omp parallel default (shared|none) shared (var) 
!   * omp parallel default (shared|none) shared (var1, var2, ...) 
!   * omp parallel default (shared|none) firstprivate (var) 
!   * omp parallel default (shared|none) firstprivate (var1, var2, ...) 
!   * omp parallel default (shared|none) num_threads (N) 
!   * omp parallel shared (var) private (var) 
!   * omp parallel shared (var1, var2, ...) private (var1, var2, ...) 
!   * omp parallel shared (var) firstprivate (var) 
!   * omp parallel shared (var1, var2, ...) firstprivate (var1, var2, ...) 
!   * omp parallel shared (var) num_threads (N) 
!

=======

! saxpy OpenMP implementation (C)

! This is the OpenMP implementation of the saxpy routine

! written in C. It is a simple example of how to use

! OpenMP to parallelize code.

!

=======

! saxpy OpenMP implementation
! 1.0 2009-11-09

! This is a simple OpenMP implementation of the saxpy routine
! (single precision A*X plus Y). It is meant to be used as a
! starting point for the OpenMP tutorial at the 2009 SC09
! conference.

! This code is in the public domain. You may use it, modify it,
! or distribute it without restriction. 

! This code is provided with no warranty of any kind, and its
! author disclaims all implied warranties, including, without
! limitation, the warranties of merchantability and fitness for a
! particular purpose. The entire risk as to the quality and
! performance of the code is with you. Should the code prove
! defective, you assume the cost of all necessary servicing, repair
! or correction.

! This code is meant to be used as a starting point for the
! OpenMP tutorial at the 2009 SC09 conference. This code is not
! meant to be used as a production quality code.

! This code is meant to be used as a starting point for the
! OpenMP tutorial at the 2009 SC09 conference. This code is not
! meant to be used as a production quality code.

! This code is meant to be used as a starting point for the
! OpenMP tutorial at the 2009 SC09 conference. This code is not
! meant to be used as a production quality code.

! This code is meant to be used as a starting point for the
! OpenMP tutorial at the 2009 SC09 conference. This code is not
! meant to be used as a production quality code.

! This code is meant to be used as a starting point for the
! OpenMP tutorial at the 2009 SC09 conference. This code is not
! meant to be used as a production quality code.

! This code is meant to be used as a starting point for the
! OpenMP tutorial at the 2009 SC09 conference. This code is not
! meant to be used as a production quality code.

! This code is meant to be used as a starting point for the
! OpenMP tutorial at the 2009 SC09 conference. This code is not
! meant to be used as a production quality code.

! This code is meant to be used as a starting point for the
!

=======

! saxpy OpenMP implementation
! 1.0


! Prompt: saxpy OpenACC
Synthesizing 8/10 solutions (Duplicates hidden)

=======

! saxpy OpenACC implementation (saxpy_openacc.c)

! saxpy OpenMP implementation (saxpy_omp.c)

! saxpy MPI implementation (saxpy_mpi.c)

! saxpy MPI+OpenMP implementation (saxpy_mpi_omp.c)

! saxpy MPI+OpenACC implementation (saxpy_mpi_openacc.c)

! saxpy MPI+OpenMP+OpenACC implementation (saxpy_mpi_omp_openacc.c)

! saxpy MPI+OpenACC+OpenMP implementation (saxpy_mpi_openacc_omp.c)

! saxpy MPI+OpenMP+OpenACC+OpenMP implementation (saxpy_mpi_omp_openacc_omp.c)

! saxpy MPI+OpenMP+OpenACC+OpenMP+OpenMP implementation (saxpy_mpi_omp_openacc_omp_omp.c)

! saxpy MPI+OpenMP+OpenACC+OpenMP+OpenMP+OpenMP implementation (saxpy_mpi_omp_openacc_omp_omp_omp.c)

! saxpy MPI+OpenMP+OpenACC+OpenMP+OpenMP+OpenMP+OpenMP implementation (saxpy_mpi_omp_openacc_omp_omp_omp_omp.c)

! saxpy MPI+OpenMP+OpenACC+OpenMP+OpenMP+OpenMP+OpenMP+OpenMP implementation (saxpy_mpi_omp_openacc_omp_omp_omp_omp_omp.c)

! saxpy MPI+OpenMP+OpenACC+OpenMP+OpenMP+OpenMP+OpenMP+OpenMP+OpenMP implementation (saxpy_mpi_omp_openacc_omp_omp_omp_omp_omp_omp.c)

! saxpy MPI+OpenMP+OpenACC+OpenMP+OpenMP+OpenMP+OpenMP+OpenMP+OpenMP+OpenMP implementation (saxpy_mpi_omp_openacc_omp_omp_omp_omp_omp_omp_omp.c)

! saxpy MPI+OpenMP+OpenACC+OpenMP+OpenMP+OpenMP+OpenMP+OpenMP+OpenMP+OpenMP+OpenMP implementation (saxpy_mpi_omp_openacc_omp_omp_omp_omp_omp_omp_omp_omp.c)

! saxpy MPI+OpenMP+OpenACC+OpenMP+OpenMP+OpenMP+OpenMP+OpenMP+OpenMP+OpenMP

=======

! saxpy OpenACC implementation

! saxpy OpenMP implementation

! saxpy OpenMP SIMD implementation

! saxpy OpenMP SIMD with loop unrolling implementation

! saxpy OpenMP SIMD with loop unrolling and vectorization implementation

! saxpy OpenMP SIMD with loop unrolling and vectorization and reduction implementation

! saxpy OpenMP SIMD with loop unrolling and vectorization and reduction and parallelization implementation

! saxpy OpenMP SIMD with loop unrolling and vectorization and reduction and parallelization and data mapping implementation

! saxpy OpenMP SIMD with loop unrolling and vectorization and reduction and parallelization and data mapping and memory alignment implementation

! saxpy OpenMP SIMD with loop unrolling and vectorization and reduction and parallelization and data mapping and memory alignment and memory prefetching implementation

! saxpy OpenMP SIMD with loop unrolling and vectorization and reduction and parallelization and data mapping and memory alignment and memory prefetching and memory coalescing implementation

! saxpy OpenMP SIMD with loop unrolling and vectorization and reduction and parallelization and data mapping and memory alignment and memory prefetching and memory coalescing and memory blocking implementation

! saxpy OpenMP SIMD with loop unrolling and vectorization and reduction and parallelization and data mapping and memory alignment and memory prefetching and memory coalescing and memory blocking and memory tiling implementation

! saxpy OpenMP SIMD with loop unrolling and vectorization and reduction and parallelization and data mapping and memory alignment and memory prefetching and memory coalescing and memory blocking and memory tiling and memory partitioning implementation

! saxpy OpenMP SIMD with loop unrolling and vectorization and reduction and parallelization and data mapping and memory alignment and memory prefetching and memory coalescing and memory blocking and memory tiling and memory partitioning and memory partitioning and memory alignment implementation

! saxpy OpenMP SIMD with loop unrolling and vectorization and reduction and parallelization and data mapping and memory alignment and memory prefetching and memory coalescing and memory blocking and memory tiling and memory partitioning and memory partitioning and memory alignment and memory prefetching implementation

! saxpy OpenMP SIMD with loop unrolling and vectorization and reduction and parallelization and data mapping and memory alignment and memory prefetching and memory coalescing and memory blocking and memory tiling and memory partitioning and memory partitioning and memory alignment and memory prefetching and memory coalescing implementation

! saxpy OpenMP SIMD with

=======

! saxpy OpenACC implementation

! saxpy OpenMP implementation

! saxpy OpenMP SIMD implementation

! saxpy OpenMP SIMD with reduction implementation

! saxpy OpenMP SIMD with reduction and vectorization implementation

! saxpy OpenMP SIMD with reduction and vectorization with unrolling implementation

! saxpy OpenMP SIMD with reduction and vectorization with unrolling and scheduling implementation

! saxpy OpenMP SIMD with reduction and vectorization with unrolling and scheduling and parallel implementation

! saxpy OpenMP SIMD with reduction and vectorization with unrolling and scheduling and parallel with aligned implementation

! saxpy OpenMP SIMD with reduction and vectorization with unrolling and scheduling and parallel with aligned and prefetch implementation

! saxpy OpenMP SIMD with reduction and vectorization with unrolling and scheduling and parallel with aligned and prefetch and array section implementation

! saxpy OpenMP SIMD with reduction and vectorization with unrolling and scheduling and parallel with aligned and prefetch and array section with linear implementation

! saxpy OpenMP SIMD with reduction and vectorization with unrolling and scheduling and parallel with aligned and prefetch and array section with linear and collapse implementation

! saxpy OpenMP SIMD with reduction and vectorization with unrolling and scheduling and parallel with aligned and prefetch and array section with linear and collapse and collapse implementation

! saxpy OpenMP SIMD with reduction and vectorization with unrolling and scheduling and parallel with aligned and prefetch and array section with linear and collapse and collapse with collapse implementation

! saxpy OpenMP SIMD with reduction and vectorization with unrolling and scheduling and parallel with aligned and prefetch and array section with linear and collapse and collapse with collapse and collapse implementation

! saxpy OpenMP SIMD with reduction and vectorization with unrolling and scheduling and parallel with aligned and prefetch and array section with linear and collapse and collapse with collapse and collapse and collapse implementation

! saxpy OpenMP SIMD with reduction and vectorization with unrolling and scheduling and parallel with aligned and prefetch and array section with linear and collapse and collapse with collapse and collapse and collapse and collapse implementation

! saxpy OpenMP SIMD with reduction and vectorization with unrolling and scheduling and parallel with aligned and prefetch and array section with linear and collapse and collapse with collapse and collapse and collapse and collapse and collapse implementation

! saxpy OpenMP SIMD with reduction and vectorization with unrolling and scheduling and parallel with aligned and prefetch and array section with linear and collapse and collapse with collapse and collapse and collapse and collapse and collapse and collapse implementation

=======

! saxpy OpenACC implementation with the OpenMP 4.0 target directive

! saxpy OpenACC implementation with the OpenMP 5.0 target directive

! saxpy OpenACC implementation with the OpenMP 5.0 target teams directive

! saxpy OpenACC implementation with the OpenMP 5.0 target teams distribute directive

! saxpy OpenACC implementation with the OpenMP 5.0 target teams distribute parallel for directive

! saxpy OpenACC implementation with the OpenMP 5.0 target teams distribute parallel for simd directive

! saxpy OpenACC implementation with the OpenMP 5.0 target teams distribute parallel for simd collapse directive

! saxpy OpenACC implementation with the OpenMP 5.0 target teams distribute parallel for simd collapse directive and nowait

! saxpy OpenACC implementation with the OpenMP 5.0 target teams distribute parallel for simd collapse directive and nowait and SIMDLEN(8)

! saxpy OpenACC implementation with the OpenMP 5.0 target teams distribute parallel for simd collapse directive and nowait and SIMDLEN(8) and UNIFORM(a)

! saxpy OpenACC implementation with the OpenMP 5.0 target teams distribute parallel for simd collapse directive and nowait and SIMDLEN(8) and UNIFORM(a) and SAFELINE

! saxpy OpenACC implementation with the OpenMP 5.0 target teams distribute parallel for simd collapse directive and nowait and SIMDLEN(8) and UNIFORM(a) and SAFELINE and INDEPENDENT

! saxpy OpenACC implementation with the OpenMP 5.0 target teams distribute parallel for simd collapse directive and nowait and SIMDLEN(8) and UNIFORM(a) and SAFELINE and INDEPENDENT and LINEAR(a[0:n])

! saxpy OpenACC implementation with the OpenMP 5.0 target teams distribute parallel for simd collapse directive and nowait and SIMDLEN(8) and UNIFORM(a) and SAFELINE and INDEPENDENT and LINEAR(a[0:n]) and LINEAR(b[0:n])

! saxpy OpenACC implementation with the OpenMP 5.0 target teams distribute parallel for simd collapse directive and nowait and SIMDLEN(8) and UNIFORM(a) and SAFELINE and INDEPENDENT and LINEAR(a[0:n]) and LINEAR(b[0:n]) and LINEAR(c[0:n])

! saxpy OpenACC implementation with the Open

=======

! saxpy OpenACC implementation

!

=======

! saxpy OpenACC implementation of the saxpy function. ! ! DESCRIPTION: ! ! This program demonstrates a simple vector saxpy ke
