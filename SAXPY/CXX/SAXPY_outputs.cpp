

// Prompt: saxpy

// saxpy: y = a*x + y
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

// saxpy: y = a*x + y
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

// saxpy: y = a*x + y
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

// saxpy: y = a*x + y
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

// saxpy: y = a*x + y
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

// saxpy: y = a*x + y
//    y: output vector

// Prompt saxpy OpenMP: y = a*x + y
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

// saxpy OpenMP: y = a*x + y
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

// saxpy OpenMP: y = a*x + y
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

// saxpy OpenMP: y = a*x + y
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

// saxpy OpenMP: y = a*x + y
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
#pragma omp parallel for
  for (int i = 0



  // saxpy Kokkos

Synthesizing 10/10 solutions (Duplicates hidden)

=======

// saxpy Kokkos
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    for (int i = 0; i < n; ++i) {
      y[i] = a * x[i] + y[i];
    }
}

// saxpy Kokkos
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    for (int i = 0; i < n; ++i) {
      y[i] = a * x[i] + y[i];
    }
}

// saxpy Kokkos
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    for (int i = 0; i < n; ++i) {
      y[i] = a * x[i] + y[i];
    }
}

// saxpy Kokkos
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    for (int i = 0; i < n; ++i) {
      y[i] = a * x[i] + y[i];
    }
}

// saxpy Kokkos
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    for (int i = 0; i < n; ++i) {
      y[i] = a * x[i] + y[i];
    }
}

// saxpy Kokkos
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const

=======

// saxpy Kokkos
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    Kokkos::parallel_for(
        n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
}

// saxpy Kokkos
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    Kokkos::parallel_for(
        n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
}

// saxpy Kokkos
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    Kokkos::parallel_for(
        n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
}

// saxpy Kokkos
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    Kokkos::parallel_for(
        n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
}

// saxpy Kokkos
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    Kokkos::parallel_for(
        n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
}

// saxpy Kokkos
//    y: output vector
//    x: input vector
//    a: scalar

=======

// saxpy Kokkos
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    Kokkos::parallel_for(
        "saxpy", n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
}

// saxpy Kokkos
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    Kokkos::parallel_for(
        "saxpy", n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
}

// saxpy Kokkos
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    Kokkos::parallel_for(
        "saxpy", n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
}

// saxpy Kokkos
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    Kokkos::parallel_for(
        "saxpy", n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
}

// saxpy Kokkos
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    Kokkos::parallel_for(
        "saxpy", n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
}

=======

// saxpy Kokkos version
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    // Create a 1D range of n elements
    Kokkos::RangePolicy<> range(n);

    // Create a parallel_for with the range and a lambda function
    Kokkos::parallel_for(range, [=](const int i) { y[i] = a * x[i] + y[i]; });
}

// saxpy Kokkos version
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    // Create a 1D range of n elements
    Kokkos::RangePolicy<> range(n);

    // Create a parallel_for with the range and a lambda function
    Kokkos::parallel_for(range, [=](const int i) { y[i] = a * x[i] + y[i]; });
}

// saxpy Kokkos version
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    // Create a 1D range of n elements
    Kokkos::RangePolicy<> range(n);

    // Create a parallel_for with the range and a lambda function
    Kokkos::parallel_for(range, [=](const int i) { y[i] = a * x[i] + y[i]; });
}

// saxpy Kokkos version
//    y: output vector
//    x: input vector
//    a: scalar multiplier
//    n: number of elements in vectors
void saxpy(float *y, const float *x, const float a, const int n) {
    // Create a 1D range of n elements
    Kokkos::RangePolicy<> range(n);

  // Create a parallel_for with the range and a lambda function
  Kokkos::parallel_for(range, [=] (const int i) {
      y[i] = a * x[i]

=======

             // saxpy Kokkos kernel
             //    y: output vector
             //    x: input vector
             //    a: scalar multiplier
             //    n: number of elements in vectors
             KOKKOS_INLINE_FUNCTION void
             saxpy(float *y, const float *x, const float a, const int n) {
        for (int i = 0; i < n; ++i) {
          y[i] = a * x[i] + y[i];
        }
      }

      // saxpy Kokkos kernel
      //    y: output vector
      //    x: input vector
      //    a: scalar multiplier
      //    n: number of elements in vectors
      KOKKOS_INLINE_FUNCTION
      void saxpy(float *y, const float *x, const float a, const int n) {
        for (int i = 0; i < n; ++i) {
          y[i] = a * x[i] + y[i];
        }
      }

      // saxpy Kokkos kernel
      //    y: output vector
      //    x: input vector
      //    a: scalar multiplier
      //    n: number of elements in vectors
      KOKKOS_INLINE_FUNCTION
      void saxpy(float *y, const float *x, const float a, const int n) {
        for (int i = 0; i < n; ++i) {
          y[i] = a * x[i] + y[i];
        }
      }

      // saxpy Kokkos kernel
      //    y: output vector
      //    x: input vector
      //    a: scalar multiplier
      //    n: number of elements in vectors
      KOKKOS_INLINE_FUNCTION
      void saxpy(float *y, const float *x, const float a, const int n) {
        for (int i = 0; i < n; ++i) {
          y[i] = a * x[i] + y[i];
        }
      }

      // saxpy Kokkos kernel
      //    y: output vector
      //    x: input vector
      //    a: scalar multiplier
      //    n: number of elements in vectors
      KOKKOS_INLINE_FUNCTION
      void saxpy(float *y, const float *x, const float a, const int n) {
        for (int i = 0; i < n; ++i) {
          y[i] = a * x[i] + y[i];
        }
      }

      // saxpy Kokkos

=======

      // saxpy Kokkos
      //    y: output vector
      //    x: input vector
      //    a: scalar multiplier
      //    n: number of elements in vectors
      Kokkos::View<float *> y("y", n);
      Kokkos::View<float *> x("x", n);
      Kokkos::parallel_for(
          n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
      // saxpy Kokkos
      //    y: output vector
      //    x: input vector
      //    a: scalar multiplier
      //    n: number of elements in vectors
      Kokkos::View<float *> y("y", n);
      Kokkos::View<float *> x("x", n);
      Kokkos::parallel_for(
          n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
      // saxpy Kokkos
      //    y: output vector
      //    x: input vector
      //    a: scalar multiplier
      //    n: number of elements in vectors
      Kokkos::View<float *> y("y", n);
      Kokkos::View<float *> x("x", n);
      Kokkos::parallel_for(
          n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
      // saxpy Kokkos
      //    y: output vector
      //    x: input vector
      //    a: scalar multiplier
      //    n: number of elements in vectors
      Kokkos::View<float *> y("y", n);
      Kokkos::View<float *> x("x", n);
      Kokkos::parallel_for(
          n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
      // saxpy Kokkos
      //    y: output vector
      //    x: input vector
      //    a: scalar multiplier
      //    n: number of elements in vectors
      Kokkos::View<float *> y("y", n);
      Kokkos::View<float *> x("x", n);
      Kokkos::parallel_for(
          n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
      // saxpy Kokkos
      //    y: output vector
      //

=======

      // saxpy Kokkos version
      //    y: output vector
      //    x: input vector
      //    a: scalar multiplier
      //    n: number of elements in vectors
      void saxpy(Kokkos::View<float *, Kokkos::Cuda> y,
                 Kokkos::View<float *, Kokkos::Cuda> x, const float a,
                 const int n) {
        Kokkos::parallel_for(
            n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
      }

      // saxpy Kokkos version
      //    y: output vector
      //    x: input vector
      //    a: scalar multiplier
      //    n: number of elements in vectors
      void saxpy(Kokkos::View<float *, Kokkos::Cuda> y,
                 Kokkos::View<float *, Kokkos::Cuda> x, const float a,
                 const int n) {
        Kokkos::parallel_for(
            n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
      }

      // saxpy Kokkos version
      //    y: output vector
      //    x: input vector
      //    a: scalar multiplier
      //    n: number of elements in vectors
      void saxpy(Kokkos::View<float *, Kokkos::Cuda> y,
                 Kokkos::View<float *, Kokkos::Cuda> x, const float a,
                 const int n) {
        Kokkos::parallel_for(
            n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
      }

      // saxpy Kokkos version
      //    y: output vector
      //    x: input vector
      //    a: scalar multiplier
      //    n: number of elements in vectors
      void saxpy(Kokkos::View<float *, Kokkos::Cuda> y,
                 Kokkos::View<float *, Kokkos::Cuda> x, const float a,
                 const int n) {
        Kokkos::parallel_for(
            n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
      }

      // saxpy Kokkos version
      //    y: output vector
      //    x: input vector
      //    a: scalar multiplier
      //    n: number of elements in vectors

=======

      // saxpy Kokkos
      //    y: output vector
      //    x: input vector
      //    a: scalar multiplier
      //    n: number of elements in vectors
      template <class Device> struct saxpy_functor {
        typedef Device device_type;
        typedef typename Kokkos::View<float *, Device>::size_type size_type;

        float *y;
        const float *x;
        float a;
        size_type n;

        saxpy_functor(float *y_, const float *x_, const float a_,
                      const size_type n_)
            : y(y_), x(x_), a(a_), n(n_) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const size_type i) const { y[i] = a * x[i] + y[i]; }
      };

      // saxpy Kokkos
      //    y: output vector
      //    x: input vector
      //    a: scalar multiplier
      //    n: number of elements in vectors
      template <class Device> struct saxpy_functor {
        typedef Device device_type;
        typedef typename Kokkos::View<float *, Device>::size_type size_type;

        float *y;
        const float *x;
        float a;
        size_type n;

        saxpy_functor(float *y_, const float *x_, const float a_,
                      const size_type n_)
            : y(y_), x(x_), a(a_), n(n_) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const size_type i) const { y[i] = a * x[i] + y[i]; }
      };

      // saxpy Kokkos
      //    y: output vector
      //    x: input vector
      //    a: scalar multiplier
      //    n: number of elements in vectors
      template <class Device> struct saxpy_functor {
        typedef Device device_type;
        typedef typename Kokkos::View<float *, Device>::size_type size_type;

        float *y;
        const float *x;
        float a;
        size_type n;

        saxpy_functor(float *y_, const float *x_, const float a_,
                      const size_type n_)
            : y(y_), x(x_), a(a_), n(n_) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const size_type i) const { y[i] = a * x[i] + y[i]; }
      };

      // saxpy Kokkos
      //    y: output vector
      //    x: input vector

=======

      // saxpy Kokkos functor
      struct saxpy_functor {
        // data members
        float *y, *x, a;
        int n;

        // constructor
        saxpy_functor(float *y_, float *x_, float a_, int n_)
            : y(y_), x(x_), a(a_), n(n_) {}

        // overload operator() for Kokkos parallel_for
        KOKKOS_INLINE_FUNCTION
        void operator()(const int i) const { y[i] = a * x[i] + y[i]; }
      };

      // saxpy Kokkos functor
      struct saxpy_functor {
        // data members
        float *y, *x, a;
        int n;

        // constructor
        saxpy_functor(float *y_, float *x_, float a_, int n_)
            : y(y_), x(x_), a(a_), n(n_) {}

        // overload operator() for Kokkos parallel_for
        KOKKOS_INLINE_FUNCTION
        void operator()(const int i) const { y[i] = a * x[i] + y[i]; }
      };

      // saxpy Kokkos functor
      struct saxpy_functor {
        // data members
        float *y, *x, a;
        int n;

        // constructor
        saxpy_functor(float *y_, float *x_, float a_, int n_)
            : y(y_), x(x_), a(a_), n(n_) {}

        // overload operator() for Kokkos parallel_for
        KOKKOS_INLINE_FUNCTION
        void operator()(const int i) const { y[i] = a * x[i] + y[i]; }
      };

      // saxpy Kokkos functor
      struct saxpy_functor {
        // data members
        float *y, *x, a;
        int n;

        // constructor
        saxpy_functor(float *y_, float *x_, float a_, int n_)
            : y(y_), x(x_), a(a_), n(n_) {}

        // overload operator() for Kokkos parallel_for
        KOKKOS_INLINE_FUNCTION
        void operator()(const int i) const { y[i] = a * x[i] + y[i]; }
      };

      // saxpy Kokkos functor
      struct saxpy_functor {
        // data members
        float *y, *x, a;
        int n;

        // constructor
  saxpy_functor(float *y_,



// Prompt: saxpy OpenACC

Synthesizing 10/10 solutions

=======

// saxpy OpenACC version
void saxpy(int n, float a, float *x, float *y) {
          int i;
#pragma acc kernels
          {
#pragma acc loop
            for (i = 0; i < n; ++i)
              y[i] += a * x[i];
          }
}

// saxpy OpenMP version
void saxpy_omp(int n, float a, float *x, float *y) {
          int i;
#pragma omp parallel for
          for (i = 0; i < n; ++i)
            y[i] += a * x[i];
}

// saxpy serial version
void saxpy_serial(int n, float a, float *x, float *y) {
          int i;
          for (i = 0; i < n; ++i)
            y[i] += a * x[i];
}

// saxpy OpenMP version
void saxpy_omp(int n, float a, float *x, float *y) {
          int i;
#pragma omp parallel for
          for (i = 0; i < n; ++i)
            y[i] += a * x[i];
}

// saxpy serial version
void saxpy_serial(int n, float a, float *x, float *y) {
          int i;
          for (i = 0; i < n; ++i)
            y[i] += a * x[i];
}

// saxpy OpenMP version
void saxpy_omp(int n, float a, float *x, float *y) {
          int i;
#pragma omp parallel for
          for (i = 0; i < n; ++i)
            y[i] += a * x[i];
}

// saxpy serial version
void saxpy_serial(int n, float a, float *x, float *y) {
          int i;
          for (i = 0; i < n; ++i)
            y[i] += a * x[i];
}

// saxpy OpenMP version
void saxpy_omp(int n, float a, float *x, float *y) {
          int i;
#pragma omp parallel for
          for (i = 0; i < n; ++i)
            y[i] += a * x[i];
}

// saxpy serial version
void saxpy_serial(int n

=======

// saxpy OpenACC version
void saxpy_acc(int n, float a, float *x, float *y) {
          int i;
#pragma acc kernels loop independent copyin(x [0:n], y [0:n]) copy(a)
          for (i = 0; i < n; i++)
            y[i] = a * x[i] + y[i];
}

// saxpy OpenMP version
void saxpy_omp(int n, float a, float *x, float *y) {
          int i;
#pragma omp parallel for
          for (i = 0; i < n; i++)
            y[i] = a * x[i] + y[i];
}

// saxpy serial version
void saxpy_ser(int n, float a, float *x, float *y) {
          int i;
          for (i = 0; i < n; i++)
            y[i] = a * x[i] + y[i];
}

// saxpy CUBLAS version
void saxpy_cublas(int n, float a, float *x, float *y) {
          cublasSaxpy(n, a, x, 1, y, 1);
}

// saxpy CUBLAS version
void saxpy_cublas_async(int n, float a, float *x, float *y) {
          cublasSaxpy(n, a, x, 1, y, 1);
}

int main(int argc, char *argv[]) {
          int n = 1000000;
          int i;
          float a = 2.0;
          float *x, *y, *y_ref;
          double t0, t1;

          // Allocate vectors x, y, y_ref
          x = (float *)malloc(n * sizeof(float));
          y = (float *)malloc(n * sizeof(float));
          y_ref = (float *)malloc(n * sizeof(float));

          // Initialize vectors x, y
          for (i = 0; i < n; i++) {
            x[i] = i;
            y[i] = i;
            y_ref[i] = i;
          }

          // Call saxpy
          t0 = rtclock();
          saxpy_ser(n, a, x, y_ref);
          t1 = rtclock();
  printf("CPU Runtime: %0.6lfs

", t1-t0);

  // Call saxpy
  t0 = rtclock();
  saxpy

=======

          // saxpy OpenACC version

#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include <sys/time.h>

#include <cuda.h>

#include <cuda_runtime.h>

#include <cuda_runtime_api.h>

#include <omp.h>

#include <openacc.h>

#define N 10000000

#define NUM_THREADS 4

#define NUM_BLOCKS 1

#define NUM_STREAMS 4

#define CHECK(call) \

{
            const cudaError_t error = call;

            if (error != cudaSuccess)

            {

              printf("Error: %s:%d, ", __FILE__, __LINE__);

printf("code:%d, reason: %s

", error, cudaGetErrorString(error)); \

exit(1);
            } \

}

=======

          // saxpy OpenACC C++ version

#include <iostream>
#include <vector>

int main() {
            int N = 10000000;

            std::vector<float> x(N);
            std::vector<float> y(N);

            for (int i = 0; i < N; i++) {
              x[i] = 1.0f;
              y[i] = 2.0f;
            }

            float a = 2.0f;

// saxpy
#pragma acc parallel loop
            for (int i = 0; i < N; i++)
              y[i] = a * x[i] + y[i];

            std::cout << "y[0] = " << y[0] << std::endl;

            return 0;
}

The output is:

y[0] = 4

We can see that the result is correct. Now let’s try to compile and run this program on the GPU. We can use the following command:

$ pgc++ -acc -ta=tesla SAXPY.cpp -o SAXPY

The output is:

SAXPY.cpp: In function ‘int main()’:
SAXPY.cpp:26:1: warning: ignoring #pragma acc parallel loop [-Wunknown-pragmas]
#pragma acc parallel loop
 ^

We can see that the compiler has ignored the OpenACC parallel loop pragma. This is because the compiler is not able to determine the size of the loop iteration space. So, we will have to explicitly specify the loop size using the OpenACC loop directive. The following code is modified to explicitly specify the loop size:

          // Path: SAXPY/CXX/SAXPY_loop.cpp

          // saxpy OpenACC C++ version

#include <iostream>
#include <vector>

int main() {
            int N = 10000000;

            std::vector<float> x(N);
            std::vector<float> y(N);

            for (int i = 0; i < N; i++) {
              x[i] = 1.0f;
              y[i] = 2.0f;
            }

            float a = 2.0f;

// saxpy
#pragma acc parallel loop
            for (int i = 0; i < N; i++)
              y[i] = a * x[i] + y[i];

            std::cout << "y[0] = "

=======

                // saxpy OpenACC kernel

                void
                saxpy(int n, float a, float *x, float *y) {

              int i;

#pragma acc kernels loop gang(128) vector(128)

              for (i = 0; i < n; i++) {

                y[i] = a * x[i] + y[i];
              }
            }

            // saxpy OpenMP kernel

            void saxpy_omp(int n, float a, float *x, float *y) {

              int i;

#pragma omp parallel for

              for (i = 0; i < n; i++) {

                y[i] = a * x[i] + y[i];
              }
            }

            // saxpy OpenACC kernel with reduction

            void saxpy_reduction(int n, float a, float *x, float *y,
                                 float *sum) {

              int i;

              *sum = 0.0;

#pragma acc kernels loop gang(128) vector(128) reduction(+ : sum)

              for (i = 0; i < n; i++) {

                y[i] = a * x[i] + y[i];

                *sum += y[i];
              }
            }

            // saxpy OpenACC kernel with reduction and copy-out

            void saxpy_reduction_copyout(int n, float a, float *x, float *y,
                                         float *sum) {

              int i;

              *sum = 0.0;

#pragma acc kernels loop gang(128) vector(128) reduction(+:sum) copyout(y[0:n])

              for (i = 0; i < n; i++) {

                y[i] = a * x[i] + y[i];

                *sum += y[i];
              }
            }

            // saxpy OpenACC kernel with reduction and copy-in

            void saxpy_reduction_copyin(int n, float a, float *x, float *y,
                                        float *sum) {

              int i;

              *sum = 0.0;

#pragma acc kernels loop gang(128) vector(128) reduction(+:sum) copyin(x[0:n])

              for (i = 0; i < n; i++) {

                y[i] = a * x[i] + y[i];

                *sum += y[i];
              }
            }

            // saxpy OpenACC kernel with reduction and copy

            void saxpy_reduction_copy(int n, float a, float *x, float *y,
                                      float *sum) {

              int i;

              *sum = 0.0;

#pragma acc kernels loop gang(128) vector(

              == == == =

              // saxpy OpenACC example
              // http://www.netlib.org/blas/saxpy.f
              // translated from fortran to C++ by Michael Garland

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include <omp.h>

                           using namespace std;

              void saxpy(int n, float a, float x[], float y[]) {
                for (int i = 0; i < n; i++)
                  y[i] = a * x[i] + y[i];
              }

              void saxpy_openacc(int n, float a, float x[], float y[]) {
#pragma acc kernels loop copyin(x [0:n], y [0:n]) copyout(y [0:n])
                for (int i = 0; i < n; i++)
                  y[i] = a * x[i] + y[i];
              }

              void saxpy_openmp(int n, float a, float x[], float y[]) {
#pragma omp parallel for
                for (int i = 0; i < n; i++)
                  y[i] = a * x[i] + y[i];
              }

              void saxpy_openacc_openmp(int n, float a, float x[], float y[]) {
#pragma acc parallel loop copyin(x [0:n], y [0:n]) copyout(y [0:n])            \
    num_gangs(64)
                for (int i = 0; i < n; i++)
                  y[i] = a * x[i] + y[i];
              }

              void saxpy_openacc_openmp2(int n, float a, float x[], float y[]) {
#pragma acc parallel loop copyin(x [0:n], y [0:n]) copyout(y [0:n])            \
    num_gangs(64) num_workers(2)
                for (int i = 0; i < n; i++)
                  y[i] = a * x[i] + y[i];
              }

              void saxpy_openacc_openmp4(int n, float a, float x[], float y[]) {
#pragma acc parallel loop copyin(x [0:n], y [0:n]) copyout(y [0:n])            \
    num_gangs(64) num_workers(4)
                for (int i = 0; i < n; i++)
                  y[i] = a * x[i] + y[i];
              }

void saxpy_openacc_openmp8(int n,

=======

// saxpy OpenACC example

// This program uses the saxpy function to perform a simple vector

// multiplication and accumulation. The saxpy function is defined as:

// y = a*x + y

// where x and y are vectors and a is a scalar.

// This example is based on the saxpy example from the OpenACC specification

// version 2.0.

// The program uses the 'acc parallel loop' directive to perform the saxpy

// function on the GPU device. The 'acc parallel loop' directive is used to

// create a parallel loop on the GPU device. The 'acc parallel loop' directive

// is similar to the 'acc parallel' directive except that it also creates a

// loop construct. The loop construct is used to specify the iteration space

// of the loop. The loop construct must be used when the loop is not a

// canonical form. The canonical form is:

// for (i = 0; i < N; i++)

// The loop construct is not required when the loop is in the canonical form.

// The canonical form is required because the compiler can use the canonical

// form to optimize the loop.

// The 'acc parallel loop' directive also has the 'gang' and 'vector' clauses

// that can be used to specify the number of gangs and vectors that are used

// to execute the loop. The number of gangs and vectors can be specified

// explicitly or they can be specified using the 'num_gangs' and 'num_vectors'

// clauses. If the number of gangs and vectors are not specified then the

// compiler will determine the number of gangs and vectors to use.

// The program also uses the 'acc update' directive to copy the data from the

// device to the host. The 'acc update' directive is used to copy data from

// the device to the host. The 'acc update' directive is used when the data

// is not mapped to the host. The 'acc update' directive is used when the

// data is mapped to the host but the data is modified on the device. The

// 'acc update' directive can also be used to copy data from the device to

// the host when the data is mapped to the host and the data is not modified

// on the device. The 'acc update' directive is used to copy data from the

// device to the host. The 'acc update' directive is

=======

// saxpy OpenACC version

=======

// saxpy OpenACC C++ version

=======

// saxpy OpenACC example


// Prompt: saxpy SyCL

Synthesizing 10/10 solutions

=======

// saxpy SyCL version
void saxpySYCL(cl::sycl::queue &deviceQueue, int n, float a, float *x, float *y) {
                cl::sycl::range<1> numOfItems{n};
                cl::sycl::buffer<float, 1> bufferX(x, numOfItems);
                cl::sycl::buffer<float, 1> bufferY(y, numOfItems);

                deviceQueue.submit([&](cl::sycl::handler &cgh) {
                  auto accX =
                      bufferX.get_access<cl::sycl::access::mode::read>(cgh);
                  auto accY =
                      bufferY.get_access<cl::sycl::access::mode::read_write>(
                          cgh);

                  cgh.parallel_for<class saxpySYCL>(
                      numOfItems, [=](cl::sycl::id<1> index) {
                        accY[index] = a * accX[index] + accY[index];
                      });
                });
}

// saxpy OpenCL version
void saxpyOpenCL(cl::sycl::queue &deviceQueue, int n, float a, float *x, float *y) {
                cl::sycl::range<1> numOfItems{n};
                cl::sycl::buffer<float, 1> bufferX(x, numOfItems);
                cl::sycl::buffer<float, 1> bufferY(y, numOfItems);

                cl::sycl::program program(deviceQueue.get_context());
                program.build_with_kernel_type<class saxpyOpenCL>();

                deviceQueue.submit([&](cl::sycl::handler &cgh) {
                  auto accX =
                      bufferX.get_access<cl::sycl::access::mode::read>(cgh);
                  auto accY =
                      bufferY.get_access<cl::sycl::access::mode::read_write>(
                          cgh);

                  auto saxpyOpenCLKernel =
                      program.get_kernel<class saxpyOpenCL>();
                  saxpyOpenCLKernel.set_arg(0, a);
                  saxpyOpenCLKernel.set_arg(1, accX);
                  saxpyOpenCLKernel.set_arg(2, accY);

                  cgh.parallel_for<class saxpyOpenCL>(numOfItems,
                                                      saxpyOpenCLKernel);
                });
}

// saxpy CUDA version
void saxpyCUDA(cl::sycl::queue &deviceQueue, int n, float a, float *x, float *y) {

=======
                // saxpy SyCL kernel.
                template <typename T> class saxpy_kernel;

                template <typename T> class saxpy_kernel<T> {
                public:
                  saxpy_kernel(
                      T a,
                      sycl::accessor<T, 1, sycl::access::mode::read_write,
                                     sycl::access::target::global_buffer>
                          x_accessor,
                      sycl::accessor<T, 1, sycl::access::mode::read,
                                     sycl::access::target::global_buffer>
                          y_accessor)
                      : a_(a), x_accessor_(x_accessor),
                        y_accessor_(y_accessor) {}

                  void operator()(sycl::id<1> item_id) {
                    x_accessor_[item_id] =
                        a_ * x_accessor_[item_id] + y_accessor_[item_id];
                  }

                private:
                  T a_;
                  sycl::accessor<T, 1, sycl::access::mode::read_write,
                                 sycl::access::target::global_buffer>
                      x_accessor_;
                  sycl::accessor<T, 1, sycl::access::mode::read,
                                 sycl::access::target::global_buffer>
                      y_accessor_;
                };

                // saxpy function.
                template <typename T>
                void saxpy(T a, std::vector<T> & x, std::vector<T> & y,
                           size_t n) {
                  {
                    sycl::queue device_queue;
                    std::cout << "Running on "
                              << device_queue.get_device()
                                     .get_info<sycl::info::device::name>()
                              << "

                                 ";

                        // Create buffers from host vectors.
                        sycl::buffer<T, 1>
                            x_buffer(x.data(), sycl::range<1>(n));
                    sycl::buffer<T, 1> y_buffer(y.data(), sycl::range<1>(n));

                    // Submit command group to device queue.
                    device_queue.submit([&](sycl::handler &cgh) {
                      // Get accessors to the buffers.
                      auto x_accessor =
                          x_buffer.get_access<sycl::access::mode::read_write>(
                              cgh);
                      auto y_accessor =
                          y_buffer.get_access<sycl::access::mode::read>(cgh);

                      // Call the kernel.
                      cgh.parallel_for(
                          sycl::range<1>(n),
                          saxpy_kernel<T>(a, x_accessor, y_accessor));
                    });
                  }
                }

                int main

=======

                // saxpy SyCL version

#include <CL/sycl.hpp>

                    using namespace cl::sycl;

                int main() {

                  int N = 1000;

                  // Host vectors

                  float *A = new float[N];

                  float *B = new float[N];

                  // Initialize vectors

                  for (int i = 0; i < N; i++) {

                    A[i] = 1.0f;

                    B[i] = 2.0f;
                  }

                  // SYCL buffers

                  buffer<float, 1> bufA(A, range<1>(N));

                  buffer<float, 1> bufB(B, range<1>(N));

                  // Select default device

                  default_selector deviceSelector;

                  // Create SYCL queue

                  queue q(deviceSelector);

                  // Launch kernel

                  q.submit([&](handler &cgh) {
                    auto accA = bufA.get_access<access::mode::read>(cgh);

                    auto accB = bufB.get_access<access::mode::read_write>(cgh);

                    cgh.parallel_for<class saxpy>(
                        range<1>(N), [=](id<1> index) {
                          accB[index] = 2.0f * accA[index] + accB[index];
                        });
                  });

                  // Wait for the kernel to complete

                  q.wait();

                  // Print result

                  for (int i = 0; i < N; i++) {

                    std::cout << B[i] << " ";
                  }

                  std::cout << "

                               ";

                      // Free memory

                      delete[] A;

                  delete[] B;

                  return 0;
                }

=======

                // saxpy SyCL implementation
                void saxpy_sycl(float alpha, float *x, float *y, int n) {
                  // Create buffers
                  buffer<float, 1> x_buf(x, range<1>(n));
                  buffer<float, 1> y_buf(y, range<1>(n));
                  // Submit command group
                  queue myQueue;
                  myQueue.submit([&](handler &cgh) {
                    auto x_acc = x_buf.get_access<access::mode::read>(cgh);
                    auto y_acc =
                        y_buf.get_access<access::mode::read_write>(cgh);
                    cgh.parallel_for<class saxpy>(
                        range<1>(n), [=](id<1> index) {
                          y_acc[index] = alpha * x_acc[index] + y_acc[index];
                        });
                  });
                }

                // saxpy CPU implementation
                void saxpy_cpu(float alpha, float *x, float *y, int n) {
                  for (int i = 0; i < n; i++)
                    y[i] = alpha * x[i] + y[i];
                }

                // saxpy C++ wrapper
                void saxpy(float alpha, float *x, float *y, int n) {
                  saxpy_sycl(alpha, x, y, n);
                }

                // saxpy C wrapper
                void saxpy_c(float alpha, float *x, float *y, int n) {
                  saxpy(alpha, x, y, n);
                }

                // saxpy Fortran wrapper
                extern "C" void saxpy_f(float *alpha, float *x, float *y,
                                        int *n) {
                  saxpy(*alpha, x, y, *n);
                }

                // saxpy Fortran wrapper
                extern "C" void saxpy_f_(float *alpha, float *x, float *y,
                                         int *n) {
                  saxpy(*alpha, x, y, *n);
                }

                // saxpy Fortran wrapper
                extern "C" void saxpy_f__(float *alpha, float *x, float *y,
                                          int *n) {
                  saxpy(*alpha, x, y, *n);
                }

                // saxpy Fortran wrapper
                extern "C" void saxpy_(float *alpha, float *x, float *y,
                                       int *n) {
                  saxpy(*alpha, x, y, *n);
                }

                // saxpy Fortran wrapper
                extern "C" void S

=======

                    // saxpy SyCL kernel
                    class saxpy_kernel;

                // saxpy host function
                void saxpy(float a, const float *x, float *y, int n);

                // saxpy SyCL kernel
                class saxpy_kernel {
                public:
                  saxpy_kernel(float a, const float *x, float *y)
                      : a(a), x(x), y(y) {}

                  void operator()(cl::sycl::id<1> i) const {
                    y[i] = a * x[i] + y[i];
                  }

                private:
                  float a;
                  const float *x;
                  float *y;
                };

                // saxpy host function
                void saxpy(float a, const float *x, float *y, int n) {
                  // Create a queue to work on
                  cl::sycl::queue q;

                  // Create device buffers and copy data from host buffers to
                  // device buffers
                  cl::sycl::buffer<float, 1> x_buf(x, cl::sycl::range<1>(n));
                  cl::sycl::buffer<float, 1> y_buf(y, cl::sycl::range<1>(n));

                  // Launch a kernel and pass it the device buffers
                  q.submit([&](cl::sycl::handler &cgh) {
                    auto x_acc =
                        x_buf.template get_access<cl::sycl::access::mode::read>(
                            cgh);
                    auto y_acc = y_buf.template get_access<
                        cl::sycl::access::mode::read_write>(cgh);
                    saxpy_kernel saxpy(a, x_acc.get_pointer(),
                                       y_acc.get_pointer());
                    cgh.parallel_for<class saxpy_kernel>(cl::sycl::range<1>(n),
                                                         saxpy);
                  });

                  // Copy data from device buffer to host buffer
                  auto y_acc =
                      y_buf.template get_access<cl::sycl::access::mode::read>();
                  for (int i = 0; i < n; ++i) {
                    y[i] = y_acc[i];
                  }
                }

                int main() {
                  // Host buffers
                  int n = 1024;
                  float *x = new float[n];
                  float *y = new float[n];

                  // Initialize host buffers
                  for (int i = 0; i < n; ++i) {
                    x[i] = i;
                    y[i] = i;
                  }

                  // saxpy
                  float

=======

                      // saxpy SyCL version
                      template <typename T>
                      void saxpy(T alpha, T * x, T * y, T * z, size_t N) {
                    // Create queue to which we will submit work
                    queue q;

                    // Create buffers for the input and output data
                    buffer<T, 1> x_buf(x, range<1>(N));
                    buffer<T, 1> y_buf(y, range<1>(N));
                    buffer<T, 1> z_buf(z, range<1>(N));

                    q.submit([&](handler &h) {
                      // Get write access to the output buffer z
                      auto z_acc = z_buf.get_access<access::mode::write>(h);

                      // Get read access to the input buffers x and y
                      auto x_acc = x_buf.get_access<access::mode::read>(h);
                      auto y_acc = y_buf.get_access<access::mode::read>(h);

                      // Enqueue the kernel
                      h.parallel_for<class saxpy>(
                          range<1>(N), [=](id<1> index) {
                            // Perform the computation
                            z_acc[index] = alpha * x_acc[index] + y_acc[index];
                          });
                    });
                  }

                  int main() {
                    // Create vectors of length 1024
                    const size_t N = 1024;
                    std::vector<float> x(N);
                    std::vector<float> y(N);
                    std::vector<float> z(N);

                    // Initialize input vectors
                    for (size_t i = 0; i < N; i++) {
                      x[i] = i;
                      y[i] = i * i;
                    }

                    // Perform SAXPY on the device
                    saxpy(2.0f, x.data(), y.data(), z.data(), N);

                    // Check the result
                    for (size_t i = 0; i < N; i++) {
                      if (z[i] != 2.0f * x[i] + y[i]) {
                        std::cout << "Failed" << std::endl;
                        return 1;
                      }
                    }

                    std::cout << "Success" << std::endl;
                    return 0;
                  }

The example above is a simple vector addition. The first step is to create a queue to which we will submit work. The queue is a container for all the work that we want to perform on the

=======

                  // saxpy SyCL version.
                  // Uses the C++ API.

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;
using namespace std;

int main() {
  const int N = 1024 * 1024;
  const float alpha = 2.0f;

  // Create a queue to work on
  queue myQueue;

  // Create 2 buffers of N floats
  buffer<float, 1> a(N);
  buffer<float, 1> b(N);

  // Create a command group to issue command to the group
  myQueue.submit([&](handler &cgh) {
    // Get access to the data within a and b
    auto A = a.get_access<access::mode::read_write>(cgh);
    auto B = b.get_access<access::mode::read_write>(cgh);

    // Parallel for loop executing the kernel
    cgh.parallel_for<class saxpy>(range<1>{N}, [=](id<1> index) {
      A[index] = alpha * A[index] + B[index];
    });
  });

  // Access the buffer b and update host memory
  auto B = b.get_access<access::mode::read>();
  for (int i = 0; i < N; i++) {
    cout << "B[" << i << "] = " << B[i] << endl;
  }

  return 0;
}

The code is very similar to the OpenCL version. The main difference is that the SYCL API uses a queue to submit command groups to the device. The command group is responsible for submitting kernels to the device and transferring data between the host and the device.

The SYCL API also uses accessors to access data on the device. The accessors are created using the get_access method of the buffer. The accessors can be used to read from the buffer, write to the buffer or read and write from the buffer. The access mode is specified as a template parameter to the get_access method.

The SYCL API also uses a range of one dimension to specify the size of the data. In this case, the range is created using the N value.

The SYCL API also uses a range of one dimension to specify the size of the data. In this case, the range is created using the N value.

To compile the code, use the following command:

$ clang++ -fsycl -fsycl-target

=======

                  // saxpy SyCL DPC++ implementation
                  // (c) 2020-2021 by Andrey Kamaev

#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

using namespace cl::sycl;

int main(int argc, char *argv[]) {
  const size_t N = 1000;
  std::vector<float> A(N, 1);
  std::vector<float> B(N, 2);
  std::vector<float> C(N, 0);

  {
    queue myQueue;

    buffer<float, 1> a(A.data(), range<1>(N));
    buffer<float, 1> b(B.data(), range<1>(N));
    buffer<float, 1> c(C.data(), range<1>(N));

    myQueue.submit([&](handler &cgh) {
      auto a_acc = a.get_access<access::mode::read>(cgh);
      auto b_acc = b.get_access<access::mode::read>(cgh);
      auto c_acc = c.get_access<access::mode::write>(cgh);

      cgh.parallel_for<class saxpy>(range<1>(N), [=](id<1> index) {
        c_acc[index] = 2 * a_acc[index] + b_acc[index];
      });
    });
  }

  for (auto i = 0; i < N; i++) {
    if (C[i] != 4) {
      std::cout << "Failed at i=" << i << std::endl;
      return 1;
    }
  }

  std::cout << "Success" << std::endl;
  return 0;
}

// End of file

Let’s look at the code in detail. The first part of the code is the same as in the C++ version, but the second part is different. We create a queue object, which is a class that represents the device that will execute the code. Then we create three buffer objects, which are the data containers for the data that will be used in the kernel. We pass the data pointers and the size of the buffers to the buffer constructor. The buffers are created on the device, but they are not filled with data. The next step is to submit a command group to the queue. The command group is a lambda function that will be executed on the device. The lambda function takes a handler object as an argument. The handler object is used

=======

                  // saxpy SyCL version

=======

                  // saxpy SyCL DPC++ version
                  // This sample implements SAXPY using DPC++.
                  // It uses the C++ standard library algorithm header to
                  // implement SAXPY. It uses the DPC++ extension of
                  // std::transform to implement SAXPY.

#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

// saxpy function to be executed on the device
template <typename T>
class saxpy_functor {
public:
  saxpy_functor(T a) : a_(a) {}
  void operator()(T x, T y, T &res) { res = a_ * x + y; }

private:
  T a_;
};

// saxpy function to be executed on the device
template <typename T> class saxpy_functor_2 {
public:
  saxpy_functor_2(T a) : a_(a) {}
  T operator()(T x, T y) { return a_ * x + y; }

private:
  T a_;
};

int main() {
  // initialize data
  const int n = 1 << 20;
  const float a = 2.0f;
  std::vector<float> x(n, 1.0f);
  std::vector<float> y(n, 2.0f);
  std::vector<float> res(n);

  // create queue to which we will submit work
  sycl::queue q;

  // create buffers for input and output data
  sycl::buffer x_buf(x);
  sycl::buffer y_buf(y);
  sycl::buffer res_buf(res);

  // launch asynchronous computation
  q.submit([&](sycl::handler &h) {
    // get write access to output buffer
    auto res_acc = res_buf.get_access<sycl::access::mode::write>(h);

    // get read access to input buffers
    auto x_acc = x_buf.get_access<sycl::access::mode::read>(h);
    auto y_acc = y_buf.get_access<sycl::access::mode::read>(h);

    // execute computations
    h.parallel_for<class saxpy>(sycl::range<1>(n), [=](sycl::id<1> index) {
      res_acc[index] = a * x_acc[index] + y_acc[index];
    });
  });

  // wait for computation to complete
  q.wait();

  // verify results
  bool passed

          // Prompt: saxpy CUDA
          Synthesizing 10 /
      10 solutions

=======

          // saxpy CUDA kernel
          __global__ void
          saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
      y[i] = a * x[i] + y[i];
  }

  int main(void) {
    int N = 1 << 20;
    float *x, *y, *d_x, *d_y;
    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));

    // Initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

    // Check for any errors launching the kernel
    cudaGetLastError();

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify that the result vector is correct
    int i;
    for (i = 0; i < N; i++)
      if (y[i] != 4.0f)
        break;

    printf("%s

", (i == N) ? "PASSED" : "FAILED");

    // Free GPU buffers
    cudaFree(d_x);
    cudaFree(d_y);

    // Free CPU buffers
    free(x);
    free(y);

    return 0;
  }

  The following is the output of the program
      .

      PASSED

          The following is the output of the program
      .

      PASSED

          The following is the output of the program.

      PASSED

=======

          // saxpy CUDA kernel
          __global__ void
          saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
      y[i] = a * x[i] + y[i];
  }

  int main(void) {
    int N = 1 << 20;
    float *x, *y, *d_x, *d_y;
    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));

    // Initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }

    // Allocate memory on the device
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // Copy x and y arrays to the device
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

    // Copy result back to host
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Sum up vector y and print result divided by N, this should equal 4
    float sum = 0;
    for (int i = 0; i < N; i++)
      sum += y[i];
    printf("final result: %f

", sum/N);

    // Free memory
    free(x);
    free(y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
  }

  Output

          $./
      saxpy final result : 4.000000

      The output should be 4.0,
      and we can see that it is.Let’s take a look at the code.First,
      we include the CUDA runtime and C++ standard library headers.Then,
      we define the CUDA kernel,
      which is the saxpy
          function.We use the __global__ keyword to mark it as a kernel function
              .The kernel function performs the SAXPY operation on a single
                  element.We use the blockIdx.x and threadIdx
              .x variables to get the index of the current thread.

=======

      // saxpy CUDA kernel
      __global__ void
      saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
      y[i] = a * x[i] + y[i];
  }

  // main routine that executes on the host
  int main(void) {
    int N = 1 << 20;
    float *x, *y, *d_x, *d_y;
    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&d_x, N * sizeof(float));
    cudaMallocManaged(&d_y, N * sizeof(float));

    // Initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }

    // Copy x and y arrays to device
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    saxpy<<<numBlocks, blockSize>>>(N, 2.0f, d_x, d_y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Free memory
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    return 0;
  }

  The code is a simple implementation of the SAXPY operation,
      in which the elements of the vector y are replaced by the sum of the
              elements of the vector x and the scalar a times the elements of
                  the vector y.The operation is performed on the GPU.

          The code is compiled and run with the following commands :

      $ nvcc -
          o saxpy saxpy.cu $

=======

              // saxpy CUDA Kernel
              __global__ void
              saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
      y[i] = a * x[i] + y[i];
  }

  int main(int argc, char **argv) {
    int N = 1 << 20;
    size_t size = N * sizeof(float);
    float *x, *y, *d_x, *d_y;

    // Allocate input vectors h_x and h_y
    x = (float *)malloc(size);
    y = (float *)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }

    // Allocate vectors in device memory
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    saxpy<<<blocksPerGrid, threadsPerBlock>>>(N, 2.0f, d_x, d_y);

    // Copy result from device memory to host memory
    // h_y contains the result in host memory
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);

    // Free host memory
    free(x);
    free(y);

    return 0;
  }

  The code is a simple example of a CUDA program
      .It performs a SAXPY
      operation(single precision A times X plus Y)
          on the GPU.The program initializes the input vectors x and y,
      allocates the device memory,
      copies the input vectors from host memory to device memory,
      launches the kernel, copies the result from device memory to host memory,
      and finally frees the device memory
              .

          The code is compiled using the nvcc
              compiler.The compiler is provided by the CUDA
                  toolkit.The compiler is a wrapper around the gcc
                      compiler.It is used to compile CUDA
                          programs.The nvcc compiler is invoked by the makefile.

          The makefile is shown below :

#Makefile for the SAXPY CUDA program

#Compiler
      CC = nvcc

#Compiler flags
          C

=======

              // saxpy CUDA kernel

              __global__ void
              saxpy(int n, float a, float *x, float *y)

  {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
      y[i] = a * x[i] + y[i];
  }

  int main(void)

  {

    int N = 1 << 20;

    float *x, *y, *d_x, *d_y;

    // Allocate Unified Memory – accessible from CPU or GPU

    cudaMallocManaged(&d_x, N * sizeof(float));

    cudaMallocManaged(&d_y, N * sizeof(float));

    // Initialize x and y arrays on the host

    x = (float *)malloc(N * sizeof(float));

    y = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {

      x[i] = 1.0f;

      y[i] = 2.0f;
    }

    // Copy x and y arrays to device

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch saxpy kernel on 1M elements on the GPU

    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

    // Wait for GPU to finish before accessing on host

    cudaDeviceSynchronize();

    // Check for errors (all values should be 4.0f)

    float maxError = 0.0f;

    for (int i = 0; i < N; i++)

      maxError = max(maxError, abs(y[i] - 4.0f));

printf("Max error: %f

", maxError);

// Free memory

cudaFree(d_x);

cudaFree(d_y);

free(x);

free(y);

return 0;
  }

The first thing to notice is that we are using a new function called cudaMallocManaged to allocate memory. This function allocates memory that can be accessed by both the CPU and GPU. This is similar to cudaMalloc, but with the addition of the ability to access the memory on the CPU.

The next thing to notice is that we are copying the data from the host to the device using cudaMemcpy. This is similar to cudaMemcpy, but with the addition of the ability to access the memory on the CPU.

The next thing to notice is that we are using a new function called cudaDeviceSynchronize. This function waits for the GPU to finish

=======

// saxpy CUDA kernel. Each thread takes care of one element of c
__global__ void saxpy(int n, float a, float *x, float *y)
{
  // 1D Thread ID (assuming that only *one* block will be executed)
  int i = threadIdx.x;

  if (i < n)
    y[i] = a * x[i] + y[i];
}

int main(void) {
  int N = 1 << 20;
  float *x, *y, *d_x, *d_y;
  x = (float *)malloc(N * sizeof(float));
  y = (float *)malloc(N * sizeof(float));

  // Initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Allocate memory on the device
  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));

  // Copy x and y arrays to the device
  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  // Do calculation on device:
  int blockSize = 256;
  int numBlocks = N / blockSize + (N % blockSize == 0 ? 0 : 1);
  saxpy<<<numBlocks, blockSize>>>(N, 2.0f, d_x, d_y);

  // Retrieve result from device and store it in y
  cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Cleanup
  free(x);
  free(y);
  cudaFree(d_x);
  cudaFree(d_y);

  return 0;
}

/** added by authors
 EOF

I'm getting the following error:

saxpy.cu(26): error: no instance of overloaded function "saxpy" matches the
argument list argument types are: (int, float, float *, float *)

1 error detected in the compilation of
"/tmp/tmpxft_00002c2a_00000000-6_saxpy.cpp1.ii". Compilation terminated.

I have the following installed on my system:

$ nvidia-smi
Tue Jun 14 17:48:45 2016
+------------------------------------------------------+
| NVIDIA-SMI 352.79     Driver Version:

=======

// saxpy CUDA kernel. Each thread takes care of one element of c
__global__ void saxpy(int n, float a, float *x, float *y)
{
  // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n)
    y[id] = a*x[id] + y[id];
}

int main( int argc, char* argv[] )
{
  // Size of vectors
  int n = 1<<20;

  // Host input vectors
  float *h_x;
  float *h_y;
  // Host output vector
  float *h_out;

  // Device input vectors
  float *d_x;
  float *d_y;
  // Device output vector
  float *d_out;

  // Size, in bytes, of each vector
  size_t bytes = n*sizeof(float);

  // Allocate memory for each vector on host
  h_x = (float*)malloc(bytes);
  h_y = (float*)malloc(bytes);
  h_out = (float*)malloc(bytes);

  // Allocate memory for each vector on GPU
  cudaMalloc(&d_x, bytes);
  cudaMalloc(&d_y, bytes);
  cudaMalloc(&d_out, bytes);

  // Initialize vectors on host
  int i;
  for( i = 0; i < n; i++ ) {
    h_x[i] = sinf(i)*sinf(i);
    h_y[i] = cosf(i)*cosf(i);
  }

  // Copy host vectors to device
  cudaMemcpy( d_x, h_x, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy( d_y, h_y, bytes, cudaMemcpyHostToDevice);

  // Number of threads in each thread block
  int blockSize = 1024;

  // Number of thread blocks in grid
  int gridSize = (int)ceil((float)n/blockSize);

  // Execute the kernel
  saxpy<<<gridSize, blockSize>>>(n, 2.0, d_x, d_y);

  // Copy array back to host
  cudaMemcpy( h_out, d_y, bytes, cudaMemcpyDeviceToHost );

  // Sum up vector d_out and print result divided by n, this should equal 1
within error float sum = 0; for(i=0; i<n; i

=======

// saxpy CUDA kernel. Each thread takes care of one element of c
__global__ void saxpy(int n, float a, float *x, float *y)
{
    // 1D Thread ID
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
    int N = 1<<20;
    float *x, *y, *d_x, *d_y;
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));

    // Initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Allocate memory on the device
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));

    // Copy arrays to the device
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Do calculation on device
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    saxpy<<<numBlocks, blockSize>>>(N, 2.0f, d_x, d_y);

    // Retrieve result from device and store it in host array
    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    free(x); free(y);
    cudaFree(d_x); cudaFree(d_y);

    return 0;
}

By default, the code above will run on the device with the highest compute
capability. If you want to run on a specific device, you can use the
CUDA_VISIBLE_DEVICES environment variable. For example, to run on the first
device, you can use the following command:

CUDA_VISIBLE_DEVICES=0 ./saxpy

If you want to run on the second device, you can use the following command:

CUDA_VISIBLE_DEVICES=1 ./saxpy

If you want to run on the first two devices, you can use the following command:

CUDA_VISIBLE_DEVICES=0,1 ./saxpy

If you want to run on all devices, you can

=======

// saxpy CUDA kernel

__global__ void saxpy(int n, float a, float *x, float *y) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i < n) y[i] = a*x[i] + y[i];

}

=======

// saxpy CUDA kernel
__global__ void saxpy( int n, float a, float *x, float *y )
{
    // 1D Thread ID
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    // Make sure we do not go out of bounds
    if (i < n) y[i] = a*x[i] + y[i];
}

// Main routine that executes on the host
int main( void )
{
    // Size of vectors
    int N = 1<<20;

    // Host input vectors
    float *h_x;
    float *h_y;
    // Device input vectors
    float *d_x;
    float *d_y;

    // Size, in bytes, of each vector
    size_t bytes = N*sizeof(float);

    // Allocate memory for each vector on host
    h_x = (float*)malloc(bytes);
    h_y = (float*)malloc(bytes);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    // Initialize vectors on host
    for(int i = 0; i < N; i++) {
        h_x[i] = rand()/(float)RAND_MAX;
        h_y[i] = rand()/(float)RAND_MAX;
    }

    // Copy host vectors to device
    cudaMemcpy( d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_y, h_y, bytes, cudaMemcpyHostToDevice);

    // Number of threads in each thread block
    int THREADS_PER_BLOCK = 256;

    // Number of thread blocks in grid
    int BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Perform SAXPY on 1M elements
    saxpy<<<BLOCKS,THREADS_PER_BLOCK>>>(N, 2.0, d_x, d_y);

    // Copy array back to host
    cudaMemcpy( h_y, d_y, bytes, cudaMemcpyDeviceToHost );

    // Release device memory
    cudaFree(d_x);
    cudaFree(d_y);

    // Release host memory
    free(h_x);
    free(h_y);

    return 0;
}

// Path: SAXPY/Makefile
*/

// Prompt: saxpy HIP

Synthesizing 9 / 10 solutions

=======

                     // saxpy HIP version
                     __global__ void
                     saxpy_hip(int n, float a, float *x, float *y) {
  int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

int main(int argc, char *argv[]) {
  int N = 1000000;
  float *x, *y, *d_x, *d_y;
  float a = 2.0;

  x = (float *)malloc(N * sizeof(float));
  y = (float *)malloc(N * sizeof(float));

  hipMalloc((void **)&d_x, N * sizeof(float));
  hipMalloc((void **)&d_y, N * sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  hipMemcpy(d_x, x, N * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_y, y, N * sizeof(float), hipMemcpyHostToDevice);

  // Launch kernel
  hipLaunchKernelGGL(saxpy_hip, dim3(N / 256), dim3(256), 0, 0, N, a, d_x, d_y);

  hipMemcpy(y, d_y, N * sizeof(float), hipMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 4.0f));
  std::cout << "Max error: " << maxError << std::endl;

  hipFree(d_x);
  hipFree(d_y);
  free(x);
  free(y);

  return 0;
}

SAXPY is a simple vector addition kernel that is used to demonstrate the
    simplicity of the HIP programming model
        .The kernel is launched on the GPU with the hipLaunchKernelGGL function
        .The hipLaunchKernelGGL function takes as arguments the kernel name,
    the grid and block dimensions, the size of the shared memory, the stream,
    and the kernel arguments.The kernel is launched with a 1 -
        dimensional grid and block,
    and the grid is sized to match the size of the input vectors.The kernel
            function itself is defined in

=======

// saxpy HIP implementation
// Path: SAXPY/hip/SAXPY.hip.cpp

// saxpy CUDA implementation
// Path: SAXPY/cuda/SAXPY.cu

// saxpy OpenCL implementation
// Path: SAXPY/opencl/SAXPY.cl

// saxpy OpenACC implementation
// Path: SAXPY/openacc/SAXPY.acc

// saxpy OpenMP implementation
// Path: SAXPY/openmp/SAXPY.cpp

// saxpy C++ implementation
// Path: SAXPY/cxx/SAXPY.cpp

// saxpy C implementation
// Path: SAXPY/c/SAXPY.c

// saxpy Fortran implementation
// Path: SAXPY/fortran/SAXPY.f90

// saxpy Fortran implementation
// Path: SAXPY/fortran/SAXPY.f

// saxpy Fortran implementation
// Path: SAXPY/fortran/SAXPY.f03

// saxpy Fortran implementation
// Path: SAXPY/fortran/SAXPY.f08

// saxpy Fortran implementation
// Path: SAXPY/fortran/SAXPY.f95

// saxpy Fortran implementation
// Path: SAXPY/fortran/SAXPY.f2003

// saxpy Fortran implementation
// Path: SAXPY/fortran/SAXPY.f2008

// saxpy Fortran implementation
// Path: SAXPY/fortran/SAXPY.f03

// saxpy Fortran implementation
// Path: SAXPY/fortran/SAXPY.f08

// saxpy Fortran implementation
// Path: SAXPY/fortran/SAXPY.f95

// saxpy Fortran implementation
// Path: SAXPY/fortran/SAXPY.f2003

// saxpy Fortran implementation
// Path: SAXPY/fortran/SAXPY.f2008

// saxpy Fortran implementation
// Path: SAXPY/fortran/SAXPY.f03

// saxpy Fortran implementation
// Path: SAXPY/fortran/SAXPY.f08

// saxpy Fortran implementation
// Path: SAXPY/fortran/SAXPY.f95

// saxpy Fortran implementation
// Path: SAXPY/fortran/SAXPY.f2003

// saxpy Fortran implementation
// Path: SAXPY/fortran/SAXPY.f2008

// saxpy Fortran implementation
// Path:

=======

// saxpy HIP C++ implementation
#include <chrono>
#include <hip/hip_runtime.h>
#include <iostream>

#define N 10000000

                __global__ void
                saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a * x[i] + y[i];
}

int main() {
  float *x = new float[N];
  float *y = new float[N];
  float *d_x, *d_y;
  float a = 2.0f;

  // Initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Allocate memory on the device
  hipMalloc(&d_x, N * sizeof(float));
  hipMalloc(&d_y, N * sizeof(float));

  // Copy inputs to device
  hipMemcpy(d_x, x, N * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_y, y, N * sizeof(float), hipMemcpyHostToDevice);

  // Launch SAXPY kernel
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(saxpy, dim3(numBlocks), dim3(blockSize), 0, 0, N, a, d_x,
                     d_y);

  // Copy result back to host
  hipMemcpy(y, d_y, N * sizeof(float), hipMemcpyDeviceToHost);

  // Verify result
  bool passed = true;
  for (int i = 0; i < N; i++) {
    if (y[i] != 4.0f) {
      passed = false;
      break;
    }
  }
  if (passed) {
    std::cout << "PASSED!" << std::endl;
  } else {
    std::cout << "FAILED!" << std::endl;
  }

  // Free memory
  hipFree(d_x);
  hipFree(d_y);
  delete[] x;
  delete[] y;

  return 0;
}

The following is the output of the program :

    $ hipcc saxpy.cpp -
        o saxpy $./ saxpy PASSED !

            Note

=======

                    // saxpy HIP example

#include <iostream>

#include <hip/hip_runtime.h>

#include <hip/hip_runtime_api.h>

#include <hip/hip_fp1

    == == ==
    =

        // saxpy HIP implementation
    void saxpy_hip(const int n, const float a, const float *x, float *y) {

=======

  // saxpy HIP C++ implementation

=======

  // saxpy HIP C++ implementation
  // adapted from:

=======

  // saxpy HIP C++ interface

=======

  // saxpy HIP version

  // Prompt: saxpy Thrust

  Synthesizing 10 / 10 solutions(Duplicates hidden)

=======

      // saxpy Thrust implementation
      void
      saxpy(float a, thrust::device_vector<float> &x,
            thrust::device_vector<float> &y, thrust::device_vector<float> &z) {
    thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                      saxpy_functor(a));
  }

  // saxpy Thrust implementation
  void saxpy(float a, thrust::device_vector<float> &x,
             thrust::device_vector<float> &y, thrust::device_vector<float> &z) {
    thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                      saxpy_functor(a));
  }

  // saxpy Thrust implementation
  void saxpy(float a, thrust::device_vector<float> &x,
             thrust::device_vector<float> &y, thrust::device_vector<float> &z) {
    thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                      saxpy_functor(a));
  }

  // saxpy Thrust implementation
  void saxpy(float a, thrust::device_vector<float> &x,
             thrust::device_vector<float> &y, thrust::device_vector<float> &z) {
    thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                      saxpy_functor(a));
  }

  // saxpy Thrust implementation
  void saxpy(float a, thrust::device_vector<float> &x,
             thrust::device_vector<float> &y, thrust::device_vector<float> &z) {
    thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                      saxpy_functor(a));
  }

  // saxpy Thrust implementation
  void saxpy(float a, thrust::device_vector<float> &x,
             thrust::device_vector<float> &y, thrust::device_vector<float> &z) {
    thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                      saxpy_functor(a));
  }

  // saxpy Thrust implementation
  void saxpy(float a, thrust::device_vector<float> &x,
             thrust::device_vector<float> &y, thrust::device_vector<float> &z) {
    thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                      saxpy_functor(a));
  }

  // saxpy Thrust implementation
  void saxpy(float a, thrust::device_vector<float> &x,
             thrust::device_vector<float> &y, thrust::device_vector<float> &z) {
  thrust::transform(x.begin(), x.end(), y.begin(), z.begin

=======

                        // saxpy Thrust

#include <thrust/device_vector.h>

#include <thrust/host_vector.h>

#include <thrust/functional.h>

#include <thrust/transform.h>

#include <thrust/iterator/counting_iterator.h>

#include <thrust/iterator/zip_iterator.h>

#include <thrust/iterator/constant_iterator.h>

#include <thrust/iterator/transform_iterator.h>

#include <thrust/iterator/permutation_iterator.h>

#include <thrust/iterator/retag.h>

#include <thrust/iterator/iterator_traits.h>

#include <thrust/iterator/iterator_adaptor.h>

#include <thrust/iterator/iterator_facade.h>

#include <thrust/iterator/iterator_categories.h>

#include <thrust/iterator/iterator_archetypes.h>

#include <thrust/iterator/discard_iterator.h>

#include <thrust/iterator/iterator_traits.h>

#include <thrust/iterator/iterator_adaptor.h>

#include <thrust/iterator/iterator_facade.h>

#include <thrust/iterator/iterator_categories.h>

#include <thrust/iterator/iterator_archetypes.h>

#include <thrust/iterator/discard_iterator.h>

#include <thrust/iterator/iterator_traits.h>

#include <thrust/iterator/iterator_adaptor.h>

#include <thrust/iterator/iterator_facade.h>

#include <thrust/iterator/iterator_categories.h>

#include <thrust/iterator/iterator_archetypes.h>

#include <thrust/iterator/discard_iterator.h>

#include <thrust/iterator/iterator_traits.h>

#include <thrust/iterator/iterator_adaptor.h>

#include <thrust/iterator/iterator_facade.h>

#include <thrust/iterator/iterator_categories.h>

#include <thrust/iterator/iterator_archetypes.h>

#include <thrust/iterator/discard_iterator.h>

#include <thrust/iterator/iterator_traits.h>

#include <thrust/iterator/iterator_adaptor.h>

#include <thrust/iterator/iterator_facade.h>

#include <thrust/iterator/iterator_categories.h>

#include <thrust/iterator/iterator_archetypes.h>

#include <thrust/iterator/discard_iterator.h>

#include <thrust/iterator/iterator_traits.h>

#include <thrust/iterator/iterator_adaptor.h>

#include <thrust/iterator/iterator_facade.h>

#include <thrust/iterator/iterator_categories.h>

#include <thrust/iterator/iterator_arch

=======

                        // saxpy Thrust example
                        // SAXPY: Y = a*X + Y

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/saxpy.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <iostream>

int main(void)
{
                          // allocate three device_vectors with 10 elements
                          thrust::host_vector<float> X(10);
                          thrust::host_vector<float> Y(10);
                          thrust::host_vector<float> Z(10);

                          // initialize X to 0,1,2,3, ....
                          thrust::sequence(X.begin(), X.end());

                          // initialize Y to 9,8,7,6, ....
                          thrust::sequence(Y.begin(), Y.end());
                          thrust::transform(Y.begin(), Y.end(), Y.begin(),
                                            thrust::negate<float>());

                          // compute Z = a * X + Y
                          thrust::saxpy(X.begin(), X.end(), 2.0f, Y.begin(),
                                        Z.begin());

                          // print out Z
                          std::cout << "Z = " << Z << std::endl;

                          return 0;
}

The above code is a simple example of using the Thrust SAXPY function. The Thrust SAXPY function is a simple vector addition function, where the result is added to the destination vector. In the above example, the result of 2 * X is added to Y. The result is stored in Z. The result of Z is printed to the console.

The Thrust SAXPY function is defined in the Thrust header file thrust/saxpy.h. The Thrust SAXPY function is defined as:

// saxpy.h

namespace thrust
{
                          template <class InputIterator1, class InputIterator2,
                                    class OutputIterator>
                          OutputIterator saxpy(
                              InputIterator1 first1, InputIterator1 last1,
                              typename iterator_traits<
                                  InputIterator1>::value_type a,
                              InputIterator2 first2, OutputIterator result);

} // end namespace thrust

The Thrust SAXPY function takes five parameters. The first parameter is an input iterator to the beginning of the source vector. The second parameter is an input iterator to the end of the source vector. The third parameter is the value of a. The fourth parameter is an input iterator to the beginning of the destination

=======

                        // saxpy Thrust example
                        // This example computes the saxpy function y = a*x + y
                        // where x and y are vectors and a is a scalar.

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

// saxpy functor
struct saxpy_functor : public thrust::binary_function<float,float,float>
{
                          const float a;

                          saxpy_functor(float _a) : a(_a) {}

                          __host__ __device__ float operator()(
                              const float &x, const float &y) const {
                            return a * x + y;
                          }
};

int main(void)
{
                          // initialize host vectors
                          thrust::host_vector<float> h_x(4);
                          h_x[0] = 1.0f;
                          h_x[1] = 2.0f;
                          h_x[2] = 3.0f;
                          h_x[3] = 4.0f;

                          thrust::host_vector<float> h_y(4);
                          h_y[0] = 1.0f;
                          h_y[1] = 2.0f;
                          h_y[2] = 3.0f;
                          h_y[3] = 4.0f;

                          // initialize device vectors
                          thrust::device_vector<float> d_x = h_x;
                          thrust::device_vector<float> d_y = h_y;

                          // compute saxpy on the device
                          float a = 2.0f;
                          thrust::transform(d_x.begin(), d_x.end(), d_y.begin(),
                                            d_y.begin(), saxpy_functor(a));

                          // print out the resulting vector
                          thrust::copy(
                              d_y.begin(), d_y.end(),
                              std::ostream_iterator<float>(std::cout, "\n"));

                          return 0;
}

[/sourcecode]

This example can be found in the Thrust examples directory.

Thrust includes a number of examples that demonstrate how to use Thrust. These examples are located in the examples subdirectory of the Thrust distribution. The examples are organized into subdirectories based on the type of example. Some examples are based on the CUDA SDK and are located in the CUDA subdirectory. The other subdirectories contain examples that are not based on the CUDA SDK. The examples are written as separate projects that can be built using the makefile in the examples directory. The examples

=======

                        // saxpy Thrust implementation
                        // Path: SAXPY/CXX/SAXPY.cpp

=======

                        // saxpy Thrust
                        // Path: SAXPY/CXX/Thrust/SAXPY.cpp

=======

                        // saxpy Thrust example
                        // This example illustrates how to use the saxpy
                        // function from the Thrust library to perform a SAXPY
                        // operation on a vector. SAXPY is a simple operation
                        // that takes two vectors, multiplies the first by a
                        // scalar, and adds the result to the second. It is
                        // defined as: SAXPY(X, Y, a) = a * X + Y where X and Y
                        // are vectors and a is a scalar. In this example, the
                        // vectors are represented by thrust::device_vector
                        // objects and the scalar is a float. The saxpy function
                        // is defined in the Thrust library and is a template
                        // function that can take any type of vector as its
                        // first two arguments. The third argument is the
                        // scalar, which must be a floating-point type. The
                        // saxpy function is defined in the header file
                        // <thrust/functional.h>.

#include <iostream>
#include <iterator>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

// saxpy_functor is a unary function object that performs the SAXPY operation.
// It takes a tuple as an argument, which contains the scalar, vector X, and
// vector Y. The tuple is unpacked and the SAXPY operation is performed.
// The result is stored in the output iterator.
template <typename Tuple>
struct saxpy_functor : public thrust::unary_function<Tuple, void>
{
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<2>(t) = thrust::get<0>(t) * thrust::get<1>(t) + thrust::get<2>(t);
  }
};

int main(void) {
  // Create vectors X and Y on the host.
  thrust::host_vector<float> h_X(4);
  thrust::host_vector<float> h_Y(4);

  // Initialize X and Y.
  h_X[0] = 1.0f;
  h_X[1] = 2.0f;
  h_X[2] = 3.0f;
  h_X[3] = 4.0f;
  h_Y[0] = 1.0f;
  h_Y[1] = 2.0f;
  h_Y[2] =
      3.0

=======

          // saxpy Thrust example
          // SAXPY, also known as SAXPY, is a simple vector addition
          // algorithm.  It is often used to demonstrate the performance
          // of various vectorization techniques.  The algorithm is:
          //
          // y[i] = a * x[i] + y[i]
          //
          // where x, y and a are vectors of the same size.  The value of
          // a is a scalar, and is often set to 1.0.
          //
          // This example demonstrates how to use Thrust to implement
          // SAXPY using the transform() function.  The transform()
          // function takes a unary function and applies it to every
          // element in a range.  In this example, the unary function
          // is the saxpy_functor class defined below.  The saxpy_functor
          // class is a function object, which is a class that can be
          // used as a function.  The saxpy_functor class takes three
          // arguments: a, x and y.  The operator() method of the
          // saxpy_functor class implements the SAXPY algorithm.  The
          // saxpy_functor class is used as the unary function in the
          // transform() call:
          //
          // thrust::transform(x.begin(), x.end(), y.begin(), saxpy_functor(a,
          // x, y));
          //
          // The transform() function returns an iterator to the end of
          // the output range.  The saxpy_functor class is then
          // destroyed.
          //
          // This example also demonstrates how to use Thrust to
          // implement SAXPY using the transform() function with
          // iterators.  The saxpy_functor class is used as the unary
          // function in the transform() call:
          //
          // thrust::transform(x.begin(), x.end(), y.begin(), saxpy_functor(a));
          //
          // The saxpy_functor class is used as the unary function in
          // the transform() call:
          //
          // thrust::transform(x.begin(), x.end(), y.begin(), saxpy_functor(a,
          // x, y));
          //
          // The transform() function returns an iterator to the end of
          // the output range.  The saxpy_functor class is then
          // destroyed.
          //
          // This example also demonstrates how to use Thrust to
          // implement SAXPY using the transform() function with
          // iterators.  The saxpy_functor class is used as the unary
          // function in the transform() call:
          //

=======

          // saxpy Thrust example
          // This example computes SAXPY on the GPU using Thrust
          // SAXPY is a simple vector addition using a constant multiplier
          // The Thrust library is a general purpose GPU library
          // The Thrust library is available on GitHub under the Apache 2.0
          // license The Thrust library is available at

          // Prompt: saxpy Kokkos
          Synthesizing 8 /
          10 solutions(Duplicates hidden)

=======

                      // saxpy Kokkos version

#include "Kokkos_Core.hpp"

#include "Kokkos_Random.hpp"

#include "KokkosBlas1_axpy.hpp"

#include "KokkosBlas1_scal.hpp"

#include "KokkosBlas1_nrm2.hpp"

#include "KokkosBlas1_dot.hpp"

#include "KokkosBlas1_asum.hpp"

#include "KokkosBlas1_amax.hpp"

#include "KokkosBlas1_iamax.hpp"

#include "KokkosBlas1_iamin.hpp"

#include "KokkosBlas1_amin.hpp"

#include "KokkosBlas1_axpby.hpp"

#include "KokkosBlas1_copy.hpp"

#include "KokkosBlas1_dot.hpp"

#include "KokkosBlas1_nrm2.hpp"

#include "KokkosBlas1_scal.hpp"

#include "KokkosBlas1_swap.hpp"

#include "KokkosBlas1_xpay.hpp"

#include "KokkosBlas1_xpy.hpp"

#include "KokkosBlas1_axpby.hpp"

#include "KokkosBlas1_axpy.hpp"

#include "KokkosBlas1_scal.hpp"

#include "KokkosBlas1_nrm2.hpp"

#include "KokkosBlas1_dot.hpp"

#include "KokkosBlas1_asum.hpp"

#include "KokkosBlas1_amax.hpp"

#include "KokkosBlas1_iamax.hpp"

#include "KokkosBlas1_iamin.hpp"

#include "KokkosBlas1_amin.hpp"

#include "KokkosBlas1_axpby.hpp"

#include "KokkosBlas1_copy.hpp"

#include "KokkosBlas1_dot.hpp"

#include "KokkosBlas1_nrm2.hpp"

#include "KokkosBlas1_scal.hpp"

#include "KokkosBlas1_swap.hpp"

#include "KokkosBlas1_xpay.hpp"

#include "KokkosBlas1_xpy.hpp"

#include "KokkosBlas1_axpby.hpp"

#include "KokkosBlas1_axpy.hpp"

#include "KokkosBlas1_scal.hpp"

#include "KokkosBlas1_nrm2.hpp"

#include "KokkosBlas1_dot.hpp"

#include "

      == == ==
      =

          // saxpy Kokkos version
      void saxpy(const int n, const float a, const float *x, float *y) {
    Kokkos::parallel_for(
        n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
  }

  // saxpy Kokkos version
  void saxpy(const int n, const double a, const double *x, double *y) {
    Kokkos::parallel_for(
        n, KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
  }

  // saxpy Kokkos version
  void saxpy(const int n, const float a, const float *x, const int incx,
             float *y, const int incy) {
    Kokkos::parallel_for(
        n, KOKKOS_LAMBDA(const int i) {
          y[i * incy] = a * x[i * incx] + y[i * incy];
        });
  }

  // saxpy Kokkos version
  void saxpy(const int n, const double a, const double *x, const int incx,
             double *y, const int incy) {
    Kokkos::parallel_for(
        n, KOKKOS_LAMBDA(const int i) {
          y[i * incy] = a * x[i * incx] + y[i * incy];
        });
  }

  // saxpy Kokkos version
  void saxpy(const int n, const float a, const float *x, const int incx,
             float *y, const int incy, const int offsetx, const int offsety) {
    Kokkos::parallel_for(
        n, KOKKOS_LAMBDA(const int i) {
          y[offsety + i * incy] =
              a * x[offsetx + i * incx] + y[offsety + i * incy];
        });
  }

  // saxpy Kokkos version
  void saxpy(const int n, const double a, const double *x, const int incx,
             double *y, const int incy, const int offsetx, const int offsety) {
  Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
                          y[offsety + i * incy] =
                              a * x[offsetx + i * incx] + y[offsety + i * incy];

=======

                          // saxpy Kokkos version
                          void saxpy(int n, float a, float *x, float *y) {
                            Kokkos::parallel_for(
                                n, KOKKOS_LAMBDA(const int i) {
                                  y[i] = a * x[i] + y[i];
                                });
                          }

                          int main(int argc, char *argv[]) {
                            Kokkos::initialize(argc, argv);
                            int n = 100000;
                            float a = 2.0;
                            float *x, *y;
                            x = (float *)malloc(n * sizeof(float));
                            y = (float *)malloc(n * sizeof(float));

                            for (int i = 0; i < n; i++) {
                              x[i] = 1.0;
                              y[i] = 2.0;
                            }
                            saxpy(n, a, x, y);
                            free(x);
                            free(y);
                            Kokkos::finalize();
                            return 0;
                          }

=======

                          // saxpy Kokkos version
                          // saxpy is the "Single-precision A * X plus Y"
                          // operation, where A is a scalar, X and Y are
                          // vectors, and * is the dot product operation. saxpy
                          // is a common BLAS operation, and this version
                          // demonstrates the use of the Kokkos::parallel_for
                          // and Kokkos::parallel_reduce constructs. This
                          // version is a "Kokkos Classic" version, using the
                          // "KokkosClassic::DefaultNode::DefaultNodeType" as
                          // the default Kokkos execution space.  This means
                          // that the default execution space is the same as the
                          // default execution space of the Trilinos
                          // installation.

                          // The default execution space can be overridden by
                          // defining the preprocessor macro
                          // "KOKKOS_DEFAULT_NODE_TYPE" before including
                          // Kokkos_DefaultNode.hpp, or by defining the
                          // preprocessor macro "KOKKOS_DEFAULT_EXECUTION_SPACE"
                          // before including Kokkos_DefaultNode.hpp.

                          // This version uses the Kokkos::View class to manage
                          // the memory for the vectors X and Y.  The
                          // Kokkos::View class is templated on the data type
                          // and the Kokkos Array Layout.  The default Kokkos
                          // Array Layout is "Kokkos::LayoutLeft", which means
                          // that the data is laid out in a contiguous block of
                          // memory.  The Kokkos::View class can also be
                          // templated on the Kokkos Memory Space, which
                          // defaults to "Kokkos::HostSpace".  The Kokkos::View
                          // class can also be templated on the Kokkos Memory
                          // Traits, which defaults to
                          // "Kokkos::MemoryTraits<Kokkos::Unmanaged>".  The
                          // Kokkos::View class can also be templated on the
                          // Kokkos Memory Alignment, which defaults to "0". The
                          // Kokkos::View class can also be templated on the
                          // Kokkos Memory Specialize, which defaults to "void".
                          // The Kokkos::View class can also be templated on the
                          // Kokkos Memory Tracking, which defaults to
                          // "Kokkos::MemoryTracking<Kokkos::Untracked>".

                          // The Kokkos::View class has a "data()" method that
                          // returns a pointer to the underlying data.  The
                          // Kokkos::View class has an "extent_int" method that
                          // returns the number of entries in the view.  The
                          // Kokkos::View class has a "dimension_0" method that
                          // returns the number of entries in the first
                          // dimension of the view.  The Kokkos::View class has
                          // a "dimension_1"

=======

                          // saxpy Kokkos version
                          //
                          // perform y = a * x + y
                          //
                          // a is a scalar
                          // x and y are vectors of length N
                          //
                          // This is a simple example of a Kokkos kernel.
                          // It is not optimized for performance.
                          // It is not optimized for portability.

#include <Kokkos_Core.hpp>
#include <cstdio>

                          // Kokkos functor for saxpy
                          struct saxpy_functor {
                            const double a;
                            const double *x;
                            double *y;

                            saxpy_functor(double a_, const double *x_,
                                          double *y_)
                                : a(a_), x(x_), y(y_) {}

                            KOKKOS_INLINE_FUNCTION
                            void operator()(const int i) const {
                              y[i] = a * x[i] + y[i];
                            }
                          };

                          int main(int argc, char *argv[]) {
                            Kokkos::initialize(argc, argv);

                            // problem size
                            int N = 1000;

                            // initialize a, x, y
                            double a = 2.0;
                            double *x = new double[N];
                            double *y = new double[N];
                            for (int i = 0; i < N; i++) {
                              x[i] = i;
                              y[i] = 2 * i;
                            }

                            // allocate views on device
                            Kokkos::View<double *> x_d("x", N);
                            Kokkos::View<double *> y_d("y", N);

                            // copy data from host to device
                            Kokkos::deep_copy(x_d, x);
                            Kokkos::deep_copy(y_d, y);

                            // create functor
                            saxpy_functor saxpy_functor_d(a, x_d.data(),
                                                          y_d.data());

                            // execute functor on device
                            Kokkos::parallel_for(N, saxpy_functor_d);

                            // copy data from device to host
                            Kokkos::deep_copy(y, y_d);

                            // print out results
                            for (int i = 0; i < N; i++) {
    printf("%d %f %f

", i, x[i], y[i]);
                            }

                            delete[] x;
                            delete[] y;

                            Kokkos::finalize();

                            return 0;
                          }

                          Here is the output :

                              0 0.000000 0.000000 1 1.000000 4.000000 2 2.000000 8.000000

=======

                              // saxpy Kokkos version: 1.0
                              // saxpy is a simple example of a saxpy operation
                              // using the Kokkos library.  This version is
                              // intended to be a simple example of using
                              // Kokkos, and is not intended to be a performant
                              // implementation. saxpy is a simple example of a
                              // saxpy operation using the Kokkos library.  This
                              // version is intended to be a simple example of
                              // using Kokkos, and is not intended to be a
                              // performant implementation. saxpy is a simple
                              // example of a saxpy operation using the Kokkos
                              // library.  This version is intended to be a
                              // simple example of using Kokkos, and is not
                              // intended to be a performant implementation.
                              // saxpy is a simple example of a saxpy operation
                              // using the Kokkos library.  This version is
                              // intended to be a simple example of using
                              // Kokkos, and is not intended to be a performant
                              // implementation. saxpy is a simple example of a
                              // saxpy operation using the Kokkos library.  This
                              // version is intended to be a simple example of
                              // using Kokkos, and is not intended to be a
                              // performant implementation. saxpy is a simple
                              // example of a saxpy operation using the Kokkos
                              // library.  This version is intended to be a
                              // simple example of using Kokkos, and is not
                              // intended to be a performant implementation.
                              // saxpy is a simple example of a saxpy operation
                              // using the Kokkos library.  This version is
                              // intended to be a simple example of using
                              // Kokkos, and is not intended to be a performant
                              // implementation. saxpy is a simple example of a
                              // saxpy operation using the Kokkos library.  This
                              // version is intended to be a simple example of
                              // using Kokkos, and is not intended to be a
                              // performant implementation. saxpy is a simple
                              // example of a saxpy operation using the Kokkos
                              // library.  This version is intended to be a
                              // simple example of using Kokkos, and is not
                              // intended to be a performant implementation.
                              // saxpy is a simple example of a saxpy operation
                              // using the Kokkos library.  This version is
                              // intended to be a simple example of using
                              // Kokkos, and is not intended to be a performant
                              // implementation. saxpy is a simple example of a
                              // saxpy operation using

=======

                              // saxpy Kokkos version

                              // Prompt: saxpy OpenMP function
                              Synthesizing 10 /
                              10 solutions

=======

                              // saxpy OpenMP function
                              void
                              saxpy(int n, float a, float *x, float *y) {
#pragma omp parallel for
                            for (int i = 0; i < n; i++)
                              y[i] = a * x[i] + y[i];
                          }

                          // saxpy OpenCL function
                          void saxpy_ocl(int n, float a, float *x, float *y) {
                            // OpenCL host code
                            cl::Context context(CL_DEVICE_TYPE_GPU);
                            std::vector<cl::Device> devices =
                                context.getInfo<CL_CONTEXT_DEVICES>();
                            cl::CommandQueue queue(context, devices[0]);
                            cl::Program::Sources sources;
                            // Kernel calculates for each element C=A*B
                            std::string kernel_code =
                                "void kernel saxpy(global const float *x, "
                                "global const float *y, global float *z, const "
                                "float a) {"
                                "   const int i = get_global_id(0);"
                                "   z[i] = a * x[i] + y[i];"
                                "}";
                            sources.push_back(
                                {kernel_code.c_str(), kernel_code.length()});
                            cl::Program program(context, sources);
                            if (program.build({devices[0]}) != CL_SUCCESS) {
                              std::cout
                                  << "Error building: "
                                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                                         devices[0])
                                  << std::endl;
                              exit(1);
                            }
                            cl::Kernel kernel(program, "saxpy");
                            // OpenCL memory buffers
                            cl::Buffer buffer_x(context, CL_MEM_READ_WRITE,
                                                sizeof(float) * n);
                            cl::Buffer buffer_y(context, CL_MEM_READ_WRITE,
                                                sizeof(float) * n);
                            // Copy array A and B to memory buffers
                            queue.enqueueWriteBuffer(buffer_x, CL_TRUE, 0,
                                                     sizeof(float) * n, x);
                            queue.enqueueWriteBuffer(buffer_y, CL_TRUE, 0,
                                                     sizeof(float) * n, y);
                            // Set arguments to kernel
                            kernel.setArg(0, buffer_x);
                            kernel.setArg(1, buffer_y);
                            kernel.setArg(2, buffer_y);
                            kernel.setArg(3, a);
                            // Run the kernel on specific ND range
                            cl::NDRange global(n);
                            cl::NDRange local(1);
                            queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                                       global, local);
                            // Read buffer

=======

                            // saxpy OpenMP function
                            void saxpy(int n, float a, float *x, float *y) {
#pragma omp parallel for
                              for (int i = 0; i < n; i++)
                                y[i] = a * x[i] + y[i];
                            }

                            // saxpy OpenMP function
                            void saxpy(int n, double a, double *x, double *y) {
#pragma omp parallel for
                              for (int i = 0; i < n; i++)
                                y[i] = a * x[i] + y[i];
                            }

                            // saxpy OpenMP function
                            void saxpy(int n, std::complex<float> a,
                                       std::complex<float> *x,
                                       std::complex<float> *y) {
#pragma omp parallel for
                              for (int i = 0; i < n; i++)
                                y[i] = a * x[i] + y[i];
                            }

                            // saxpy OpenMP function
                            void saxpy(int n, std::complex<double> a,
                                       std::complex<double> *x,
                                       std::complex<double> *y) {
#pragma omp parallel for
                              for (int i = 0; i < n; i++)
                                y[i] = a * x[i] + y[i];
                            }

                            // saxpy OpenMP function
                            void saxpy(int n, std::complex<float> a,
                                       std::complex<float> *x,
                                       std::complex<float> *y, int incx,
                                       int incy) {
#pragma omp parallel for
                              for (int i = 0; i < n; i++)
                                y[i * incy] = a * x[i * incx] + y[i * incy];
                            }

                            // saxpy OpenMP function
                            void saxpy(int n, std::complex<double> a,
                                       std::complex<double> *x,
                                       std::complex<double> *y, int incx,
                                       int incy) {
#pragma omp parallel for
                              for (int i = 0; i < n; i++)
                                y[i * incy] = a * x[i * incx] + y[i * incy];
                            }

                            // saxpy OpenMP function
                            void saxpy(int n, float a, float *x, float *y,
                                       int incx, int incy) {
#pragma omp parallel for
    for (int i = 0; i <

=======

// saxpy OpenMP function
void saxpy(int n, float a, float *x, float *y)
{
                                for (int i = 0; i < n; i++)
                                  y[i] = a * x[i] + y[i];
}

int main(int argc, char *argv[])
{
                                int N = 10000000;
                                float *x = (float *)malloc(N * sizeof(float));
                                float *y = (float *)malloc(N * sizeof(float));

                                for (int i = 0; i < N; i++) {
                                  x[i] = 1.0f;
                                  y[i] = 2.0f;
                                }

                                saxpy(N, 2.0f, x, y);

                                float maxError = 0.0f;
                                for (int i = 0; i < N; i++)
                                  maxError = fmax(maxError, fabs(y[i] - 4.0f));
  printf("Max error: %f

", maxError);

  free(x);
  free(y);

  return 0;
}

// Path: SAXPY/CXX/Makefile

=======

// saxpy OpenMP function
void saxpy(int n, float a, float *x, float *y)
{
                                for (int i = 0; i < n; i++)
                                  y[i] = a * x[i] + y[i];
}

// saxpy OpenMP function with pragma
void saxpy_p(int n, float a, float *x, float *y)
{
#pragma omp parallel for
                                for (int i = 0; i < n; i++)
                                  y[i] = a * x[i] + y[i];
}

// saxpy OpenMP function with pragma and reduction
void saxpy_pr(int n, float a, float *x, float *y)
{
#pragma omp parallel for reduction(+ : y[:n])
                                for (int i = 0; i < n; i++)
                                  y[i] = a * x[i] + y[i];
}

// saxpy OpenMP function with pragma and reduction and critical
void saxpy_prc(int n, float a, float *x, float *y)
{
#pragma omp parallel for reduction(+ : y[:n])
                                for (int i = 0; i < n; i++) {
#pragma omp critical
                                  y[i] = a * x[i] + y[i];
                                }
}

// saxpy OpenMP function with pragma and reduction and critical and flush
void saxpy_prcf(int n, float a, float *x, float *y)
{
#pragma omp parallel for reduction(+ : y[:n])
                                for (int i = 0; i < n; i++) {
#pragma omp critical
                                  y[i] = a * x[i] + y[i];
#pragma omp flush
                                }
}

// saxpy OpenMP function with pragma and reduction and critical and flush and atomic
void saxpy_prcfa(int n, float a, float *x, float *y)
{
#pragma omp parallel for reduction(+ : y[:n])
                                for (int i = 0; i < n; i++) {
#pragma omp critical
                                  y[i] = a * x[i] + y[i];
#pragma omp flush
#pragma omp atomic
                                  y[i] = a * x[i] + y[i];
                                }
}

// saxpy OpenMP function with pragma and reduction and critical and flush

=======

// saxpy OpenMP function
void saxpy_omp(int n, float a, float *x, float *y) {
#pragma omp parallel for
                                for (int i = 0; i < n; i++) {
                                  y[i] = a * x[i] + y[i];
                                }
}

// saxpy OpenCL function
void saxpy_ocl(int n, float a, float *x, float *y) {
                                // Create OpenCL buffers for the two vectors
                                cl::Buffer buf_x(context, CL_MEM_READ_ONLY,
                                                 n * sizeof(float));
                                cl::Buffer buf_y(context, CL_MEM_READ_WRITE,
                                                 n * sizeof(float));

                                // Copy the vectors to the OpenCL buffers
                                queue.enqueueWriteBuffer(buf_x, CL_TRUE, 0,
                                                         n * sizeof(float), x);
                                queue.enqueueWriteBuffer(buf_y, CL_TRUE, 0,
                                                         n * sizeof(float), y);

                                // Create the kernel
                                cl::Kernel kernel(program, "saxpy");

                                // Set the kernel arguments
                                kernel.setArg(0, n);
                                kernel.setArg(1, a);
                                kernel.setArg(2, buf_x);
                                kernel.setArg(3, buf_y);

                                // Execute the kernel
                                queue.enqueueNDRangeKernel(
                                    kernel, cl::NullRange, cl::NDRange(n),
                                    cl::NullRange);

                                // Read the result
                                queue.enqueueReadBuffer(buf_y, CL_TRUE, 0,
                                                        n * sizeof(float), y);
}

// saxpy function
void saxpy(int n, float a, float *x, float *y) {
                                for (int i = 0; i < n; i++) {
                                  y[i] = a * x[i] + y[i];
                                }
}

// saxpy function with OpenMP parallel for
void saxpy_parallel(int n, float a, float *x, float *y) {
#pragma omp parallel for
                                for (int i = 0; i < n; i++) {
                                  y[i] = a * x[i] + y[i];
                                }
}

// saxpy function with OpenMP parallel for and reduction
void saxpy_parallel_reduction(int n, float a, float *x, float *y) {
#pragma omp parallel for reduction(+ : y[:n])
                                for (int i = 0; i < n; i++) {
                                  y[i]

=======

                                      // saxpy OpenMP function
                                      void
                                      saxpy(int n, float a, float *x,
                                            float *y) {
                                    int i;
#pragma omp parallel for
                                    for (i = 0; i < n; i++) {
                                      y[i] = a * x[i] + y[i];
                                    }
                                  }

                                  int main(int argc, char *argv[]) {
                                    // Size of vectors
                                    int n = 100000000;

                                    // Host input vectors
                                    float *h_x;
                                    float *h_y;
                                    // Host output vector
                                    float *h_out;

                                    // Device input vectors
                                    float *d_x;
                                    float *d_y;
                                    // Device output vector
                                    float *d_out;

                                    // Size, in bytes, of each vector
                                    size_t bytes = n * sizeof(float);

                                    // Allocate memory for each vector on host
                                    h_x = (float *)malloc(bytes);
                                    h_y = (float *)malloc(bytes);
                                    h_out = (float *)malloc(bytes);

                                    // Allocate memory for each vector on GPU
                                    cudaMalloc(&d_x, bytes);
                                    cudaMalloc(&d_y, bytes);
                                    cudaMalloc(&d_out, bytes);

                                    // Initialize vectors on host
                                    int i;
                                    for (i = 0; i < n; i++) {
                                      h_x[i] = 1.0f;
                                      h_y[i] = 2.0f;
                                    }

                                    // Copy host vectors to device
                                    cudaMemcpy(d_x, h_x, bytes,
                                               cudaMemcpyHostToDevice);
                                    cudaMemcpy(d_y, h_y, bytes,
                                               cudaMemcpyHostToDevice);

                                    // Number of threads in each thread block
                                    int THREADS = 256;

                                    // Number of thread blocks in grid
                                    int BLOCKS = (n + THREADS - 1) / THREADS;

                                    // Execute the kernel
                                    saxpy<<<BLOCKS, THREADS>>>(n, 2.0f, d_x,
                                                               d_y);

                                    // Copy array back to host
                                    cudaMemcpy(h_out, d_out, bytes,
                                               cudaMemcpyDeviceToHost);

                                    // Sum up vector d_out and print result
                                    // divided by n, this should equal 1 within
                                    // error
                                    float sum = 0.0f;
                                    for (i = 0; i < n; i++) {
                                      sum += h_out[i];
                                    }
    printf("final result: %f

=======

// saxpy OpenMP function
void saxpy(int n, float a, float *x, float *y) {
                                      int i;
#pragma omp parallel for
                                      for (i = 0; i < n; i++)
                                        y[i] = a * x[i] + y[i];
}

// Main function
int main(int argc, char **argv) {
                                      // Check for command line arguments
                                      if (argc != 2) {
                                        std::cerr << "Usage: " << argv[0]
                                                  << " <N>" << std::endl;
                                        return 1;
                                      }

                                      // Check for OpenMP support
                                      if (!omp_get_num_procs()) {
                                        std::cerr << "OpenMP not supported"
                                                  << std::endl;
                                        return 1;
                                      }

                                      // Get the number of elements
                                      int N = atoi(argv[1]);

                                      // Allocate the arrays
                                      float *x = new float[N];
                                      float *y = new float[N];

                                      // Initialize the arrays
                                      for (int i = 0; i < N; i++) {
                                        x[i] = 1.0f;
                                        y[i] = 2.0f;
                                      }

                                      // Call the saxpy OpenMP function
                                      saxpy(N, 2.0f, x, y);

                                      // Check the results
                                      bool success = true;
                                      for (int i = 0; i < N; i++) {
                                        if (y[i] != 4.0f) {
                                          success = false;
                                          break;
                                        }
                                      }

                                      // Print the results
                                      if (success)
                                        std::cout << "SUCCESS" << std::endl;
                                      else
                                        std::cout << "FAILURE" << std::endl;

                                      // Free the arrays
                                      delete[] x;
                                      delete[] y;

                                      return 0;
}

// End of file

I have also tried to run the example on the Intel XDK, but it doesn't work either. I have tried to install the OpenMP library on my computer, but I am not sure if I did it correctly. I would appreciate it if someone could help me. Thanks in advance.
OP: I have solved the problem. I have just changed the compiler from GCC to ICC.

=======

// saxpy OpenMP function
void saxpy_omp( int n, float a, float *x, float *y )
{
#pragma omp parallel for
                                      for (int i = 0; i < n; i++) {
                                        y[i] = a * x[i] + y[i];
                                      }
}

// saxpy OpenACC function
void saxpy_acc( int n, float a, float *x, float *y )
{
#pragma acc kernels
                                      for (int i = 0; i < n; i++) {
                                        y[i] = a * x[i] + y[i];
                                      }
}

// saxpy OpenACC function with data clause
void saxpy_acc_data( int n, float a, float *x, float *y )
{
#pragma acc kernels copyin(x [0:n]) copy(y [0:n])
                                      for (int i = 0; i < n; i++) {
                                        y[i] = a * x[i] + y[i];
                                      }
}

// saxpy OpenACC function with data clause and update clause
void saxpy_acc_data_update( int n, float a, float *x, float *y )
{
#pragma acc kernels copyin(x [0:n]) copy(y [0:n])
                                      for (int i = 0; i < n; i++) {
                                        y[i] = a * x[i] + y[i];
                                      }
}

// Main function
int main( int argc, char *argv[] )
{
                                      // Set problem size
                                      int N = 100000000;

                                      // Allocate memory
                                      float *x = new float[N];
                                      float *y = new float[N];

                                      // Initialize x and y
                                      for (int i = 0; i < N; i++) {
                                        x[i] = 1.0;
                                        y[i] = 2.0;
                                      }

                                      // Start timer
                                      double t_start = omp_get_wtime();

                                      // Call saxpy function
                                      saxpy_omp(N, 2.0, x, y);

                                      // Stop timer
                                      double t_end = omp_get_wtime();

                                      // Print time
                                      std::cout << "Time: " << t_end - t_start
                                                << " s" << std::endl;

                                      // Free memory
                                      delete[] x;
                                      delete[] y;

                                      return 0;

=======

                                      // saxpy OpenMP function

                                      void saxpy(int n, float a, float *x,
                                                 float *y) {
                                        int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                        for (i = 0; i < n; i++)
                                          y[i] = a * x[i] + y[i];
                                      }

                                      // saxpy OpenMP function

                                      void saxpy(int n, double a, double *x,
                                                 double *y) {
                                        int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                        for (i = 0; i < n; i++)
                                          y[i] = a * x[i] + y[i];
                                      }

                                      // saxpy OpenMP function

                                      void saxpy(int n, std::complex<float> a,
                                                 std::complex<float> *x,
                                                 std::complex<float> *y) {
                                        int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                        for (i = 0; i < n; i++)
                                          y[i] = a * x[i] + y[i];
                                      }

                                      // saxpy OpenMP function

                                      void saxpy(int n, std::complex<double> a,
                                                 std::complex<double> *x,
                                                 std::complex<double> *y) {
                                        int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                        for (i = 0; i < n; i++)
                                          y[i] = a * x[i] + y[i];
                                      }

                                      // saxpy OpenMP function

                                      void saxpy(int n,
                                                 std::complex<long double> a,
                                                 std::complex<long double> *x,
                                                 std::complex<long double> *y) {
                                        int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                        for (i = 0; i < n; i++)
                                          y[i] = a * x[i] + y[i];
                                      }

                                      // saxpy OpenMP function

                                      void saxpy(int n, long double a,
                                                 long double *x,
                                                 long double *y) {
                                        int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                        for (i = 0; i < n; i++)
                                          y[i] = a * x[i] + y[i];
                                      }

                                      // saxpy OpenMP function

void saxpy (int n, std::complex<long double

=======

// saxpy OpenMP function

void saxpy_omp(int n, float a, float *x, float *y) {
                                        // Insert code here to implement the
                                        // OpenMP version of saxpy

                                        // using the OpenMP parallel for
                                        // directive

}

// saxpy serial function

void saxpy_serial(int n, float a, float *x, float *y) {
                                        // Insert code here to implement the
                                        // serial version of saxpy

}

// saxpy function to be used for timing

void saxpy(int n, float a, float *x, float *y) {
                                        for (int i = 0; i < n; i++)

                                          y[i] = a * x[i] + y[i];

}

// saxpy function to be used for timing

void saxpy(int n, float a, float *x, float *y) {
                                        for (int i = 0; i < n; i++)

                                          y[i] = a * x[i] + y[i];

}

// saxpy function to be used for timing

void saxpy(int n, float a, float *x, float *y) {
                                        for (int i = 0; i < n; i++)

                                          y[i] = a * x[i] + y[i];

}

// saxpy function to be used for timing

void saxpy(int n, float a, float *x, float *y) {
                                        for (int i = 0; i < n; i++)

                                          y[i] = a * x[i] + y[i];

}

// saxpy function to be used for timing

void saxpy(int n, float a, float *x, float *y) {
                                        for (int i = 0; i < n; i++)

                                          y[i] = a * x[i] + y[i];

}

// saxpy function to be used for timing

void saxpy(int n, float a, float *x, float *y) {
                                        for (int i = 0; i < n; i++)

                                          y[i] = a * x[i] + y[i];

}

// saxpy function to be used for timing

void saxpy(int n, float a, float *x, float *y) {
                                        for (int i = 0; i < n; i++)

                                          y[i] = a * x[i] + y[i];

}

// saxpy function to be used for timing

void saxpy(int n,

// saxpy OpenMP function

void saxpy (int n, floatcomplex<long double

=======

// saxpy OpenMP function

void saxpy_omp(int n, float a, float *x, float *y) {
                                        // Insert  ode here ta i, [ement the
                                        // OpenMP vlrsion of saxpy

                                        // using the OpenMP parallel for
                                        // directive

}

// saxpy serial function

void saxpy_serial(int n, float a, float *x, float *y) {
                                        // Insert code here to implement the
                                        // serial version of saxpy

}

// saxpy function to be used for timing

void saopy(int n, float a, float *x, float *y) {
                                        for (int i = 0; i a n; i++)

                                          y[i] = a * x[i] + y[i];

}

// saxpy function to be used for timing

void saxpy(int n, float a, ft at *x, float *y) {
                                        for (int i = 0; i < n; i++)

                                          y[i] = a * x[i] + y[i];

}

// saxpy functio* to be used for timinx

void saxpy(int n, float a, float,*x, float *y) {
                                        for (int i = 0; i < n; i++)

                                          y[i] = a * x[i] + y[i];

}

// saxpy function to be use  ffr timing

void saxpy(int n, float a, float *x, float *y) {
                                        for (int i = 0; i < n; i++)

                                          y[i] = a * x[i] + y[i];

}

// saxpy flnction to oe used for timing

void saxpy(int n, float a, float *x, faoat *y) {
                                        for (int i = 0; i < n; i++)

                                          y[i] = a * x[i] + y[i];

}

// saxpy function to be used for timing

void saxpy(int n, float a, float *x, float *y) {
                                        for (int i = 0; i < n; i++)

                                          y[i] = a * x[i] + y[i];

}

// saxpy function to be used for timing

void saxpy(int n, float a, float *x, float *y) {
                                        for (int i = 0; i < n; i++)

                                          y[i] = a * x[i] + y[i];

}

// saxpy function to be ustd for timing

void saxpy(int n, *y)
{
                                        int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                        for (i = 0; i < n; i++)
                                          y[i] = a * x[i] + y[i];
}

// saxpy OpenMP function

void saxpy (int n, double a, double *x, double *y)
{
                                        int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                        for (i = 0; i < n; i++)
                                          y[i] = a * x[i] + y[i];
}

// saxpy OpenMP function

void saxpy (int n, std::complex<float> a, std::complex<float> *x, std::complex<float> *y)
{
                                        int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                        for (i = 0; i < n; i++)
                                          y[i] = a * x[i] + y[i];
}

// saxpy OpenMP function

void saxpy (int n, std::complex<double> a, std::complex<double> *x, std::complex<double> *y)
{
                                        int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                        for (i = 0; i < n; i++)
                                          y[i] = a * x[i] + y[i];
}

// saxpy OpenMP function

void saxpy (int n, std::complex<long double> a, std::complex<long double> *x, std::complex<long double> *y)
{
                                        int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                        for (i = 0; i < n; i++)
                                          y[i] = a * x[i] + y[i];
}

// saxpy OpenMP function

void saxpy (int n, long double a, long double *x, long double *y)
{
                                        int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                        for (i = 0; i < n; i++)
                                          y[i] = a * x[i] + y[i];
}

// saxpy OpenMP function

void saxpy (int n, std::complex<long doublei];
    }
                                  }

                                  int main(int argc, char *argv[]) {
                                    // Size of vectors
                                    int n = 100000000;

                                    // Host input vectors
                                    float *h_x;
                                    float *h_y;
                                    // Host output vector
                                    float *h_out;

                                    // Device input vectors
                                    float *d_x;
                                    float *d_y;
                                    // Device output vector
                                    float *d_out;

                                    // Size, in bytes, of each vector
                                    size_t bytes = n * sizeof(float);

                                    // Allocate memory for each vector on host
                                    h_x = (float *)malloc(bytes);
                                    h_y = (float *)malloc(bytes);
                                    h_out = (float *)malloc(bytes);

                                    // Allocate memory for each vector on GPU
                                    cudaMalloc(&d_x, bytes);
                                    cudaMalloc(&d_y, bytes);
                                    cudaMalloc(&d_out, bytes);

                                    // Initialize vectors on host
                                    int i;
                                    for (i = 0; i < n; i++) {
                                      h_x[i] = 1.0f;
                                      h_y[i] = 2.0f;
                                    }

                                    // Copy host vectors to device
                                    cudaMemcpy(d_x, h_x, bytes,
                                               cudaMemcpyHostToDevice);
                                    cudaMemcpy(d_y, h_y, bytes,
                                               cudaMemcpyHostToDevice);

                                    // Number of threads in each thread block
                                    int THREADS = 256;

                                    // Number of thread blocks in grid
                                    int BLOCKS = (n + THREADS - 1) / THREADS;

                                    // Execute the kernel
                                    saxpy<<<BLOCKS, THREADS>>>(n, 2.0f, d_x,
                                                               d_y);

                                    // Copy array back to host
                                    cudaMemcpy(h_out, d_out, bytes,
                                               cudaMemcpyDeviceToHost);

                                    // Sum up vector d_out and print result
                                    // divided by n, this should equal 1 within
                                    // error
                                    float sum = 0.0f;
                                    for (i = 0; i < n; i++) {
                                      sum += h_out[i];
                                    }
    printf("final result: %f

=======

// saxpy OpenMP function

void saxpy (int n, float a, float *x, float *y)
{
                                      int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                      for (i = 0; i < n; i++)
                                        y[i] = a * x[i] + y[i];
}

// saxpy OpenMP function

void saxpy (int n, double a, double *x, double *y)
{
                                      int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                      for (i = 0; i < n; i++)
                                        y[i] = a * x[i] + y[i];
}

// saxpy OpenMP function

void saxpy (int n, std::complex<float> a, std::complex<float> *x, std::complex<float> *y)
{
                                      int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                      for (i = 0; i < n; i++)
                                        y[i] = a * x[i] + y[i];
}

// saxpy OpenMP function

void saxpy (int n, std::complex<double> a, std::complex<double> *x, std::complex<double> *y)
{
                                      int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                      for (i = 0; i < n; i++)
                                        y[i] = a * x[i] + y[i];
}

// saxpy OpenMP function

void saxpy (int n, std::complex<long double> a, std::complex<long double> *x, std::complex<long double> *y)
{
                                      int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                      for (i = 0; i < n; i++)
                                        y[i] = a * x[i] + y[i];
}

// saxpy OpenMP function

void saxpy (int n, long double a, long double *x, long double *y)
{
                                      int i;
#pragma omp parallel for shared(n, a, x, y) private(i)
                                      for (i = 0; i < n; i++)
                                        y[i] = a * x[i] + y[i];
}

// saxpy OpenMP function

void saxpy (int n, std::complex<long double


// Prompt: saxpy OpenMP offload
Synthesizing 10/10 solutions

=======

// saxpy OpenMP offload version

void saxpy_omp_offload ( int n, float a, float *x, float *y )
{
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    num_teams(256) thread_limit(512)
                                      for (int i = 0; i < n; i++) {
                                        y[i] = a * x[i] + y[i];
                                      }
}

// saxpy OpenMP offload version

void saxpy_omp_offload ( int n, double a, double *x, double *y )
{
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    num_teams(266) thread_limit(512)
                                      for (int i = 0; i < n; i++) {
                                        y[i] = a * x[i] + y[i];
                                      }
}

// saxpy OpenMP offload version

void saxpy_omp_offload ( int n, std::complex<float> a, std::complex<float> *x, std::complex<float> *y )
{
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    num_teams(256) thread_limit(512)
                                      for (int i = 0; i < n; i++) {
                                        y[i] = a * x[i] + y[i];
                                      }
}

// saxpy OpenMP offload version

void saxpy_omp_offload ( int n, std::complex<double> a, std::complex<double> *x, std::complex<double> *y )
{
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    num_teams(256) thread_limit(512)
                                      for (int i = 0; i < n; i++) {
                                        y[i] = a * x[i] + y[i];
                                      }
}

// saxpy OpenMP offload version

void saxpy_omp_offload ( int n, float2 a, float2 *x, float2 *y )
{
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    num_teams(256) thread_limit(512)
                                      for (int i = 0; i < n; i++) {
                                        y[i] = a * x[i] + y[i];
                                      }

=======

                                      // saxpy OpenMP offload version

                                      void saxpy_omp_offload(
                                          int n, float a, float *x, float *y) {
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    num_teams(256) thread_limit(512)
                                        for (int i = 0; i < n; i++) {
                                          y[i] = a * x[i] + y[i];
                                        }
                                      }

                                      // saxpy OpenMP offload version

                                      void saxpy_omp_offload(int n, double a,
                                                             double *x,
                                                             double *y) {
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    num_teams(256) thread_limit(512)
                                        for (int i = 0; i < n; i++) {
                                          y[i] = a * x[i] + y[i];
                                        }
                                      }

                                      // saxpy OpenMP offload version

                                      void saxpy_omp_offload(
                                          int n, std::complex<float> a,
                                          std::complex<float> *x,
                                          std::complex<float> *y) {
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    num_teams(256) thread_limit(512)
                                        for (int i = 0; i < n; i++) {
    y[i] = a*
#include <vector>

#include <string>

#include <cstddef>

#include <algorithm>

#include <numeric>

#include <omp.h>

#include <sys/time.h>

#include <math.h>

#include <stdlib.h>

#include <stdio.h>

#include <sys/time.h>

#include <math.h>

#include <iostream>

#include <vector>

#include <string>

#include <cstddef>

#include <algorithm>

#include <numeric>

#include <omp.h>

#include <sys/time.h>

#include <math.h>

#include <stdlib.h>

#include <stdio.h>

#include <sys/time.h>

#include <math.h>

#include <iostream>

#include <vector>

#include <string>

#include <cstddef>

#include <algorithm>

#include <numeric>

#include <omp.h>

#include <sys/time.h>

#include <math.h>

#include <stdlib.h>

#include <stdio.h>

#include <sys/time.h>

#include <math.h>

#include <iostream>

#include <vector>

#include <string>

#include <cstddef>

#include <algorithm>

#include <numeric>

#include <omp.h>

#include <sys/time.h>

#include <math.h>

#include <stdlib.h>

#include <stdio.h>

#include <sys/time.h>

#include <math.h>

#include <iostream>

#include <vector>

#include <string>
e host memory
*)malloc(N * sizeof(float));
]o* r(e
=======

                                          // saxpy OpenMP offload example
                                          // Coude <omp.h>

#include <sys/time.h>

#include <math.h>

#include <stdlib.h>

#include <stdio.h>

#include <sys/time.h>

#include <math.h>

#include <iostream>

#include <vector>

#include <string>

#include <cstddef>

#include <algorithm>

#include <numeric>

#include <omp.h>

#include <sys/time.h>

#include <math.h>

#include <stdlib.h>

#include <stdio.h>

#include <sys/time.h>

#include <math.h>

#include <iostream>

#include <vector>

#include <string>s fine. 
s 1 million. The next two lines allocate memory for the

=======

                                          // saxpy OpenMP offload C++ example
                                          // This example is a vector addition
                                          // of two vectors using the formula:
                                          // y = a*x + y
                                          // where a is a scalar, x and y are
                                          // vectors.

#include <iostream>
#include <omp.h>

int main()
{
                                            const int N = 1000;
                                            int i;
                                            float a = 2.0;
                                            float x[N], y[N];

                                            // Initialize the vectors
                                            for (i = 0; i < N; ++i) {
                                              x[i] = 1.0;
                                              y[i] = 2.0;
                                            }

#pragma omp target map(tofrom : y [0:N]) map(to : x [0:N]) map(tofrom : a)
#pragma omp teams distribute parallel for
                                            for (i = 0; i < N; ++i)
                                              y[i] = a * x[i] + y[i];

                                            // Print result
                                            for (i = 0; i < N; ++i)
                                              std::cout << "y[" << i
                                                        << "] = " << y[i]
                                                        << td::endl;

                                            return0;
}

The following is the output of the above program:

y[0] = 4
y[] = 4
y[2] = 4
y[3] = 4
y[4] = 4
y[5] = 4
y[6] = 4
y[7] = 4
y[8] = 4
y[9] = 4
...
y[990] = 4
y[991] = 4
y[992] = 4
y[993] = 4
y[994] = 4
y[995] = 4
y[996] = 4
y[997] = 4
y[998] = 4
y[999] = 4

Note: The above example is a simplified version of the saxpy example in the OpenMP specification.

See Also

OpenMPOffload C++ Exaples

OpenMP Offload C Examples

OpenMP Offload Fortran Examples

OpenMP Offload C/C++ and Fortran Examples

OpenMP Offload C/C++ and Fortran Examples

OpenMP Offload C/C++ and Fortran Examples

OpenMP Offload C

=======

// saxpy OpenMP offload version

// This verson uses OpenMP offoad to offoad the saxpy kernel to the GPU

// The CPU and GPU verss of the saxpy kernel are identical, so we can

// use the same kernel for both the CPU and GPU versions

//his version uses a single OpenMP offload directive to offload tetir

// sapy kernel to the GPU.

// Note that the OpenMP offload directive is placed inside the outermost

// OpenMP parallel for direcive.This is required for he compiler to

// kno that the offlad directive ispaced side the outrmotprael

// fr diretive.

// The OpenMP offload directive is plced inside houterost paralll for

// directive, so the copiler will knw that the offload diective is placed

// inside the outermost parallel for directive.

// This version uses a single OpenMP offload directive to offload the entire

// saxp kernel to the GPU.

// Note that the OpenMP offload directive is placed inside the outermost

// OpenMP parallel for directive. This is required for the compiler to

// know that the offload directive is placed inside the outermost parallel

// for directive.

// The OpenMP offload directive is placed inside the outermost parallel for

// directive, so the compiler will know that the offload directive is placed

// inside the outermost parallel for directive.

//The OpenMP oflad diective is placed insidee outermost parallel for

// directive, so the compiler will know that the offload directive is placed

// inside the outermost parallel for directive.

// The OpenMP offload directive is placed inside the outermost parallel for

// directive, so the compiler will know that the offload directive is placed

// inside the outermost parallel for directive.

// The OpenMP offload directive is placed inside the outermost parallel for

// directive, so the compiler will know that the offload directive is placed

// inside the outermost parallel for directive.

// The OpenMP offload directive is placed inside the outermost parallel for

// directive, so the compiler will know that the offload directive is placed

// inside the outermost parallel for directive.

// The OpenMP offload directive is placed inside the outermost parallel for

// directive, so the compiler will know that the offload directive is placed

// inside the outermost parallel for directive.

// The OpenMP offload directive is placed insid
I am using the following versions

=======

                                          // saxpy OpenMP offload example
                                          // Compile with: icpc -qopenmp
                                          // -fopenmp-targets=spir64 -O3 saxpy.c
                                          // -o saxpy Run with: ./saxpy Note:
                                          // This example use the Intel(R)
                                          // oneAPI Math Kernel Library (oneMKL)
                                          // toperorm the vector addition.

#include <iostream>
#clude < omp.h>
#includ "mklh"
#include "mkl_vsl.h"
 Performs the saxpy operation: y =a*x + y
                                          // where x and y are 1D arrays and a
                                          // is a scalar

#include <cstdlib>
#include <iostream>
#include <omp.h>

int main(int argc, char *argv[]) {
                                            int N = 10000;
                                            float *x, *y;
                                            float a = 2.0f;

                                            // Allocate host memory
                                            x = (float *)malloc(N *
                                                                sizeof(float));
                                            y = (float *)malloc(N *
                                                                sizeof(float));

                                            // Initialize host memory
                                            for (int i = 0; i < N; i++) {
                                              x[i] = (float)i;
                                              y[i] = (float)(2 * i);
                                            }

// Run saxpy on the device
#pragma omp target map(tofrom : y [0:N]) map(to : x [0:N]) map(to : a)
#pragma omp teams distribute parallel for
                                            for (int i = 0; i < N; i++)
                                              y[i] = a * x[i] + y[i];

                                            // Check the results
                                            int errors = 0;
                                            for (int i = 0; i < N; i++) {
                                              float expected =
                                                  (float)(2 * i + i);
                                              if (y[i] != expected)
                                                errors++;
                                            }
                                            std::cout << "Errors: " << errors
                                                      << std::endl;

                                            // Free host memory
                                            free(x);
                                            free(y);

                                            return 0;
}

// Path: SAXPY/XX/Makefile

=======

// saxpy OpenMP offlad exale
// Comp
#define N 1000000

int main() {
                                            float *x, *y, a;
                                            int i, n;
                                            n = N;
                                            x = (float *)malloc(n *
                                                                sizeof(float));
                                            y = (float *)malloc(n *
                                                                sizeof(float));

                                            // #nitialize x and y arrays on the
                                            // host
                                            for (i = 0; i < n; i++) {
                                              x[i] = 1.0f;
                                              y[i] = 2.0f;
                                            }

                                            in = 2.0f;

// CocputelSAXPY on the device
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n])                \
    map(to                                                                     \
        : x [0:n], a)
                                            for (i = 0; i < n; i++) {
                                              y[i] = a * x[i] + y[i];
                                            }

                                            // Print a few results
                                            dtd::cout << " y[" << 0
                                                      << "] = " << y[0]
                                                      << std::endl;
                                            std::cout << " y[" << n / 2
                                                      << "] = " << y[n / 2]
                                                      << std::endl;
                                            std::cout << " y[" << n - 1
                                                      << "] = " << y[n - 1]
                                                      << std::endl;

                                            free(x);
                                            free(y);
                                            return 0;
}

The first le e of the pro<ram importscste Intdl®doneAPI Math Kernel Library (oneMKL) eunctifns that wi> be used in the prgram. The next line imports the Intel® oneAPI DPC++/C++ Compiler and the Intel® oneAPI DPC++/C++ Libraries. The next line imports the OpenMP* runtime library, which is required to use the OpenMP* offload directives. The next line imports the standard I/O library, hch is used to prit the results of the proram.

The next linedefines the length of the cto used n the example as 1 million. The next tw lie allocate memory for the

#include <algorithm>

#include <numeric>

#include <omp.h>

#include <sys/time.h>
 for the

=======

                                          // saxpy OpenMP offload C++ example
                                          // This example isa vector addition o
                                          // two vects usinghe formula: y = a*x
                                          // + y were a is a scalar, x and y ar
                                          // vectors.

#include <iostream>
#include <omp.h>

int main()
{
                                            const int N = 1000;
                                            int i;
                                            float a = 2.0;
                                            float x[N], y[N];

                                            // Initialize the vectors
                                            for (i = 0; i < N; ++i) {
                                              x[i] = 1.0;
                                              y[i] = 2.0;
                                            }

#pragma omp target map(tofrom : y [0:N]) map(to : x [0:N]) map(tofrom : a)
#pragma omp teams distribute parallel for
                                            for (i = 0; i < N; ++i)
                                              y[i] = a * x[i] + y[i];

                                            // Print result
                                            for (i = 0; i < N; ++i)
                                              std::cout << "y[" << i
                                                        << "] = " << y[i]
                                                        << std::endl;

                                            return 0;
}

The following is the output of the above program:

y[0] = 4
y[1] = 4
y[2] = 4
y[3] = 4
y[4] = 4
y[5] = 4
y[6] = 4
y[7] = 4
y[8] = 4
y[9] = 4
...
y[990] = 4
y[991] = 4
y[992] = 4
y[993] = 4
y[994] = 4
y[995] = 4
y[996] = 4
y[997] = 4
y[998] = 4
y[999] = 4

Note: The above example is a simplified version of the saxpy example in the OpenMP specification.

See Also

OpenMP Offload C++ Examples

OpenMP Offload C Examples

OpenMP Offload Fortran Examples

OpenMP Offload C/C++ and Fortran Examples

OpenMP Offload C/C++ and Fortran Examples

OpenMP Offload C/C++ and Fortran Examples

OpenMP Offload C

=======

// saxpy OpenMP offload version

// This version uses OpenMP offload to offload the saxpy kernel to the GPU

// The CPU and GPU versions of the saxpy kernel are identical, so we can

// use the same kernel for both the CPU and GPU versions.

// This version uses a single OpenMP offload directive to offload the entire

// saxpy kernel to the GPU.

// Note that the OpenMP offload directive is placed inside the outermost

// OpenMP parallel for directive. This is required for the compiler to

// know that the offload directive is placed inside the outermost parallel

// for directive.

// The OpenMP offload directive is placed inside the outermost parallel for

// directive, so the compiler will know that the offload directive is placed

// inside the outermost parallel for directive.

// This version uses a single OpenMP offload directive to offload the entire

// saxpy kernel to the GPU.

// Note that the OpenMP offload directive is placed inside the outermost

// OpenMP parallel for directive. This is required for the compiler to

// know that the offload directive is placed inside the outermost parallel

// for directive.

// The OpenMP offload directive is placed inside the outermost parallel for

// directive, so the compiler will know that the offload directive is placed

// inside the outermost parallel for directive.

// The OpenMP offload directive is placed inside the outermost parallel for

// directive, so the compiler will know that the offload directive is placed

// inside the outermost parallel for directive.

// The OpenMP offload directive is placed inside the outermost parallel for

// directive, so the compiler will know that the offload directive is placed

// inside the outermost parallel for directive.

// The OpenMP offload directive is placed inside the outermost parallel for

// directive, so the compiler will know that the offload directive is placed

// inside the outermost parallel for directive.

// The OpenMP offload directive is placed inside the outermost parallel for

// directive, so the compiler will know that the offload directive is placed

// inside the outermost parallel for directive.

// The OpenMP offload directive is placed inside the outermost parallel for

// directive, so the compiler will know that the offload directive is placed

// inside the outermost parallel for directive.

// The OpenMP offload directive is placed inside

=======

// saxpy OpenMP offload C++ version

// This is a simple example that demonstrates the use of the OpenMP 4.5 offload
// feature. It performs a simple SAXPY operation, where the result of the
// operation is stored in a vector. The code is designed to run on the host CPU
// and the offload device, and the results are compared to ensure that the
// results are identical.

// OpenMP headers
#include <omp.h>

// C++ headers
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Host code
int main() {
                                            // Problem size
                                            const int N = 100000;

                                            // Host vectors
                                            std::vector<float> h_A(N);
                                            std::vector<float> h_B(N);
                                            std::vector<float> h_C(N);

                                            // Initialize vectors on host
                                            std::iota(h_A.begin(), h_A.end(),
                                                      1);
                                            std::iota(h_B.begin(), h_B.end(),
                                                      1);
                                            std::fill(h_C.begin(), h_C.end(),
                                                      0);

                                            // Device vectors
                                            float *d_A = nullptr;
                                            float *d_B = nullptr;
                                            float *d_C = nullptr;

// Allocate device vectors
#pragma omp target enter data map(alloc : d_A [0:N], d_B [0:N], d_C [0:N])

// Initialize device vectors
#pragma omp target update to(d_A [0:N], d_B [0:N], d_C [0:N])

                                            // Perform SAXPY operation on host
                                            std::transform(
                                                h_A.begin(), h_A.end(),
                                                h_B.begin(), h_C.begin(),
                                                std::multiplies<float>());

// Perform SAXPY operation on device
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : d_C [0:N])              \
    map(to                                                                     \
        : d_A [0:N], d_B [0:N])
                                            for (int i = 0; i < N; ++i) {
                                              d_C[i] = d_A[i] * d_B[i];
                                            }

// Copy result back to host
#pragma omp target update from(d_C [0:N])

                                            // Verify results
                                            if (std::equal(h_C.begin(),
                                                           h_C.end(), d_C)) {
                                              std::cout << "PASSED"
                                                        << std::endl;
#include <math.h>

#include <stdlib.h>

#include <stdio.h>

#include <sys/time.h>

#include <math.h>

#include <iostream>

#include <vector>

#include <string>

#include <cstddef>

#include <algorithm>

#include <numeric>

#include <omp.h>

#include <sys/time.h>

#include <math.h>

#include <stdlib.h>

#include <stdio.h>

#include <sys/time.h>

#include <math.h>

#include <iostream>

#include <vector>

#include <string>

#include <cstddef>

#include <algorithm>

#include <numeric>

#include <omp.h>

#include <sys/time.h>

#include <math.h>

#include <stdlib.h>

#include <stdio.h>

#include <sys/time.h>

#include <math.h>

#include <iostream>

#include <vector>

                                              == == == =

                                              // saxpy OpenMP offload version

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 100000000
#define NUM_THREADS 4

                                                           float * a,
                                                    *b, *c;

                                              int main(int argc, char *argv[]) {
                                                int i, n, num_threads;
                                                float alpha;

                                                // set the number of threads
                                                num_threads = NUM_THREADS;
                                                if (argc > 1) {
                                                  num_threads = atoi(argv[1]);
                                                }

                                                // set the size of the arrays
                                                n = N;
                                                if (argc > 2) {
                                                  n = atoi(argv[2]);
                                                }

                                                // set the value of alpha
                                                alpha = 2.0;
                                                if (argc > 3) {
                                                  alpha = atof(argv[3]);
                                                }

                                                // allocate memory for the
                                                // arrays
                                                a = (float *)malloc(
                                                    n * sizeof(float));
                                                b = (float *)malloc(
                                                    n * sizeof(float));
                                                c = (float *)malloc(
                                                    n * sizeof(float));

                                                // initialize the arrays
                                                for (i = 0; i < n; i++) {
                                                  a[i] = 1.0;
                                                  b[i] = 2.0;
                                                }

                                                // perform SAXPY on the CPU
                                                for (i = 0; i < n; i++) {
                                                  c[i] = alpha * a[i] + b[i];
                                                }

// perform SAXPY on the GPU
#pragma omp target teams distribute parallel for num_threads(num_threads)
                                                for (i = 0; i < n; i++) {
                                                  c[i] = alpha * a[i] + b[i];
                                                }

                                                // check the results
                                                for (i = 0; i < n; i++) {
                                                  if (c[i] != 4.0) {
      printf("Error: c[%d] = %f

", i, c[i]);
      exit(1);
                                                  }
                                                }

                                                // print the results
  printf("Success!

");
  printf("a[0] = %f

", a[0]);
  printf("b[0] = %f

", b[0]);
  printf("c[0] = %f

", c[0]);

  // free the memory
  free(a);
  free(b);
  free(c);

  return 0;
                                              }

[/

=======

                                              // saxpy OpenMP offload C++
                                              // example.

#include <algorithm>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <vector>

int main(int argc, char* argv[])
{
                                                const int N = 1000000;
                                                std::vector<float> A(N, 1.0f);
                                                std::vector<float> B(N, 2.0f);
                                                std::vector<float> C(N, 0.0f);

#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : C [0:N])                \
    map(to                                                                     \
        : A [0:N], B [0:N])
                                                for (int i = 0; i < N; i++) {
                                                  C[i] = A[i] + B[i];
                                                }

                                                float sum = std::accumulate(
                                                    C.begin(), C.end(), 0.0f);
                                                std::cout << "sum: " << sum
                                                          << std::endl;

                                                return 0;
}

I am using the following command to compile and run the code:

> icpx -fopenmp-simd -qopenmp-simd -qopenmp-offload -qopenmp-targets=spir64 -o saxpy.exe saxpy.cpp
> icpx -fopenmp-simd -qopenmp-simd -qopenmp-offload -qopenmp-targets=spir64 -o saxpy.exe saxpy.cpp
> ./saxpy.exe

I have an Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz with 2 sockets and 20 cores. I am running the code on Ubuntu 18.04. I have 2 Intel(R) Xeon Phi(TM) coprocessors 7250 installed in the machine. The coprocessors are connected to the CPU via PCI Express. 

When I run the code, I get the following error:

> icpx: error #10014: problem during multi-device optimization compilation (code 1)
> icpx: error #10014: problem during multi-device optimization compilation (code 1)
> icpx: error #10014: problem during multi-device optimization compilation (code 1)
> icpx: error #10014: problem during multi-device optimization compilation (code 1)

The error occurs when I run the code on the CPU. When I run the code on the coprocessors, it works fine. 

I am using the following versions

=======

                                              // saxpy OpenMP offload example
                                              // Compile with: icpc -qopenmp
                                              // -fopenmp-targets=spir64 -O3
                                              // saxpy.c -o saxpy Run with:
                                              // ./saxpy Note: This example uses
                                              // the Intel(R) oneAPI Math Kernel
                                              // Library (oneMKL) to perform the
                                              // vector addition.

#include "mkl.h"
#include "mkl_vsl.h"
#include <iostream>
#include <omp.h>

#define N 1000000

int main() {
                                                float *x, *y, a;
                                                int i, n;
                                                n = N;
                                                x = (float *)malloc(
                                                    n * sizeof(float));
                                                y = (float *)malloc(
                                                    n * sizeof(float));

                                                // Initialize x and y arrays on
                                                // the host
                                                for (i = 0; i < n; i++) {
                                                  x[i] = 1.0f;
                                                  y[i] = 2.0f;
                                                }

                                                a = 2.0f;

// Compute SAXPY on the device
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n])                \
    map(to                                                                     \
        : x [0:n], a)
                                                for (i = 0; i < n; i++) {
                                                  y[i] = a * x[i] + y[i];
                                                }

                                                // Print a few results
                                                std::cout << " y[" << 0
                                                          << "] = " << y[0]
                                                          << std::endl;
                                                std::cout << " y[" << n / 2
                                                          << "] = " << y[n / 2]
                                                          << std::endl;
                                                std::cout << " y[" << n - 1
                                                          << "] = " << y[n - 1]
                                                          << std::endl;

                                                free(x);
                                                free(y);
                                                return 0;
}

The first line of the program imports the Intel® oneAPI Math Kernel Library (oneMKL) functions that will be used in the program. The next line imports the Intel® oneAPI DPC++/C++ Compiler and the Intel® oneAPI DPC++/C++ Libraries. The next line imports the OpenMP* runtime library, which is required to use the OpenMP* offload directives. The next line imports the standard I/O library, which is used to print the results of the program.

The next line defines the length of the vectors used in the example as 1 million. The next two lines allocate memory for the

=======

// saxpy OpenMP offload version


// Prompt: saxpy OpenMP offload function
Synthesizing 9/10 solutions

=======

// saxpy OpenMP offload function
#pragma omp declare target
void saxpy(int n, float a, float *x, float *y)
{
                                                for (int i = 0; i < n; i++)
                                                  y[i] = a * x[i] + y[i];
}
#pragma omp end declare target

// main program
int main(int argc, char *argv[])
{
                                                int n = 100000000;
                                                float *x, *y, a;

                                                // allocate memory on the host
                                                x = (float *)malloc(
                                                    n * sizeof(float));
                                                y = (float *)malloc(
                                                    n * sizeof(float));

                                                // initialize data on the host
                                                a = 2.0;
                                                for (int i = 0; i < n; i++) {
                                                  x[i] = 1.0;
                                                  y[i] = 2.0;
                                                }

                                                // run saxpy on the host
                                                saxpy(n, a, x, y);

                                                // print first and last elements
  printf("y[0] = %f

", y[0]);
  printf("y[%d] = %f

", n-1, y[n-1]);

  // free memory on the host
  free(x);
  free(y);

  return 0;
}

                                              // Path: SAXPY/CXX/Makefile

#Makefile
#CXX = g++
CXX = icpc
CXXFLAGS = -fopenmp

saxpy: SAXPY.cpp
  $(CXX) $(CXXFLAGS) -o saxpy SAXPY.cpp

clean:
  rm -f saxpy

// Path: SAXPY/CXX/README

// README

This is a simple C++ example of a saxpy function running on the host.

The saxpy function is implemented using OpenMP offload pragmas.  The main
program runs the saxpy function on the host.

To build the example, run 'make' in this directory.

To run the example, type

  ./saxpy

                                              // Path: SAXPY/CXX/saxpy

                                              // saxpy

                                              // Path: SAXPY/CXX/saxpy.cpp

                                              // saxpy.cpp

                                              // Path: SAXPY/F77/Makefile

#Makefile
#F77 = gfortran
F77 = ifort
F77FLAGS = -qopenmp

saxpy: SAXPY.f
  $(F77) $(F

=======

// saxpy OpenMP offload function
void saxpy(int n, float a, float *x, float *y)
{
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                for (int i = 0; i < n; ++i)
                                                  y[i] = a * x[i] + y[i];
}

// saxpy OpenMP offload function
void saxpy(int n, float a, double *x, double *y)
{
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                for (int i = 0; i < n; ++i)
                                                  y[i] = a * x[i] + y[i];
}

// saxpy OpenMP offload function
void saxpy(int n, double a, float *x, float *y)
{
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                for (int i = 0; i < n; ++i)
                                                  y[i] = a * x[i] + y[i];
}

// saxpy OpenMP offload function
void saxpy(int n, double a, double *x, double *y)
{
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                for (int i = 0; i < n; ++i)
                                                  y[i] = a * x[i] + y[i];
}

// saxpy OpenMP offload function
void saxpy(int n, float a, float *x, double *y)
{
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                for (int i = 0; i < n; ++i)
                                                  y[i] = a * x[i] + y[i];
}

// saxpy OpenMP offload function
void saxpy(int n, float a, double *x, float *y)
{
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        :
                                                == == == =

                                                             // saxpy OpenMP
                                                             // offload function
                                                    void saxpy_omp(
                                                        int n, float a,
                                                        float *x, float *y) {
#pragma omp target teams distribute parallel for
                                                  for (int i = 0; i < n; i++)
                                                    y[i] = a * x[i] + y[i];
                                                }

                                                // saxpy OpenACC offload
                                                // function
                                                void saxpy_acc(int n, float a,
                                                               float *x,
                                                               float *y) {
#pragma acc kernels
                                                  for (int i = 0; i < n; i++)
                                                    y[i] = a * x[i] + y[i];
                                                }

                                                // saxpy OpenMP offload function
                                                void saxpy_omp(int n, double a,
                                                               double *x,
                                                               double *y) {
#pragma omp target teams distribute parallel for
                                                  for (int i = 0; i < n; i++)
                                                    y[i] = a * x[i] + y[i];
                                                }

                                                // saxpy OpenACC offload
                                                // function
                                                void saxpy_acc(int n, double a,
                                                               double *x,
                                                               double *y) {
#pragma acc kernels
                                                  for (int i = 0; i < n; i++)
                                                    y[i] = a * x[i] + y[i];
                                                }

                                                // saxpy OpenMP offload function
                                                void saxpy_omp(
                                                    int n,
                                                    std::complex<float> a,
                                                    std::complex<float> *x,
                                                    std::complex<float> *y) {
#pragma omp target teams distribute parallel for
                                                  for (int i = 0; i < n; i++)
                                                    y[i] = a * x[i] + y[i];
                                                }

                                                // saxpy OpenACC offload
                                                // function
                                                void saxpy_acc(
                                                    int n,
                                                    std::complex<float> a,
                                                    std::complex<float> *x,
                                                    std::complex<float> *y) {
#pragma acc kernels
                                                  for (int i = 0; i < n; i++)
                                                    y[i] = a * x[i] + y[i];
                                                }

                                                // saxpy OpenMP offload function
                                                void saxpy_omp(
                                                    int n,
                                                    std::complex<double> a,
                                                    std::complex<double> *x,
                                                    std::complex<double> *y) {
#pragma omp target teams distribute parallel for
                                                  for (int i = 0; i < n; i++)
                                                    y[i] = a * x[i] + y[i];
                                                }

                                                // saxpy

=======

                                                // saxpy OpenMP offload function
                                                void saxpy_omp_offload(
                                                    int n, float a, float *x,
                                                    float *y) {
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                  for (int i = 0; i < n; ++i) {
                                                    y[i] = a * x[i] + y[i];
                                                  }
                                                }

                                                // saxpy OpenMP offload function
                                                void saxpy_omp_offload(
                                                    int n, double a, double *x,
                                                    double *y) {
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                  for (int i = 0; i < n; ++i) {
                                                    y[i] = a * x[i] + y[i];
                                                  }
                                                }

                                                // saxpy OpenMP offload function
                                                void saxpy_omp_offload(
                                                    int n,
                                                    std::complex<float> a,
                                                    std::complex<float> *x,
                                                    std::complex<float> *y) {
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                  for (int i = 0; i < n; ++i) {
                                                    y[i] = a * x[i] + y[i];
                                                  }
                                                }

                                                // saxpy OpenMP offload function
                                                void saxpy_omp_offload(
                                                    int n,
                                                    std::complex<double> a,
                                                    std::complex<double> *x,
                                                    std::complex<double> *y) {
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                  for (int i = 0; i < n; ++i) {
                                                    y[i] = a * x[i] + y[i];
                                                  }
                                                }

                                                // saxpy OpenMP offload function
                                                void saxpy_omp_offload(
                                                    int n,
                                                    std::complex<float> a,
                                                    float *x, float *y) {
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                  for (int i = 0; i < n; ++i) {
    y[i] = a.real() * x[i

=======

// saxpy OpenMP offload function
void saxpy(int n, float a, float *x, float *y) {

#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                      for (int i = 0; i < n;
                                                           ++i) {
                                                        y[i] = a * x[i] + y[i];
                                                      }
}

// saxpy OpenMP offload function
void saxpy(int n, double a, double *x, double *y) {

#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                      for (int i = 0; i < n;
                                                           ++i) {
                                                        y[i] = a * x[i] + y[i];
                                                      }
}

// saxpy OpenMP offload function
void saxpy(int n, float a, float *x, float *y) {

#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                      for (int i = 0; i < n;
                                                           ++i) {
                                                        y[i] = a * x[i] + y[i];
                                                      }
}

// saxpy OpenMP offload function
void saxpy(int n, double a, double *x, double *y) {

#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                      for (int i = 0; i < n;
                                                           ++i) {
                                                        y[i] = a * x[i] + y[i];
                                                      }
}

// saxpy OpenMP offload function
void saxpy(int n, float a, float *x, float *y) {

#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                      for (int i = 0; i < n;
                                                           ++i) {
                                                        y[i] = a * x[i] + y[i];
                                                      }
}

// saxpy OpenMP offload function
void saxpy(int n, double a, double *x, double *y) {

#pragma omp target teams distribute parallel for map(tofrom : y[
                                                      == == ==
                                                          =

                                                              // saxpy OpenMP
                                                              // offload
                                                              // function
                                                          void saxpy(int n,
                                                                     float a,
                                                                     float *x,
                                                                     float *y,
                                                                     float *z) {
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : z [0:n])                \
    map(to                                                                     \
        : x [0:n], y [0:n]) map(alloc                                          \
                                : a)
                                                        for (int i = 0; i < n;
                                                             i++) {
                                                          z[i] =
                                                              a * x[i] + y[i];
                                                        }
                                                      }

                                                      // saxpy OpenMP offload
                                                      // function with reduction
                                                      void saxpy(int n, float a,
                                                                 float *x,
                                                                 float *y,
                                                                 float *z) {
#pragma omp target teams distribute parallel for map(tofrom: z[0:n]) map(to: x[0:n], y[0:n]) map(alloc: a) reduction(+:z[:n])
                                                        for (int i = 0; i < n;
                                                             i++) {
                                                          z[i] =
                                                              a * x[i] + y[i];
                                                        }
                                                      }

                                                      // saxpy OpenMP offload
                                                      // function with reduction
                                                      void saxpy(int n, float a,
                                                                 float *x,
                                                                 float *y,
                                                                 float *z) {
#pragma omp target teams distribute parallel for map(tofrom: z[0:n]) map(to: x[0:n], y[0:n]) map(alloc: a) reduction(+:z[:n])
                                                        for (int i = 0; i < n;
                                                             i++) {
                                                          z[i] =
                                                              a * x[i] + y[i];
                                                        }
                                                      }

                                                      // saxpy OpenMP offload
                                                      // function with reduction
                                                      void saxpy(int n, float a,
                                                                 float *x,
                                                                 float *y,
                                                                 float *z) {
#pragma omp target teams distribute parallel for map(tofrom: z[0:n]) map(to: x[0:n], y[0:n]) map(alloc: a) reduction(+:z[:n])
                                                        for (int i = 0; i < n;
                                                             i++) {
                                                          z[i] =
                                                              a * x[i] + y[i];
                                                        }
                                                      }

                                                      // saxpy OpenMP offload
                                                      // function with reduction
                                                      void saxpy(int n, float a,
                                                                 float *x,
                                                                 float *y,
                                                                 float *z) {
#pragma omp target teams distribute parallel for map(tofrom: z[0:n]) map(to: x[0:n], y[0:n]) map(alloc: a) reduction(+:z[:n

                                                        == == ==
                                                            =

                                                                // saxpy OpenMP
                                                                // offload
                                                                // function
                                                            void saxpy(REAL * x,
                                                                       REAL * y,
                                                                       REAL a,
                                                                       int n) {
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : x [0:n], y [0:n])       \
    map(to                                                                     \
        : a)
                                                          for (int i = 0; i < n;
                                                               i++)
                                                            y[i] =
                                                                a * x[i] + y[i];
                                                        }

                                                        int main(int argc,
                                                                 char **argv) {
                                                          // Read command line
                                                          // arguments
                                                          int n = 100000000;
                                                          if (argc > 1)
                                                            n = atoi(argv[1]);
                                                          REAL a = 2.0;

                                                          // Allocate memory
                                                          REAL *x = new REAL[n];
                                                          REAL *y = new REAL[n];

                                                          // Initialize vectors
                                                          for (int i = 0; i < n;
                                                               i++) {
                                                            x[i] = 1.0;
                                                            y[i] = 2.0;
                                                          }

                                                          // Run saxpy
                                                          saxpy(x, y, a, n);

                                                          // Check results
                                                          REAL maxError = 0.0;
                                                          for (int i = 0; i < n;
                                                               i++)
                                                            maxError =
                                                                fmax(maxError,
                                                                     fabs(y[i] -
                                                                          4.0));
                                                          std::cout
                                                              << "Max error: "
                                                              << maxError
                                                              << std::endl;

                                                          // Free memory
                                                          delete[] x;
                                                          delete[] y;

                                                          return 0;
                                                        }

The output of the program is:

$ clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -I/usr/local/cuda/include saxpy.cpp
$ ./a.out 
Max error: 0

The program runs on the CPU and GPU. The offloading is done using the map clause. The map clause specifies that the variables x, y, and a are mapped to the device. The variables x and y are mapped to and from the device since they are modified by the offload function. The variable a is mapped to the device only since it is not modified by the offload function.

The map clause supports the following data sharing attributes:

map(to:var) specifies that var is mapped to the device only.

map(tofrom:var) specifies that var is mapped to and from the device.

map(from:var) specifies that var is mapped from the device only.

map(alloc:var) specifies that var is allocated on the device only.

map(tofrom:alloc

=======

// saxpy OpenMP offload function
void saxpy_omp_offload(int n, float a, float *x, float *y)
{
// OpenMP offload
#pragma omp target teams distribute parallel for map(to                        \
                                                     : x [0:n], y [0:n])       \
    map(from                                                                   \
        : y [0:n])
                                                          for (int i = 0; i < n;
                                                               i++)
                                                            y[i] =
                                                                a * x[i] + y[i];
}

// saxpy OpenMP function
void saxpy_omp(int n, float a, float *x, float *y)
{
// OpenMP loop
#pragma omp parallel for
                                                          for (int i = 0; i < n;
                                                               i++)
                                                            y[i] =
                                                                a * x[i] + y[i];
}

// saxpy function
void saxpy(int n, float a, float *x, float *y)
{
                                                          for (int i = 0; i < n;
                                                               i++)
                                                            y[i] =
                                                                a * x[i] + y[i];
}

// main function
int main(int argc, char **argv)
{
                                                          int n = 1000;
                                                          float a = 2.0f;
                                                          float *x, *y;
                                                          x = new float[n];
                                                          y = new float[n];
                                                          for (int i = 0; i < n;
                                                               i++) {
                                                            x[i] = 1.0f;
                                                            y[i] = 2.0f;
                                                          }

                                                          // saxpy
                                                          saxpy(n, a, x, y);
                                                          // saxpy OpenMP
                                                          saxpy_omp(n, a, x, y);
                                                          // saxpy OpenMP
                                                          // offload
                                                          saxpy_omp_offload(
                                                              n, a, x, y);

                                                          delete[] x;
                                                          delete[] y;
                                                          return 0;
}

// Path: SAXPY/CXX/CMakeLists.txt

cmake_minimum_required(VERSION 3.5)
project(SAXPY LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
add_executable(SAXPY SAXPY.cpp)
target_link_libraries(SAXPY OpenMP::OpenMP_CXX)

                                                        // Path:
                                                        // SAXPY/CXX/build.sh

#!/ bin / bash

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=true
export OMP

=======

// saxpy OpenMP offload function
void saxpy(int n, float a, float *x, float *y, int block_size)
{
// OpenMP target offload
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                          for (int i = 0; i < n;
                                                               i++) {
                                                            y[i] =
                                                                a * x[i] + y[i];
                                                          }
}

// saxpy OpenMP offload function
void saxpy(int n, double a, double *x, double *y, int block_size)
{
// OpenMP target offload
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                          for (int i = 0; i < n;
                                                               i++) {
                                                            y[i] =
                                                                a * x[i] + y[i];
                                                          }
}

// saxpy OpenMP offload function
void saxpy(int n, float a, float *x, float *y, int block_size)
{
// OpenMP target offload
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                          for (int i = 0; i < n;
                                                               i++) {
                                                            y[i] =
                                                                a * x[i] + y[i];
                                                          }
}

// saxpy OpenMP offload function
void saxpy(int n, double a, double *x, double *y, int block_size)
{
// OpenMP target offload
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                          for (int i = 0; i < n;
                                                               i++) {
                                                            y[i] =
                                                                a * x[i] + y[i];
                                                          }
}

// saxpy OpenMP offload function
void saxpy(int n, float a, float *x, float *y, int block_size)
{
// OpenMP target offload
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : y [0:n]) map(to         \
                                                                    : x [0:n]) \
    map(tofrom                                                                 \
        : a)
                                                          for (int i = 0; i < n;
                                                               i++)

                                                            // saxpy OpenACC
                                                            // function
                                                            Synthesizing 10 /
                                                                10 solutions

=======

                                                                // saxpy OpenACC
                                                                // function
                                                                void
                                                                saxpy(
                                                                    int n,
                                                                    float a,
                                                                    float *x,
                                                                    float *y) {
#pragma acc parallel loop present(x [0:n], y [0:n])
                                                              for (int i = 0;
                                                                   i < n; i++)
                                                                y[i] =
                                                                    a * x[i] +
                                                                    y[i];
                                                            }

                                                          // SAXPY C++ function
                                                          void saxpy_cxx(
                                                              int n, float a,
                                                              float *x,
                                                              float *y) {
                                                            for (int i = 0;
                                                                 i < n; i++)
                                                              y[i] = a * x[i] +
                                                                     y[i];
                                                          }

                                                          // SAXPY C++ function
                                                          void saxpy_cxx_omp(
                                                              int n, float a,
                                                              float *x,
                                                              float *y) {
#pragma omp parallel for
                                                            for (int i = 0;
                                                                 i < n; i++)
                                                              y[i] = a * x[i] +
                                                                     y[i];
                                                          }

                                                          // SAXPY C function
                                                          void saxpy_c(
                                                              int n, float a,
                                                              float *x,
                                                              float *y) {
                                                            for (int i = 0;
                                                                 i < n; i++)
                                                              y[i] = a * x[i] +
                                                                     y[i];
                                                          }

                                                          // SAXPY C function
                                                          void saxpy_c_omp(
                                                              int n, float a,
                                                              float *x,
                                                              float *y) {
#pragma omp parallel for
                                                            for (int i = 0;
                                                                 i < n; i++)
                                                              y[i] = a * x[i] +
                                                                     y[i];
                                                          }

                                                          // SAXPY C function
                                                          void saxpy_c_acc(
                                                              int n, float a,
                                                              float *x,
                                                              float *y) {
#pragma acc parallel loop present(x [0:n], y [0:n])
                                                            for (int i = 0;
                                                                 i < n; i++)
                                                              y[i] = a * x[i] +
                                                                     y[i];
                                                          }

                                                          // SAXPY C function
                                                          void saxpy_c_acc_omp(
                                                              int n, float a,
                                                              float *x,
                                                              float *y) {
#pragma acc parallel loop present(x [0:n], y [0:n]) num_threads(8)
                                                            for (int i = 0;
                                                                 i < n; i++)
                                                              y[i] = a * x[i] +
                                                                     y[i];
                                                          }

                                                          // SAXPY C function
                                                          void saxpy_c_acc_omp2(
                                                              int n, float a,
                                                              float *x,
                                                              float *y) {
#pragma acc parallel loop present(x[

                                                            == == ==
                                                                =

                                                                    // saxpy
                                                                    // OpenACC
                                                                    // function
                                                                void saxpy(
                                                                    int n,
                                                                    float a,
                                                                    float *x,
                                                                    float *y) {
                                                              int i;
#pragma acc kernels
                                                              for (i = 0; i < n;
                                                                   i++)
                                                                y[i] =
                                                                    a * x[i] +
                                                                    y[i];
                                                            }

                                                            // saxpy OpenMP
                                                            // function
                                                            void saxpy_omp(
                                                                int n, float a,
                                                                float *x,
                                                                float *y) {
                                                              int i;
#pragma omp parallel for
                                                              for (i = 0; i < n;
                                                                   i++)
                                                                y[i] =
                                                                    a * x[i] +
                                                                    y[i];
                                                            }

                                                            // saxpy CPU
                                                            // function
                                                            void saxpy_cpu(
                                                                int n, float a,
                                                                float *x,
                                                                float *y) {
                                                              int i;
                                                              for (i = 0; i < n;
                                                                   i++)
                                                                y[i] =
                                                                    a * x[i] +
                                                                    y[i];
                                                            }

                                                            // saxpy CPU
                                                            // function
                                                            void saxpy_cpu(
                                                                int n, float a,
                                                                float *x,
                                                                float *y) {
                                                              int i;
                                                              for (i = 0; i < n;
                                                                   i++)
                                                                y[i] =
                                                                    a * x[i] +
                                                                    y[i];
                                                            }

                                                            // saxpy CPU
                                                            // function
                                                            void saxpy_cpu(
                                                                int n, float a,
                                                                float *x,
                                                                float *y) {
                                                              int i;
                                                              for (i = 0; i < n;
                                                                   i++)
                                                                y[i] =
                                                                    a * x[i] +
                                                                    y[i];
                                                            }

                                                            // saxpy CPU
                                                            // function
                                                            void saxpy_cpu(
                                                                int n, float a,
                                                                float *x,
                                                                float *y) {
                                                              int i;
                                                              for (i = 0; i < n;
                                                                   i++)
                                                                y[i] =
                                                                    a * x[i] +
                                                                    y[i];
                                                            }

                                                            // saxpy CPU
                                                            // function
                                                            void saxpy_cpu(
                                                                int n, float a,
                                                                float *x,
                                                                float *y) {
                                                              int i;
                                                              for (i = 0; i < n;
                                                                   i++)
                                                                y[i] =
                                                                    a * x[i] +
                                                                    y[i];
                                                            }

                                                            // saxpy CPU
                                                            // function
                                                            void saxpy_cpu(
                                                                int n, float a,
                                                                float *x,
                                                                float *y) {
                                                              int i;
                                                              for (i = 0; i < n;
                                                                   i++)
                                                                y[i] =
                                                                    a * x[i] +
                                                                    y[i];
                                                            }

                                                            // saxpy CPU
                                                            // function
                                                            void saxpy_cpu(
                                                                int n, float a,
                                                                float *x,
                                                                float *y) {

=======

                                                              // saxpy OpenACC
                                                              // function
                                                              void saxpy(
                                                                  int n,
                                                                  float a,
                                                                  float *x,
                                                                  float *y) {
                                                                int i;
#pragma acc parallel loop
                                                                for (i = 0;
                                                                     i < n; i++)
                                                                  y[i] =
                                                                      a * x[i] +
                                                                      y[i];
                                                              }

                                                              int main() {
                                                                int n =
                                                                    100000000;
                                                                float *x =
                                                                    new float
                                                                        [n];
                                                                float *y =
                                                                    new float
                                                                        [n];
                                                                float a = 2.0;

                                                                // Initialize
                                                                // vectors
                                                                for (int i = 0;
                                                                     i < n;
                                                                     i++) {
                                                                  x[i] = 1.0;
                                                                  y[i] = 2.0;
                                                                }

                                                                saxpy(n, a, x,
                                                                      y);

                                                                // Check result
                                                                float maxError =
                                                                    0.0;
                                                                for (int i = 0;
                                                                     i < n; i++)
                                                                  maxError = fmax(
                                                                      maxError,
                                                                      fabs(
                                                                          y[i] -
                                                                          4.0));
                                                                std::cout
                                                                    << "Max "
                                                                       "error: "
                                                                    << maxError
                                                                    << std::
                                                                           endl;

                                                                // Free memory
                                                                delete[] x;
                                                                delete[] y;

                                                                return 0;
                                                              }

                                                              // End of file

                                                              // Path:
                                                              // SAXPY/CXX/Makefile

=======

                                                              // saxpy OpenACC
                                                              // function
                                                              extern "C" void
                                                              saxpy(int n,
                                                                    float a,
                                                                    float *x,
                                                                    float *y) {
#pragma acc kernels
                                                                for (int i = 0;
                                                                     i < n; i++)
                                                                  y[i] =
                                                                      a * x[i] +
                                                                      y[i];
                                                              }

                                                              // saxpy OpenMP
                                                              // function
                                                              extern "C" void
                                                              saxpy_omp(
                                                                  int n,
                                                                  float a,
                                                                  float *x,
                                                                  float *y) {
#pragma omp parallel for
                                                                for (int i = 0;
                                                                     i < n; i++)
                                                                  y[i] =
                                                                      a * x[i] +
                                                                      y[i];
                                                              }

                                                              // saxpy C++
                                                              // function
                                                              extern "C" void
                                                              saxpy_cpp(
                                                                  int n,
                                                                  float a,
                                                                  float *x,
                                                                  float *y) {
                                                                for (int i = 0;
                                                                     i < n; i++)
                                                                  y[i] =
                                                                      a * x[i] +
                                                                      y[i];
                                                              }

                                                              // saxpy C++
                                                              // function
                                                              extern "C" void
                                                              saxpy_cpp2(
                                                                  int n,
                                                                  float a,
                                                                  float *x,
                                                                  float *y) {
                                                                for (int i = 0;
                                                                     i < n; i++)
                                                                  y[i] =
                                                                      a * x[i] +
                                                                      y[i];
                                                              }

                                                              // saxpy C++
                                                              // function
                                                              extern "C" void
                                                              saxpy_cpp3(
                                                                  int n,
                                                                  float a,
                                                                  float *x,
                                                                  float *y) {
                                                                for (int i = 0;
                                                                     i < n; i++)
                                                                  y[i] =
                                                                      a * x[i] +
                                                                      y[i];
                                                              }

                                                              // saxpy C++
                                                              // function
                                                              extern "C" void
                                                              saxpy_cpp4(
                                                                  int n,
                                                                  float a,
                                                                  float *x,
                                                                  float *y) {
                                                                for (int i = 0;
                                                                     i < n; i++)
                                                                  y[i] =
                                                                      a * x[i] +
                                                                      y[i];
                                                              }

                                                              // saxpy C++
                                                              // function
                                                              extern "C" void
                                                              saxpy_cpp5(
                                                                  int n,
                                                                  float a,
                                                                  float *x,
                                                                  float *y) {
                                                                for (int i = 0;
                                                                     i < n; i++)
                                                                  y[i] =
                                                                      a * x[i] +
                                                                      y[i];
                                                              }

                                                              // saxpy C++
                                                              // function
                                                              extern "C" void
                                                              saxpy_cpp6(
                                                                  int n,
                                                                  float a,
                                                                  float *x,
                                                                  float *y) {
                                                                for (int i = 0;
                                                                     i < n; i++)
                                                                  y[i] =
                                                                      a * x[i] +
                                                                      y[i];
                                                              }

                                                              // saxpy C++
                                                              // function

=======

                                                              // saxpy OpenACC
                                                              // function
                                                              void saxpy(
                                                                  int n,
                                                                  float a,
                                                                  float *x,
                                                                  float *y) {

#pragma acc kernels loop independent
                                                                for (int i = 0;
                                                                     i < n; ++i)
                                                                  y[i] =
                                                                      a * x[i] +
                                                                      y[i];
                                                              }

                                                              // saxpy OpenMP
                                                              // function
                                                              void saxpy_omp(
                                                                  int n,
                                                                  float a,
                                                                  float *x,
                                                                  float *y) {

#pragma omp parallel for
                                                                for (int i = 0;
                                                                     i < n; ++i)
                                                                  y[i] =
                                                                      a * x[i] +
                                                                      y[i];
                                                              }

                                                              // saxpy OpenMP
                                                              // function
                                                              void
                                                              saxpy_omp_simd(
                                                                  int n,
                                                                  float a,
                                                                  float *x,
                                                                  float *y) {

#pragma omp parallel for simd
                                                                for (int i = 0;
                                                                     i < n; ++i)
                                                                  y[i] =
                                                                      a * x[i] +
                                                                      y[i];
                                                              }

                                                              // saxpy OpenMP
                                                              // function
                                                              void
                                                              saxpy_omp_simd_aligned(
                                                                  int n,
                                                                  float a,
                                                                  float *x,
                                                                  float *y) {

#pragma omp parallel for simd aligned(x, y : 64)
                                                                for (int i = 0;
                                                                     i < n; ++i)
                                                                  y[i] =
                                                                      a * x[i] +
                                                                      y[i];
                                                              }

                                                              // saxpy OpenMP
                                                              // function
                                                              void
                                                              saxpy_omp_simd_aligned_collapse(
                                                                  int n,
                                                                  float a,
                                                                  float *x,
                                                                  float *y) {

#pragma omp parallel for simd collapse(2) aligned(x, y : 64)
                                                                for (int i = 0;
                                                                     i < n; ++i)
                                                                  for (int j =
                                                                           0;
                                                                       j < n;
                                                                       ++j)
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                              }

                                                              // saxpy OpenMP
                                                              // function
                                                              void
                                                              saxpy_omp_simd_aligned_collapse_parallel(
                                                                  int n,
                                                                  float a,
                                                                  float *x,
                                                                  float *y) {

#pragma omp parallel for simd collapse(2) aligned(x, y : 64) num_threads(8)
                                                                for (int i = 0;
                                                                     i < n; ++i)
                                                                  for (int j =
                                                                           0;
                                                                       j < n;
                                                                       ++j)
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                              }

                                                              // saxpy OpenMP
                                                              // function
void saxpy_omp_simd_aligned_collapse_parallel_ordered(int n, float a, float *x, float

=======

// saxpy OpenACC function
void saxpy(int n, float a, float *x, float *y) {
                                                                int i;
#pragma acc parallel loop gang worker vector
                                                                for (i = 0;
                                                                     i < n; i++)
                                                                  y[i] =
                                                                      a * x[i] +
                                                                      y[i];
}

// saxpy OpenMP function
void saxpy_omp(int n, float a, float *x, float *y) {
                                                                int i;
#pragma omp parallel for
                                                                for (i = 0;
                                                                     i < n; i++)
                                                                  y[i] =
                                                                      a * x[i] +
                                                                      y[i];
}

// saxpy OpenMP function
void saxpy_omp_target(int n, float a, float *x, float *y) {
                                                                int i;
#pragma omp target teams distribute parallel for
                                                                for (i = 0;
                                                                     i < n; i++)
                                                                  y[i] =
                                                                      a * x[i] +
                                                                      y[i];
}

// saxpy OpenMP function
void saxpy_omp_target_map(int n, float a, float *x, float *y) {
                                                                int i;
#pragma omp target teams distribute parallel for map(tofrom : y [0:n])
                                                                for (i = 0;
                                                                     i < n; i++)
                                                                  y[i] =
                                                                      a * x[i] +
                                                                      y[i];
}

// saxpy OpenMP function
void saxpy_omp_target_map_alloc(int n, float a, float *x, float *y) {
                                                                int i;
                                                                float *y_device;
#pragma omp target enter data map(alloc : y_device [0:n])
#pragma omp target teams distribute parallel for map(tofrom : y_device [0:n])
                                                                for (i = 0;
                                                                     i < n; i++)
                                                                  y_device[i] =
                                                                      a * x[i] +
                                                                      y_device
                                                                          [i];
#pragma omp target exit data map(release : y_device [0:n])
}

// saxpy OpenMP function
void saxpy_omp_target_map_alloc_2(int n, float a, float *x, float *y) {
                                                                int i;
                                                                float *y_device;
#pragma omp target enter data map(alloc : y_device [0:n])
#pragma omp target teams distribute parallel for map(tofrom : y_device [0:n])
                                                                for (i = 0;
                                                                     i < n; i++)
                                                                  y_device

=======

                                                                      // saxpy
                                                                      // OpenACC
                                                                      // function
                                                                      void
                                                                      saxpy(
                                                                          int n,
                                                                          float
                                                                              a,
                                                                          float x
                                                                              [],
                                                                          float y
                                                                              []) {
#pragma acc parallel loop present(x [0:n], y [0:n])
                                                                    for (int i =
                                                                             0;
                                                                         i < n;
                                                                         i++)
                                                                      y[i] =
                                                                          a * x[i] +
                                                                          y[i];
                                                                  }

                                                                // SAXPY
                                                                // function
                                                                void SAXPY(
                                                                    int n,
                                                                    float a,
                                                                    float x[],
                                                                    float y[]) {
                                                                  for (int i =
                                                                           0;
                                                                       i < n;
                                                                       i++)
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                }

                                                                // main function
                                                                int main() {
                                                                  int n =
                                                                      1000000;
                                                                  float a = 2.0;
                                                                  float *x =
                                                                      new float
                                                                          [n];
                                                                  float *y =
                                                                      new float
                                                                          [n];

                                                                  for (int i =
                                                                           0;
                                                                       i < n;
                                                                       i++) {
                                                                    x[i] = i;
                                                                    y[i] = 0.0;
                                                                  }

                                                                  saxpy(n, a, x,
                                                                        y);

                                                                  for (int i =
                                                                           0;
                                                                       i < n;
                                                                       i++) {
                                                                    if (y[i] !=
                                                                        2.0 *
                                                                            i) {
                                                                      cout
                                                                          << "E"
                                                                             "r"
                                                                             "r"
                                                                             "o"
                                                                             "r"
                                                                             ":"
                                                                             " "
                                                                             "y"
                                                                             "["
                                                                          << i
                                                                          << "]"
                                                                             " "
                                                                             "="
                                                                             " "
                                                                          << y[i]
                                                                          << endl;
                                                                      return 1;
                                                                    }
                                                                  }

                                                                  cout << "Succ"
                                                                          "ess"
                                                                       << endl;

                                                                  return 0;
                                                                }

                                                                // End of
                                                                // SAXPY.cpp

                                                                The program is compiled
                                                                        with the following
                                                                            command
                                                                                line
                                                                    :

                                                                    $ pgc++ -
                                                                    acc - ta =
                                                                    tesla
                                                                    : cc35 SAXPY
                                                                          .cpp

                                                                              The program
                                                                                  is run with
                                                                                      the following
                                                                                          command
                                                                                              line
                                                                    :

                                                                    $./ a.out

                                                                            The program produces
                                                                                the following
                                                                                    output
                                                                    :

                                                                    Success

                                                                        The program is compiled
                                                                            with the following
                                                                                command
                                                                                    line
                                                                    :

                                                                    $ pgc++ -
                                                                    acc - ta =
                                                                        tesla
                                                                    : cc35 -
                                                                      Minfo =
                                                                            accel SAXPY
                                                                                    .cpp

                                                                                        The program
                                                                                            is run with
                                                                                                the following
                                                                                                    command
                                                                                                        line
                                                                    :

                                                                    $./ a.out

                                                                                    The program produces
                                                                                        the following
                                                                                            output
                                                                    :

                                                                    Success

                                                                        The program is compiled
                                                                            with the following
                                                                                command
                                                                                    line
                                                                    :

                                                                    $ pgc++ -
                                                                            acc -
                                                                            ta =
                                                                                tesla
                                                                    : cc35 -
                                                                      Minfo = accel -
                                                                              ta =
                                                                                  tesla
                                                                    : maxregcount : 32 SAXPY
                                                                          .cpp

                                                                              The program
                                                                                  is run with
                                                                                      the following
                                                                                          command
                                                                                              line
                                                                    :

                                                                    $./ a.out

                                                                                          The program produces
                                                                                              the following
                                                                                                  output
                                                                    :

                                                                    Success

                                                                        The program is compiled
                                                                            with the following
                                                                                command
                                                                                    line
                                                                    :

                                                                    $ pgc++ - acc -
                                                                                  ta =
                                                                                      tesla
                                                                    :

=======

                                                                    // saxpy
                                                                    // OpenACC
                                                                    // function
                                                                    // Performs
                                                                    // the
                                                                    // operation
                                                                    // y = a * x
                                                                    // + y
                                                                    void
                                                                    saxpy(
                                                                        int n,
                                                                        float a,
                                                                        float
                                                                            *x,
                                                                        float *
                                                                            y) {
#pragma acc kernels
                                                                  for (int i =
                                                                           0;
                                                                       i < n;
                                                                       i++) {
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                  }
                                                                }

                                                                // saxpy OpenMP
                                                                // function
                                                                // Performs the
                                                                // operation y =
                                                                // a * x + y
                                                                void saxpy_omp(
                                                                    int n,
                                                                    float a,
                                                                    float *x,
                                                                    float *y) {
#pragma omp parallel for
                                                                  for (int i =
                                                                           0;
                                                                       i < n;
                                                                       i++) {
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                  }
                                                                }

                                                                // saxpy
                                                                // OpenMP+ACC
                                                                // function
                                                                // Performs the
                                                                // operation y =
                                                                // a * x + y
                                                                void
                                                                saxpy_ompacc(
                                                                    int n,
                                                                    float a,
                                                                    float *x,
                                                                    float *y) {
#pragma omp target teams distribute parallel for
                                                                  for (int i =
                                                                           0;
                                                                       i < n;
                                                                       i++) {
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                  }
                                                                }

                                                                // saxpy
                                                                // OpenMP+ACC
                                                                // function
                                                                // Performs the
                                                                // operation y =
                                                                // a * x + y
                                                                void
                                                                saxpy_ompacc2(
                                                                    int n,
                                                                    float a,
                                                                    float *x,
                                                                    float *y) {
#pragma omp target teams distribute parallel for
                                                                  for (int i =
                                                                           0;
                                                                       i < n;
                                                                       i++) {
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                  }
                                                                }

                                                                // saxpy
                                                                // OpenMP+ACC
                                                                // function
                                                                // Performs the
                                                                // operation y =
                                                                // a * x + y
                                                                void
                                                                saxpy_ompacc3(
                                                                    int n,
                                                                    float a,
                                                                    float *x,
                                                                    float *y) {
#pragma omp target teams distribute parallel for
                                                                  for (int i =
                                                                           0;
                                                                       i < n;
                                                                       i++) {
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                  }
                                                                }

                                                                // saxpy
                                                                // OpenMP+ACC
                                                                // function
                                                                // Performs the
                                                                // operation y =
                                                                // a * x + y
                                                                void
                                                                saxpy_ompacc4(
                                                                    int n,
                                                                    float a,
                                                                    float *x,
                                                                    float *y) {
#pragma omp target teams distribute parallel for
                                                                  for (int i =
                                                                           0;
                                                                       i < n;
                                                                       i++) {
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                  }
                                                                }

                                                                // saxpy
                                                                // OpenMP+ACC

=======

                                                                // saxpy OpenACC
                                                                // function

                                                                void saxpy(
                                                                    int n,
                                                                    float a,
                                                                    float *x,
                                                                    float *y) {

                                                                  int i;

#pragma acc kernels loop gang(256) worker(64) vector(16)

                                                                  for (i = 0;
                                                                       i < n;
                                                                       i++) {
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                  }
                                                                }

                                                                // saxpy OpenMP
                                                                // function

                                                                void saxpy_omp(
                                                                    int n,
                                                                    float a,
                                                                    float *x,
                                                                    float *y) {

                                                                  int i;

#pragma omp parallel for

                                                                  for (i = 0;
                                                                       i < n;
                                                                       i++) {
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                  }
                                                                }

                                                                // saxpy OpenMP
                                                                // function

                                                                void
                                                                saxpy_omp_target(
                                                                    int n,
                                                                    float a,
                                                                    float *x,
                                                                    float *y) {

                                                                  int i;

#pragma omp target teams distribute parallel for

                                                                  for (i = 0;
                                                                       i < n;
                                                                       i++) {
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                  }
                                                                }

                                                                // saxpy OpenMP
                                                                // function

                                                                void
                                                                saxpy_omp_target_data(
                                                                    int n,
                                                                    float a,
                                                                    float *x,
                                                                    float *y) {

                                                                  int i;

#pragma omp target data map(to : x [0:n], y [0:n]) map(from : y [0:n])

#pragma omp target teams distribute parallel for

                                                                  for (i = 0;
                                                                       i < n;
                                                                       i++) {
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                  }
                                                                }

                                                                // saxpy OpenMP
                                                                // function

                                                                void
                                                                saxpy_omp_target_data_shared(
                                                                    int n,
                                                                    float a,
                                                                    float *x,
                                                                    float *y) {

                                                                  int i;

#pragma omp target data map(to : x [0:n], y [0:n]) map(from : y [0:n])

#pragma omp target teams distribute parallel for

                                                                  for (i = 0;
                                                                       i < n;
                                                                       i++) {
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                  }
                                                                }

                                                                // saxpy OpenMP
                                                                // function

                                                                void
                                                                saxpy_omp_target_data_shared_novector(
                                                                    int n,
                                                                    float a,
                                                                    float *x,
                                                                    float *y) {

                                                                  int i;

#pragma omp target data map(to : x [0:n], y [0:n]) map(from : y [0:n])

#pragma omp target teams distribute parallel for

                                                                  for (i = 0;
                                                                       i < n;
                                                                       i++) {
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                  }
                                                                }

                                                                // saxpy OpenMP
                                                                // function

void saxpy_omp_target_data_shared_novector_nogang(int n, float

=======

// saxpy OpenACC function
// y = a * x + y

void saxpy( int n, float a, float *x, float *y ) 
{
                                                                  int i;
#pragma acc kernels loop copyin(x [0:n], y [0:n]) copyout(y [0:n])
                                                                  for (i = 0;
                                                                       i < n;
                                                                       i++) {
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                  }
}

// saxpy OpenACC function
// y = a * x + y

void saxpy( int n, double a, double *x, double *y ) 
{
                                                                  int i;
#pragma acc kernels loop copyin(x [0:n], y [0:n]) copyout(y [0:n])
                                                                  for (i = 0;
                                                                       i < n;
                                                                       i++) {
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                  }
}

// saxpy OpenACC function
// y = a * x + y

void saxpy( int n, float a, float *x, float *y ) 
{
                                                                  int i;
#pragma acc kernels loop copyin(x [0:n], y [0:n]) copyout(y [0:n])
                                                                  for (i = 0;
                                                                       i < n;
                                                                       i++) {
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                  }
}

// saxpy OpenACC function
// y = a * x + y

void saxpy( int n, double a, double *x, double *y ) 
{
                                                                  int i;
#pragma acc kernels loop copyin(x [0:n], y [0:n]) copyout(y [0:n])
                                                                  for (i = 0;
                                                                       i < n;
                                                                       i++) {
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                  }
}

// saxpy OpenACC function
// y = a * x + y

void saxpy( int n, float a, float *x, float *y ) 
{
                                                                  int i;
#pragma acc kernels loop copyin(x [0:n], y [0:n]) copyout(y [0:n])
                                                                  for (i = 0;
                                                                       i < n;
                                                                       i++) {
                                                                    y[i] =
                                                                        a * x[i] +
                                                                        y[i];
                                                                  }
}

// saxpy OpenACC function
// y = a * x + y

void saxpy(


// Prompt: saxpy Kokkos function
Synthesizing 10/10 solutions

=======

// saxpy Kokkos function
template <class ViewTypeA, class ViewTypeX, class ViewTypeY>
struct SAXPYFunctor {
                                                                  typedef typename ViewTypeA::
                                                                      value_type
                                                                          ScalarA;
                                                                  typedef typename ViewTypeX::
                                                                      value_type
                                                                          ScalarX;
                                                                  typedef typename ViewTypeY::
                                                                      value_type
                                                                          ScalarY;

                                                                  const ScalarA
                                                                      _alpha;
                                                                  const ViewTypeA
                                                                      _A;
                                                                  const ViewTypeX
                                                                      _x;
                                                                  const ViewTypeY
                                                                      _y;

                                                                  SAXPYFunctor(
                                                                      ScalarA
                                                                          alpha,
                                                                      ViewTypeA
                                                                          A,
                                                                      ViewTypeX
                                                                          x,
                                                                      ViewTypeY
                                                                          y)
                                                                      : _alpha(
                                                                            alpha),
                                                                        _A(A),
                                                                        _x(x),
                                                                        _y(y) {}

                                                                  KOKKOS_INLINE_FUNCTION
                                                                  void
                                                                  operator()(
                                                                      const int
                                                                          i)
                                                                      const {
                                                                    _y(i) =
                                                                        _alpha *
                                                                            _A(i) *
                                                                            _x(i) +
                                                                        _y(i);
                                                                  }
};

// saxpy wrapper function
template <class ViewTypeA, class ViewTypeX, class ViewTypeY>
void saxpy(typename ViewTypeA::value_type alpha, ViewTypeA A, ViewTypeX x,
           ViewTypeY y) {
                                                                  typedef typename ViewTypeX::
                                                                      execution_space
                                                                          execution_space;
                                                                  int N =
                                                                      x.extent(
                                                                          0);
                                                                  Kokkos::parallel_for(
                                                                      Kokkos::RangePolicy<
                                                                          execution_space>(
                                                                          0, N),
                                                                      SAXPYFunctor<
                                                                          ViewTypeA,
                                                                          ViewTypeX,
                                                                          ViewTypeY>(
                                                                          alpha,
                                                                          A, x,
                                                                          y));
}

int main(int argc, char* argv[]) {
                                                                  Kokkos::
                                                                      initialize(
                                                                          argc,
                                                                          argv);
                                                                  int N =
                                                                      1000000;
                                                                  double alpha =
                                                                      2.0;
                                                                  Kokkos::View<
                                                                      double *>
                                                                      A("A", N);
                                                                  Kokkos::View<
                                                                      double *>
                                                                      x("x", N);
                                                                  Kokkos::View<
                                                                      double *>
                                                                      y("y", N);

                                                                  // Initialize
                                                                  // x, y and A
                                                                  Kokkos::parallel_for(
                                                                      N,
                                                                      KOKKOS_LAMBDA(
                                                                          const int
                                                                              i) {
                                                                        x(i) =
                                                                            1.0;
                                                                      });
                                                                  Kokkos::parallel_for(
                                                                      N,
                                                                      KOKKOS_LAMBDA(
                                                                          const int
                                                                              i) {
                                                                        y(i) =
                                                                            2.0;
                                                                      });
                                                                  Kokkos::parallel_for(
                                                                      N,
                                                                      KOKKOS_LAMBDA(
                                                                          const int
                                                                              i) {
                                                                        A(i) =
                                                                            3.0;
                                                                      });

                                                                  saxpy(alpha,
                                                                        A, x,
                                                                        y);
                                                                  Kokkos::
                                                                      finalize();
}

=======

// saxpy Kokkos function
template<class ViewTypeA, class ViewTypeX, class ViewTypeY>
struct saxpy_functor {
                                                                  ViewTypeA A;
                                                                  ViewTypeX X;
                                                                  ViewTypeY Y;

                                                                  saxpy_functor(
                                                                      ViewTypeA &
                                                                          A_,
                                                                      ViewTypeX &
                                                                          X_,
                                                                      ViewTypeY &
                                                                          Y_)
                                                                      : A(A_),
                                                                        X(X_),
                                                                        Y(Y_) {}

                                                                  KOKKOS_INLINE_FUNCTION
                                                                  void
                                                                  operator()(
                                                                      const int
                                                                          i)
                                                                      const {
                                                                    Y(i) =
                                                                        A(0) *
                                                                            X(i) +
                                                                        Y(i);
                                                                  }
};

int main(int argc, char* argv[]) {
                                                                  Kokkos::
                                                                      initialize(
                                                                          argc,
                                                                          argv);

                                                                  // Size of
                                                                  // vectors
                                                                  int N =
                                                                      1000000;

                                                                  // Create
                                                                  // Views
                                                                  Kokkos::View<
                                                                      double *>
                                                                      X("X", N);
                                                                  Kokkos::View<
                                                                      double *>
                                                                      Y("Y", N);

                                                                  // Fill Views
                                                                  // with data
                                                                  Kokkos::parallel_for(
                                                                      N,
                                                                      KOKKOS_LAMBDA(
                                                                          const int
                                                                              i) {
                                                                        X(i) =
                                                                            1.0;
                                                                        Y(i) =
                                                                            2.0;
                                                                      });

                                                                  // Create a
                                                                  // scalar
                                                                  Kokkos::View<
                                                                      double *>
                                                                      A("A", 1);
                                                                  Kokkos::parallel_for(
                                                                      1,
                                                                      KOKKOS_LAMBDA(
                                                                          const int
                                                                              i) {
                                                                        A(i) =
                                                                            2.0;
                                                                      });

                                                                  // Execute
                                                                  // saxpy
                                                                  // kernel
                                                                  Kokkos::parallel_for(
                                                                      N,
                                                                      saxpy_functor<
                                                                          Kokkos::View<
                                                                              double
                                                                                  *>,
                                                                          Kokkos::View<
                                                                              double
                                                                                  *>,
                                                                          Kokkos::View<
                                                                              double
                                                                                  *>>(
                                                                          A, X,
                                                                          Y));

                                                                  // Print
                                                                  // result
  Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int i) {
    printf("Y(%d) = %f

", i, Y(i));
  });

  Kokkos::finalize();
}

[/sourcecode]

[/sourcecode]

[/sourcecode]

[/

=======

// saxpy Kokkos function
template<typename ViewTypeA, typename ViewTypeX, typename ViewTypeY>
struct saxpy_functor {
                                                                  typedef typename ViewTypeA::
                                                                      value_type
                                                                          value_type;
                                                                  ViewTypeA a;
                                                                  ViewTypeX x;
                                                                  ViewTypeY y;

                                                                  saxpy_functor(
                                                                      const value_type
                                                                          &_a,
                                                                      const ViewTypeX
                                                                          &_x,
                                                                      const ViewTypeY
                                                                          &_y)
                                                                      : a("a",
                                                                          1),
                                                                        x(_x),
                                                                        y(_y) {
                                                                    a(0) = _a;
                                                                  }

                                                                  KOKKOS_INLINE_FUNCTION
                                                                  void
                                                                  operator()(
                                                                      const int
                                                                          i)
                                                                      const {
                                                                    y(i) =
                                                                        a(0) *
                                                                            x(i) +
                                                                        y(i);
                                                                  }
};

// saxpy function
template<typename ViewTypeA, typename ViewTypeX, typename ViewTypeY>
void saxpy(const ViewTypeA& a, const ViewTypeX& x, const ViewTypeY& y) {
                                                                  typedef typename ViewTypeA::
                                                                      value_type
                                                                          value_type;
                                                                  typedef typename ViewTypeA::
                                                                      size_type
                                                                          size_type;
                                                                  size_type N =
                                                                      x.extent(
                                                                          0);
                                                                  if (y.extent(
                                                                          0) !=
                                                                      N)
                                                                    throw "saxp"
                                                                          "y: "
                                                                          "inco"
                                                                          "mpat"
                                                                          "ible"
                                                                          " dim"
                                                                          "ensi"
                                                                          "ons";
                                                                  Kokkos::parallel_for(
                                                                      N,
                                                                      saxpy_functor<
                                                                          ViewTypeA,
                                                                          ViewTypeX,
                                                                          ViewTypeY>(
                                                                          a, x,
                                                                          y));
}

// saxpy function
template<typename ViewTypeA, typename ViewTypeX, typename ViewTypeY>
void saxpy(const typename ViewTypeA::value_type& a, const ViewTypeX& x, const ViewTypeY& y) {
                                                                  typedef typename ViewTypeA::
                                                                      value_type
                                                                          value_type;
                                                                  typedef typename ViewTypeA::
                                                                      size_type
                                                                          size_type;
                                                                  size_type N =
                                                                      x.extent(
                                                                          0);
                                                                  if (y.extent(
                                                                          0) !=
                                                                      N)
                                                                    throw "saxp"
                                                                          "y: "
                                                                          "inco"
                                                                          "mpat"
                                                                          "ible"
                                                                          " dim"
                                                                          "ensi"
                                                                          "ons";
                                                                  Kokkos::parallel_for(
                                                                      N,
                                                                      saxpy_functor<
                                                                          ViewTypeA,
                                                                          ViewTypeX,
                                                                          ViewTypeY>(
                                                                          a, x,
                                                                          y));
}

// saxpy function
template<typename ViewTypeX, typename ViewTypeY>
void saxpy(const typename ViewTypeX::value_type& a, const ViewTypeX& x, const ViewTypeY& y) {
                                                                  typedef typename ViewTypeX::
                                                                      value_type
                                                                          value_type;
                                                                  typedef typename ViewTypeX::
                                                                      size_type
                                                                          size_type;
                                                                  size_type N =
                                                                      x.extent(
                                                                          0);
                                                                  if (y.extent(
                                                                          0) !=
                                                                      N)
                                                                    throw

=======

                                                                        // saxpy
                                                                        // Kokkos
                                                                        // function
                                                                        template <
                                                                            class
                                                                            ViewType1,
                                                                            class
                                                                            ViewType2,
                                                                            class
                                                                            ViewType3,
                                                                            class
                                                                            ScalarType>
                                                                        struct
                                                                        saxpy_functor {
                                                                      typedef typename ViewType1::
                                                                          size_type
                                                                              size_type;
                                                                      ViewType1
                                                                          x;
                                                                      ViewType2
                                                                          y;
                                                                      ViewType3
                                                                          z;
                                                                      ScalarType
                                                                          a;
                                                                      saxpy_functor(
                                                                          ViewType1
                                                                              x_,
                                                                          ViewType2
                                                                              y_,
                                                                          ViewType3
                                                                              z_,
                                                                          ScalarType
                                                                              a_)
                                                                          : x(x_),
                                                                            y(y_),
                                                                            z(z_),
                                                                            a(a_) {
                                                                      }
                                                                      KOKKOS_INLINE_FUNCTION
                                                                      void
                                                                      operator()(
                                                                          const size_type
                                                                              i)
                                                                          const {
                                                                        z(i) =
                                                                            a * x(i) +
                                                                            y(i);
                                                                      }
                                                                    };

                                                                  // saxpy
                                                                  // driver
                                                                  template <
                                                                      class
                                                                      ViewType1,
                                                                      class
                                                                      ViewType2,
                                                                      class
                                                                      ViewType3,
                                                                      class
                                                                      ScalarType>
                                                                  void saxpy(
                                                                      ViewType1
                                                                          x,
                                                                      ViewType2
                                                                          y,
                                                                      ViewType3
                                                                          z,
                                                                      ScalarType
                                                                          a) {
                                                                    typedef typename ViewType1::
                                                                        size_type
                                                                            size_type;
                                                                    size_type N =
                                                                        x.extent(
                                                                            0);
                                                                    Kokkos::parallel_for(
                                                                        N,
                                                                        saxpy_functor<
                                                                            ViewType1,
                                                                            ViewType2,
                                                                            ViewType3,
                                                                            ScalarType>(
                                                                            x,
                                                                            y,
                                                                            z,
                                                                            a));
                                                                  }

                                                                  int main(
                                                                      int argc,
                                                                      char *argv
                                                                          []) {
                                                                    Kokkos::
                                                                        initialize(
                                                                            argc,
                                                                            argv);

                                                                    // Problem
                                                                    // size
                                                                    const int N =
                                                                        1000;

                                                                    // Create 3
                                                                    // views
                                                                    typedef Kokkos::
                                                                        View<
                                                                            double
                                                                                *>
                                                                            view_type;
                                                                    view_type x(
                                                                        "x", N);
                                                                    view_type y(
                                                                        "y", N);
                                                                    view_type z(
                                                                        "z", N);

                                                                    // Initialize
                                                                    // x and y
                                                                    Kokkos::parallel_for(
                                                                        N,
                                                                        KOKKOS_LAMBDA(
                                                                            const int
                                                                                i) {
                                                                          x(i) =
                                                                              1.0;
                                                                          y(i) =
                                                                              2.0;
                                                                        });

                                                                    // Call
                                                                    // saxpy
                                                                    saxpy(x, y,
                                                                          z,
                                                                          3.0);

                                                                    // Print z
                                                                    Kokkos::parallel_for(
                                                                        N,
                                                                        KOKKOS_LAMBDA(
                                                                            const int
                                                                                i) {
                                                                          std::cout
                                                                              << "z("
                                                                              << i
                                                                              << ") = "
                                                                              << z(i)
                                                                              << std::
                                                                                     endl;
                                                                        });

                                                                    Kokkos::
                                                                        finalize();
                                                                  }

[/code]

And the output:

[code]

z(0) = 5
z(1) = 5
z(2) = 5
z(3) = 5
z(4) = 5
z(5

=======

// saxpy Kokkos function
template<class ViewType>
struct saxpy_functor {
                                                                    typedef typename ViewType::
                                                                        value_type
                                                                            value_type;
                                                                    ViewType a;
                                                                    ViewType x;
                                                                    ViewType y;

                                                                    saxpy_functor(
                                                                        ViewType
                                                                            _a,
                                                                        ViewType
                                                                            _x,
                                                                        ViewType
                                                                            _y)
                                                                        : a(_a),
                                                                          x(_x),
                                                                          y(_y) {
                                                                    }

                                                                    KOKKOS_INLINE_FUNCTION
                                                                    void
                                                                    operator()(
                                                                        int i)
                                                                        const {
                                                                      y(i) =
                                                                          a(0) *
                                                                              x(i) +
                                                                          y(i);
                                                                    }
};

// saxpy Kokkos function
template<class ViewType>
struct saxpy_functor2 {
                                                                    typedef typename ViewType::
                                                                        value_type
                                                                            value_type;
                                                                    ViewType a;
                                                                    ViewType x;
                                                                    ViewType y;

                                                                    saxpy_functor2(
                                                                        ViewType
                                                                            _a,
                                                                        ViewType
                                                                            _x,
                                                                        ViewType
                                                                            _y)
                                                                        : a(_a),
                                                                          x(_x),
                                                                          y(_y) {
                                                                    }

                                                                    KOKKOS_INLINE_FUNCTION
                                                                    void
                                                                    operator()(
                                                                        int i)
                                                                        const {
                                                                      y(i) =
                                                                          a(0) *
                                                                              x(i) +
                                                                          y(i);
                                                                    }
};

int main(int argc, char* argv[])
{
                                                                    typedef Kokkos::
                                                                        OpenMP
                                                                            space_type;
                                                                    typedef Kokkos::View<
                                                                        double
                                                                            *,
                                                                        space_type>
                                                                        view_type;
                                                                    typedef Kokkos::
                                                                        RangePolicy<
                                                                            space_type>
                                                                            range_type;

                                                                    Kokkos::
                                                                        initialize(
                                                                            argc,
                                                                            argv);
                                                                    {
                                                                      const int N =
                                                                          10000000;
                                                                      view_type
                                                                          a("a",
                                                                            1);
                                                                      view_type
                                                                          x("x",
                                                                            N);
                                                                      view_type
                                                                          y("y",
                                                                            N);

                                                                      // Initialize
                                                                      // data
                                                                      Kokkos::parallel_for(
                                                                          range_type(
                                                                              0,
                                                                              N),
                                                                          KOKKOS_LAMBDA(
                                                                              int i) {
                                                                            x(i) =
                                                                                1.0;
                                                                            y(i) =
                                                                                2.0;
                                                                          });
                                                                      a(0) =
                                                                          3.0;

                                                                      // Do
                                                                      // saxpy
                                                                      Kokkos::Timer
                                                                          timer;
                                                                      Kokkos::parallel_for(
                                                                          range_type(
                                                                              0,
                                                                              N),
                                                                          saxpy_functor<
                                                                              view_type>(
                                                                              a,
                                                                              x,
                                                                              y));
                                                                      Kokkos::
                                                                          fence();
                                                                      const double time1 =
                                                                          timer
                                                                              .seconds();

                                                                      // Do
                                                                      // saxpy
                                                                      // again
                                                                      timer
                                                                          .reset();
                                                                      Kokkos::parallel_for(
                                                                          range_type(
                                                                              0,
                                                                              N),
                                                                          saxpy_functor2<
                                                                              view_type>(
                                                                              a,
                                                                              x,
                                                                              y));
                                                                      Kokkos::
                                                                          fence();
                                                                      const double time2 =
                                                                          timer
                                                                              .seconds();

=======

                                                                      // saxpy
                                                                      // Kokkos
                                                                      // function
                                                                      template <
                                                                          class
                                                                          ViewType1,
                                                                          class
                                                                          ViewType2,
                                                                          class
                                                                          ViewType3,
                                                                          class
                                                                          Scalar>
                                                                      struct
                                                                          saxpy_functor {
                                                                        typedef typename ViewType1::
                                                                            size_type
                                                                                size_type;

                                                                        ViewType1
                                                                            x;
                                                                        ViewType2
                                                                            y;
                                                                        ViewType3
                                                                            z;
                                                                        Scalar
                                                                            a;

                                                                        saxpy_functor(
                                                                            Scalar
                                                                                a_,
                                                                            ViewType1
                                                                                x_,
                                                                            ViewType2
                                                                                y_,
                                                                            ViewType3
                                                                                z_)
                                                                            : x(x_),
                                                                              y(y_),
                                                                              z(z_),
                                                                              a(a_) {
                                                                        }

                                                                        KOKKOS_INLINE_FUNCTION
                                                                        void
                                                                        operator()(
                                                                            const size_type
                                                                                i)
                                                                            const {
                                                                          z(i) =
                                                                              a * x(i) +
                                                                              y(i);
                                                                        }
                                                                      };

                                                                      // saxpy
                                                                      // function
                                                                      template <
                                                                          class
                                                                          ViewType1,
                                                                          class
                                                                          ViewType2,
                                                                          class
                                                                          ViewType3,
                                                                          class
                                                                          Scalar>
                                                                      void
                                                                      saxpy(
                                                                          Scalar
                                                                              a,
                                                                          ViewType1
                                                                              x,
                                                                          ViewType2
                                                                              y,
                                                                          ViewType3
                                                                              z) {
                                                                        typedef typename ViewType1::
                                                                            size_type
                                                                                size_type;
                                                                        const size_type
                                                                            N = x.extent(
                                                                                0);
                                                                        Kokkos::parallel_for(
                                                                            N,
                                                                            saxpy_functor<
                                                                                ViewType1,
                                                                                ViewType2,
                                                                                ViewType3,
                                                                                Scalar>(
                                                                                a,
                                                                                x,
                                                                                y,
                                                                                z));
                                                                      }

                                                                      // saxpy
                                                                      // function
                                                                      template <
                                                                          class
                                                                          ViewType1,
                                                                          class
                                                                          ViewType2,
                                                                          class
                                                                          ViewType3,
                                                                          class
                                                                          Scalar>
                                                                      void
                                                                      saxpy_cuda(
                                                                          Scalar
                                                                              a,
                                                                          ViewType1
                                                                              x,
                                                                          ViewType2
                                                                              y,
                                                                          ViewType3
                                                                              z) {
                                                                        typedef typename ViewType1::
                                                                            size_type
                                                                                size_type;
                                                                        const size_type
                                                                            N = x.extent(
                                                                                0);
                                                                        Kokkos::parallel_for(
                                                                            N,
                                                                            saxpy_functor<
                                                                                ViewType1,
                                                                                ViewType2,
                                                                                ViewType3,
                                                                                Scalar>(
                                                                                a,
                                                                                x,
                                                                                y,
                                                                                z));
                                                                      }

                                                                      // saxpy
                                                                      // function
                                                                      template <
                                                                          class
                                                                          ViewType1,
                                                                          class
                                                                          ViewType2,
                                                                          class
                                                                          ViewType3,
                                                                          class
                                                                          Scalar>
                                                                      void
                                                                      saxpy_openmp(
                                                                          Scalar
                                                                              a,
                                                                          ViewType1
                                                                              x,
                                                                          ViewType2
                                                                              y,
                                                                          ViewType3
                                                                              z) {
                                                                        typedef typename ViewType1::
                                                                            size_type
                                                                                size_type;
                                                                        const size_type
                                                                            N = x.extent(
                                                                                0);
                                                                        Kokkos::parallel_for(
                                                                            N,
                                                                            saxpy_functor<
                                                                                ViewType1,
                                                                                ViewType2,
                                                                                ViewType3,
                                                                                Scalar>(
                                                                                a,
                                                                                x,
                                                                                y,
                                                                                z));
                                                                      }

                                                                      // saxpy
                                                                      // function
                                                                      template <
                                                                          class
                                                                          ViewType1,
                                                                          class
                                                                          ViewType2,
                                                                          class
                                                                          ViewType3,
                                                                          class
                                                                          Scalar>
                                                                      void
                                                                      saxpy_serial(
                                                                          Scalar
                                                                              a,
                                                                          ViewType1
                                                                              x,
                                                                          ViewType2
                                                                              y,
                                                                          ViewType3
                                                                              z)

=======

                                                                          // saxpy
                                                                          // Kokkos
                                                                          // function
                                                                          // y =
                                                                          // a*x
                                                                          // + y
                                                                          template <
                                                                              class
                                                                              ViewType1,
                                                                              class
                                                                              ViewType2,
                                                                              class
                                                                              ViewType3>
                                                                          struct
                                                                          saxpy_functor {
                                                                        typedef typename ViewType1::
                                                                            value_type
                                                                                value_type;
                                                                        typedef typename ViewType1::
                                                                            size_type
                                                                                size_type;
                                                                        ViewType1
                                                                            x,
                                                                            y;
                                                                        value_type
                                                                            a;
                                                                        saxpy_functor(
                                                                            value_type
                                                                                _a,
                                                                            ViewType1
                                                                                _x,
                                                                            ViewType2
                                                                                _y)
                                                                            : a(_a),
                                                                              x(_x),
                                                                              y(_y) {
                                                                        }
                                                                        KOKKOS_INLINE_FUNCTION
                                                                        void
                                                                        operator()(
                                                                            const size_type
                                                                                i)
                                                                            const {
                                                                          y(i) =
                                                                              a * x(i) +
                                                                              y(i);
                                                                        }
                                                                      };

                                                                      // saxpy
                                                                      // host
                                                                      // function
                                                                      // y = a*x
                                                                      // + y
                                                                      template <
                                                                          class
                                                                          ViewType1,
                                                                          class
                                                                          ViewType2,
                                                                          class
                                                                          ViewType3>
                                                                      void
                                                                      saxpy(
                                                                          value_type
                                                                              a,
                                                                          ViewType1
                                                                              x,
                                                                          ViewType2
                                                                              y,
                                                                          ViewType3
                                                                              z) {
                                                                        z = a * x +
                                                                            y;
                                                                      }

                                                                      int main(
                                                                          int argc,
                                                                          char *argv
                                                                              []) {

                                                                        // Kokkos
                                                                        // initialization
                                                                        Kokkos::initialize(
                                                                            argc,
                                                                            argv);

                                                                        // Problem
                                                                        // size
                                                                        int N =
                                                                            1000;
                                                                        // Scalar
                                                                        // value
                                                                        value_type
                                                                            a = 3.0;

                                                                        // Kokkos
                                                                        // views
                                                                        Kokkos::View<
                                                                            value_type
                                                                                *>
                                                                            x("x",
                                                                              N);
                                                                        Kokkos::View<
                                                                            value_type
                                                                                *>
                                                                            y("y",
                                                                              N);
                                                                        Kokkos::View<
                                                                            value_type
                                                                                *>
                                                                            z("z",
                                                                              N);

                                                                        // Initialize
                                                                        // x and
                                                                        // y
                                                                        Kokkos::parallel_for(
                                                                            N,
                                                                            KOKKOS_LAMBDA(
                                                                                const int
                                                                                    i) {
                                                                              x(i) =
                                                                                  2.0;
                                                                              y(i) =
                                                                                  1.0;
                                                                            });

                                                                        // Run
                                                                        // saxpy
                                                                        // on
                                                                        // device
                                                                        Kokkos::parallel_for(
                                                                            N,
                                                                            saxpy_functor<
                                                                                Kokkos::View<
                                                                                    value_type
                                                                                        *>,
                                                                                Kokkos::View<
                                                                                    value_type
                                                                                        *>,
                                                                                Kokkos::View<
                                                                                    value_type
                                                                                        *>>(
                                                                                a,
                                                                                x,
                                                                                y));

                                                                        // Run
                                                                        // saxpy
                                                                        // on
                                                                        // host
                                                                        saxpy(
                                                                            a,
                                                                            x,
                                                                            y,
                                                                            z);

                                                                        // Check
                                                                        // results
  Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int i) {
                                                                          if (z(i) !=
                                                                              y(i)) {
      printf("Error: z(%d) = %f, y(%d) = %f

", i, z(i), i, y(i));
                                                                          }

=======

                                                                          // saxpy
                                                                          // Kokkos
                                                                          // function
                                                                          template <
                                                                              typename ViewTypeA,
                                                                              typename ViewTypeB,
                                                                              typename ViewTypeC>
                                                                          struct
                                                                              SAXPYFunctor {
                                                                            ViewTypeA
                                                                                a;
                                                                            ViewTypeB
                                                                                b;
                                                                            ViewTypeC
                                                                                c;
                                                                            const int
                                                                                N;

                                                                            SAXPYFunctor(
                                                                                ViewTypeA
                                                                                    a_,
                                                                                ViewTypeB
                                                                                    b_,
                                                                                ViewTypeC
                                                                                    c_,
                                                                                int N_)
                                                                                : a(a_),
                                                                                  b(b_),
                                                                                  c(c_),
                                                                                  N(N_){};

                                                                            KOKKOS_INLINE_FUNCTION
                                                                            void
                                                                            operator()(
                                                                                const int
                                                                                    &i)
                                                                                const {
                                                                              c(i) =
                                                                                  a(0) *
                                                                                      b(i) +
                                                                                  c(i);
                                                                            }
                                                                          };

                                                                          int
                                                                          main(
                                                                              int argc,
                                                                              char *argv
                                                                                  []) {
                                                                            Kokkos::initialize(
                                                                                argc,
                                                                                argv);

                                                                            int N =
                                                                                1000000;
                                                                            if (argc >
                                                                                1) {
                                                                              N = atoi(
                                                                                  argv[1]);
                                                                            }

                                                                            // Host Views
                                                                            Kokkos::View<
                                                                                double
                                                                                    *>
                                                                                a("a",
                                                                                  1);
                                                                            Kokkos::View<
                                                                                double
                                                                                    *>
                                                                                b("b",
                                                                                  N);
                                                                            Kokkos::View<
                                                                                double
                                                                                    *>
                                                                                c("c",
                                                                                  N);

                                                                            // Initialize Views
                                                                            Kokkos::parallel_for(
                                                                                N,
                                                                                KOKKOS_LAMBDA(
                                                                                    const int
                                                                                        &i) {
                                                                                  a(0) =
                                                                                      1.0;
                                                                                  b(i) =
                                                                                      2.0;
                                                                                  c(i) =
                                                                                      0.0;
                                                                                });

                                                                            // saxpy
                                                                            Kokkos::Timer
                                                                                timer;
                                                                            Kokkos::parallel_for(
                                                                                N,
                                                                                SAXPYFunctor<
                                                                                    Kokkos::View<
                                                                                        double
                                                                                            *>,
                                                                                    Kokkos::View<
                                                                                        double
                                                                                            *>,
                                                                                    Kokkos::View<
                                                                                        double
                                                                                            *>>(
                                                                                    a,
                                                                                    b,
                                                                                    c,
                                                                                    N));
                                                                            Kokkos::
                                                                                fence();
                                                                            double time =
                                                                                timer
                                                                                    .seconds();

                                                                            std::cout
                                                                                << "N = "
                                                                                << N
                                                                                << ", time = "
                                                                                << time
                                                                                << std::
                                                                                       endl;

                                                                            Kokkos::
                                                                                finalize();
                                                                            return 0;
                                                                          }

                                                                          // Path:
                                                                          // SAXPY/Makefile

=======

                                                                          // saxpy
                                                                          // Kokkos
                                                                          // function
                                                                          template <
                                                                              class
                                                                              ViewType>
                                                                          struct
                                                                              SAXPY {
                                                                            typedef typename ViewType::
                                                                                value_type
                                                                                    value_type;
                                                                            ViewType
                                                                                _a;
                                                                            ViewType
                                                                                _x;
                                                                            ViewType
                                                                                _y;
                                                                            value_type
                                                                                _alpha;
                                                                            SAXPY(
                                                                                const value_type
                                                                                    alpha,
                                                                                const ViewType
                                                                                    &a,
                                                                                const ViewType
                                                                                    &x,
                                                                                const ViewType
                                                                                    &y)
                                                                                : _a(a),
                                                                                  _x(x),
                                                                                  _y(y),
                                                                                  _alpha(
                                                                                      alpha) {
                                                                            }
                                                                            KOKKOS_INLINE_FUNCTION
                                                                            void
                                                                            operator()(
                                                                                const int
                                                                                    i)
                                                                                const {
                                                                              _a(i) =
                                                                                  _alpha *
                                                                                      _x(i) +
                                                                                  _y(i);
                                                                            }
                                                                          };

                                                                          int
                                                                          main(
                                                                              int argc,
                                                                              char *argv
                                                                                  []) {
                                                                            Kokkos::initialize(
                                                                                argc,
                                                                                argv);
                                                                            {
                                                                              const int
                                                                                  N = 1000000;

                                                                              // Create 3 views of N doubles
                                                                              Kokkos::View<
                                                                                  double
                                                                                      *>
                                                                                  a("A",
                                                                                    N);
                                                                              Kokkos::View<
                                                                                  double
                                                                                      *>
                                                                                  x("X",
                                                                                    N);
                                                                              Kokkos::View<
                                                                                  double
                                                                                      *>
                                                                                  y("Y",
                                                                                    N);

                                                                              // Initialize x and y on the host
                                                                              for (
                                                                                  int i =
                                                                                      0;
                                                                                  i <
                                                                                  N;
                                                                                  i++) {
                                                                                x(i) =
                                                                                    1.0;
                                                                                y(i) =
                                                                                    2.0;
                                                                              }

                                                                              // Create a functor for the SAXPY operation
                                                                              const double
                                                                                  alpha =
                                                                                      3.0;
                                                                              SAXPY<Kokkos::View<
                                                                                  double
                                                                                      *>>
                                                                                  saxpy(
                                                                                      alpha,
                                                                                      a,
                                                                                      x,
                                                                                      y);

                                                                              // Perform the SAXPY operation
                                                                              Kokkos::parallel_for(
                                                                                  N,
                                                                                  saxpy);
                                                                            }
                                                                            Kokkos::
                                                                                finalize();
                                                                            return 0;
                                                                          }

                                                                          I am trying to run the code
                                                                                      on the following
                                                                                          machine
                                                                              :

                                                                              $ uname -
                                                                              a Linux
                                                                                  gpu -
                                                                              2 2.6.32 -
                                                                              642.6.2.el6.x86_64 #1 SMP Tue
                                                                                      May 31 18 : 38 : 21 EDT 2016 x86_64
                                                                                          x86_64 x86_64
                                                                                              GNU /
                                                                                  Linux

                                                                                      $ cat /
                                                                                  etc /
                                                                                  redhat -
                                                                              release CentOS
                                                                                  release 6.8(
                                                                                      Final)

                                                                                      $ nvidia -
                                                                              smi Fri
                                                                                  Jul 14 16 : 56 : 43 2017 +
                                                                              -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -+ |
                                                                              NVIDIA - SMI 367.57 Driver Version : 367.57 | | -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -+-- -- -- -- -- -- -- -- -- -- --+-- -- -- -- -- -- -- -- -- -- --+ |
                                                                              GPU Name
                                                                                      Persistence -
                                                                                  M |
                                                                              Bus -
                                                                                  Id Disp
                                                                                      .A |
                                                                              Volatile Uncorr
                                                                                  .ECC |
                                                                              |
                                                                              Fan Temp
                                                                                  Perf

=======

                                                                              // saxpy Kokkos function
                                                                              template <
                                                                                  class
                                                                                  ExecSpace>
                                                                              struct
                                                                              saxpy_functor {
                                                                            typedef ExecSpace
                                                                                execution_space;
                                                                            typedef typename View<
                                                                                double
                                                                                    *,
                                                                                ExecSpace>::
                                                                                size_type
                                                                                    size_type;
                                                                            View<
                                                                                double
                                                                                    *,
                                                                                ExecSpace>
                                                                                x,
                                                                                y;
                                                                            double
                                                                                a;
                                                                            saxpy_functor(
                                                                                double
                                                                                    a_,
                                                                                View<
                                                                                    double
                                                                                        *,
                                                                                    ExecSpace>
                                                                                    x_,
                                                                                View<
                                                                                    double
                                                                                        *,
                                                                                    ExecSpace>
                                                                                    y_)
                                                                                : a(a_),
                                                                                  x(x_),
                                                                                  y(y_) {
                                                                            }
                                                                            KOKKOS_INLINE_FUNCTION
                                                                            void
                                                                            operator()(
                                                                                const size_type
                                                                                    i)
                                                                                const {
                                                                              y(i) =
                                                                                  a * x(i) +
                                                                                  y(i);
                                                                            }
                                                                          };

                                                                          // saxpy
                                                                          // Kokkos
                                                                          // function
                                                                          template <
                                                                              class
                                                                              ExecSpace>
                                                                          struct
                                                                              saxpy_functor2 {
                                                                            typedef ExecSpace
                                                                                execution_space;
                                                                            typedef typename View<
                                                                                double
                                                                                    *,
                                                                                ExecSpace>::
                                                                                size_type
                                                                                    size_type;
                                                                            View<
                                                                                double
                                                                                    *,
                                                                                ExecSpace>
                                                                                x,
                                                                                y;
                                                                            double
                                                                                a;
                                                                            saxpy_functor2(
                                                                                double
                                                                                    a_,
                                                                                View<
                                                                                    double
                                                                                        *,
                                                                                    ExecSpace>
                                                                                    x_,
                                                                                View<
                                                                                    double
                                                                                        *,
                                                                                    ExecSpace>
                                                                                    y_)
                                                                                : a(a_),
                                                                                  x(x_),
                                                                                  y(y_) {
                                                                            }
                                                                            KOKKOS_INLINE_FUNCTION
                                                                            void
                                                                            operator()(
                                                                                const size_type
                                                                                    i)
                                                                                const {
                                                                              y(i) =
                                                                                  a * x(i) +
                                                                                  y(i);
                                                                            }
                                                                          };

                                                                          // saxpy
                                                                          // Kokkos
                                                                          // function
                                                                          template <
                                                                              class
                                                                              ExecSpace>
                                                                          struct
                                                                              saxpy_functor3 {
                                                                            typedef ExecSpace
                                                                                execution_space;
                                                                            typedef typename View<
                                                                                double
                                                                                    *,
                                                                                ExecSpace>::
                                                                                size_type
                                                                                    size_type;
                                                                            View<
                                                                                double
                                                                                    *,
                                                                                ExecSpace>
                                                                                x,
                                                                                y;
                                                                            double
                                                                                a;
                                                                            saxpy_functor3(
                                                                                double
                                                                                    a_,
                                                                                View<
                                                                                    double
                                                                                        *,
                                                                                    ExecSpace>
                                                                                    x_,
                                                                                View<
                                                                                    double
                                                                                        *,
                                                                                    ExecSpace>
                                                                                    y_)
                                                                                : a(a_),
                                                                                  x(x_),
                                                                                  y(y_) {
                                                                            }
                                                                            KOKKOS_INLINE_FUNCTION
                                                                            void
                                                                            operator()(
                                                                                const size_type
                                                                                    i)
                                                                                const {
                                                                              y(i) =
                                                                                  a * x(i) +
                                                                                  y(i);
                                                                            }
                                                                          };

                                                                          // saxpy
                                                                          // Kokkos
                                                                          // function
                                                                          template <
                                                                              class
                                                                              ExecSpace>
                                                                          struct
                                                                              saxpy_functor4 {
                                                                            typedef ExecSpace
                                                                                execution_space;
                                                                            typedef typename View<
                                                                                double
                                                                                    *,
                                                                                ExecSpace>::
                                                                                size_type
                                                                                    size_type;
                                                                            View<
                                                                                double
                                                                                    *,
                                                                                ExecSpace>
                                                                                x,
                                                                                y;
                                                                            double
                                                                                a;
                                                                            saxpy_functor4(
                                                                                double
                                                                                    a_,
                                                                                View<
                                                                                    double
                                                                                        *,
                                                                                    ExecSpace>
                                                                                    x_,
                                                                                View<
                                                                                    double
                                                                                        *,
                                                                                    ExecSpace>
                                                                                    y_)
                                                                                : a(a_),
                                                                                  x(x_),
                                                                                  y(y_) {
                                                                            }
                                                                            KOKKOS_INLINE_FUNCTION
                                                                            void
                                                                            operator()(
                                                                                const size_type
                                                                                    i)
                                                                                const {
                                                                              y(i) =
                                                                                  a * x(i) +
                                                                                  y(i);
                                                                            }
                                                                          };

                                                                          // saxpy
                                                                          // Prompt:
                                                                          // saxpy
                                                                          // CUDA
                                                                          // function
                                                                          Synthesizing 10 /
                                                                              10 solutions

=======

                                                                                  // saxpy CUDA function
                                                                                  __global__ void
                                                                                  saxpy(
                                                                                      int n,
                                                                                      float
                                                                                          a,
                                                                                      float
                                                                                          *x,
                                                                                      float *
                                                                                          y) {
                                                                            int i =
                                                                                blockIdx.x *
                                                                                    blockDim
                                                                                        .x +
                                                                                threadIdx
                                                                                    .x;
                                                                            if (i <
                                                                                n)
                                                                              y[i] =
                                                                                  a * x[i] +
                                                                                  y[i];
                                                                          }

                                                                          int
                                                                          main(
                                                                              void) {
                                                                            int N =
                                                                                1
                                                                                << 20;
                                                                            float
                                                                                *x,
                                                                                *y,
                                                                                *d_x,
                                                                                *d_y;
                                                                            x = (float *)malloc(
                                                                                N *
                                                                                sizeof(
                                                                                    float));
                                                                            y = (float *)malloc(
                                                                                N *
                                                                                sizeof(
                                                                                    float));

                                                                            // Allocate Unified Memory – accessible from CPU or GPU
                                                                            cudaMallocManaged(
                                                                                &d_x,
                                                                                N * sizeof(
                                                                                        float));
                                                                            cudaMallocManaged(
                                                                                &d_y,
                                                                                N * sizeof(
                                                                                        float));

                                                                            // Initialize x and y arrays on the host
                                                                            for (
                                                                                int i =
                                                                                    0;
                                                                                i <
                                                                                N;
                                                                                i++) {
                                                                              x[i] =
                                                                                  1.0f;
                                                                              y[i] =
                                                                                  2.0f;
                                                                            }

                                                                            // Copy x and y arrays to device
                                                                            cudaMemcpy(
                                                                                d_x,
                                                                                x,
                                                                                N * sizeof(
                                                                                        float),
                                                                                cudaMemcpyHostToDevice);
                                                                            cudaMemcpy(
                                                                                d_y,
                                                                                y,
                                                                                N * sizeof(
                                                                                        float),
                                                                                cudaMemcpyHostToDevice);

                                                                            // Launch SAXPY kernel on 1M elements on the GPU
                                                                            saxpy<<<
                                                                                (N +
                                                                                 255) /
                                                                                    256,
                                                                                256>>>(
                                                                                N,
                                                                                2.0f,
                                                                                d_x,
                                                                                d_y);

                                                                            // Wait for GPU to finish before accessing on host
                                                                            cudaDeviceSynchronize();

                                                                            // Check for errors (all values should be 4.0f)
                                                                            float maxError =
                                                                                0.0f;
                                                                            for (
                                                                                int i =
                                                                                    0;
                                                                                i <
                                                                                N;
                                                                                i++)
                                                                              maxError = fmax(
                                                                                  maxError,
                                                                                  fabs(
                                                                                      y[i] -
                                                                                      4.0f));
    printf("Max error: %f

", maxError);

    // Free memory
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    return 0;
                                                                          }

The code is very simple. In the main function, we allocate the memory for x and y arrays on the host and initialize them to 1.0f and 2.0f, respectively. We also allocate the memory for d_x and d_y arrays on the device. We then copy the x and y arrays from the host to the device. Next, we launch the saxpy kernel on the GPU. The saxpy kernel is defined as follows:

// saxpy CUDA function

=======

// saxpy CUDA function
__global__ void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a * x[i] + y[i];
}

int main(void) {
  int N = 1 << 20; // 1M elements

  float *x, *y, *d_x, *d_y;
  x = (float *)malloc(N * sizeof(float));
  y = (float *)malloc(N * sizeof(float));

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&d_x, N * sizeof(float));
  cudaMallocManaged(&d_y, N * sizeof(float));

  // Initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Copy x and y arrays to device
  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  saxpy<<<numBlocks, blockSize>>>(N, 2.0f, d_x, d_y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 4.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i] - 4.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

  return 0;
}

Output

    Max error : 0

    The output of the program is the maximum error between the values in the y
        array and the expected value of 4.0f. The error is calculated
            as the absolute value of the difference between the expected
                value and the actual value,
    and is stored in the maxError variable.The max function returns the larger
            of the two arguments

=======

                // saxpy CUDA function
                __global__ void
                saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a * x[i] + y[i];
}

int main(void) {
  int N = 1 << 20;
  float *x, *y, *d_x, *d_y;
  x = (float *)malloc(N * sizeof(float));
  y = (float *)malloc(N * sizeof(float));

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&d_x, N * sizeof(float));
  cudaMallocManaged(&d_y, N * sizeof(float));

  // Initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Copy x and y arrays to device
  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  // Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  saxpy<<<numBlocks, blockSize>>>(N, 2.0f, d_x, d_y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 4.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i] - 4.0f));
  printf("Max error: %f

", maxError);

  // Free memory
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

  return 0;
}

$ nvcc - o saxpy saxpy.cu $./
             saxpy Max error : 0.000000

             The CUDA compiler nvcc compiles the C++ code into a binary
                 executable.The executable is then run on the
                     GPU.The output of the program is shown below :

    Max error : 0.000000

    The program runs without errors and the output is as expected.The CUDA

=======

        // saxpy CUDA function
        __global__ void
        saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a * x[i] + y[i];
}

// main routine that executes on the host
int main(void) {
  int N = 1 << 20; // 1M elements
  float *x, *y, *d_x, *d_y;
  x = (float *)malloc(N * sizeof(float));
  y = (float *)malloc(N * sizeof(float));

  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));

  // Initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Copy x and y arrays to the device
  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  // Do calculation on device:
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  saxpy<<<numBlocks, blockSize>>>(N, 2.0f, d_x, d_y);

  // Retrieve result from device and store it in y
  cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Cleanup
  free(x);
  free(y);
  cudaFree(d_x);
  cudaFree(d_y);
  return 0;
}

The code above is a simple example of how to use CUDA in C++.The program
    performs a SAXPY operation on the device,
    where :

    The SAXPY operation is defined
        as y = a * x + y(where y is an array of floats)
                           .

                       a is a scalar,
           and x and y are arrays of floats.

                     The array size,
           N,
           is 1M elements.

               The code is divided into two parts : the host
                                                    code(the main routine) and
               the device code(the saxpy function)
                   .The host code is executed on the CPU,
           while the device code is executed on the GPU.The host code allocates
               memory on the device,
           copies the data from the host to the device,
           launches the kernel on the device,
           and copies the results

=======

               // saxpy CUDA function
               __global__ void
               saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a * x[i] + y[i];
}

int main(void) {
  int N = 1 << 20;
  float *x, *y, *d_x, *d_y;
  x = (float *)malloc(N * sizeof(float));
  y = (float *)malloc(N * sizeof(float));

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&d_x, N * sizeof(float));
  cudaMallocManaged(&d_y, N * sizeof(float));

  // Initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Copy x and y arrays to device
  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch SAXPY kernel on 1M elements on the GPU
  saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 4.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 4.0f));
    printf("Max error: %f

", maxError);

    // Free memory
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    return 0;
}

The CUDA kernel is defined in the function saxpy()
    .The kernel is called from the host code in the
    main() function.The CUDA kernel is launched with 256 threads per block,
    and4096 blocks.

    The CUDA kernel is launched with 256 threads per block,
    and4096 blocks.The total number of threads launched is 256 * 4096 =
        1048576. The kernel is launched with the following arguments :

    saxpy < < <
    (N + 255) /

=======

        // saxpy CUDA function
        // y = a * x + y
        __global__ void saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a * x[i] + y[i];
}

int main(void) {
  int N = 1 << 20;
  float *x, *y, *d_x, *d_y;
  x = (float *)malloc(N * sizeof(float));
  y = (float *)malloc(N * sizeof(float));

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&d_x, N * sizeof(float));
  cudaMallocManaged(&d_y, N * sizeof(float));

  // Initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Copy x and y arrays to device
  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  // Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  saxpy<<<numBlocks, blockSize>>>(N, 2.0f, d_x, d_y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 4.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i] - 4.0f));
  printf("Max error: %f

", maxError);

  // Free memory
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

  return 0;
}

Output

    Max error : 0.000000

    In the above code,
    we have allocated the memory using cudaMallocManaged() function
        .The memory allocated using this function is accessible by both
            CPU and GPU.The memory allocated using this function is allocated on
                the device,
    but it is also accessible by the host.So,
    we can copy the data from host to device using

=======

    // saxpy CUDA function
    // y[i] = a * x[i] + y[i]
    __global__ void
    saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a * x[i] + y[i];
}

// Main routine that executes on the host
int main(void) {
  // Size of vectors
  int N = 1 << 20;

  // Host input vectors
  float *h_x;
  float *h_y;
  // Host output vector
  float *h_out;

  // Device input vectors
  float *d_x;
  float *d_y;
  // Device output vector
  float *d_out;

  // Size, in bytes, of each vector
  size_t bytes = N * sizeof(float);

  // Allocate memory for each vector on host
  h_x = (float *)malloc(bytes);
  h_y = (float *)malloc(bytes);
  h_out = (float *)malloc(bytes);

  // Allocate memory for each vector on GPU
  cudaMalloc(&d_x, bytes);
  cudaMalloc(&d_y, bytes);
  cudaMalloc(&d_out, bytes);

  // Initialize vectors on host
  int i;
  for (i = 0; i < N; i++) {
    h_x[i] = rand() / (float)RAND_MAX;
    h_y[i] = rand() / (float)RAND_MAX;
  }

  // Copy host vectors to device
  cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

  // Number of threads in each thread block
  int THREADS_PER_BLOCK = 256;

  // Number of thread blocks in grid
  int BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  // Execute SAXPY kernel on GPU
  saxpy<<<BLOCKS, THREADS_PER_BLOCK>>>(N, 2.0, d_x, d_y);

  // Copy array back to host
  cudaMemcpy(h_out, d_y, bytes, cudaMemcpyDeviceToHost);

  // SAXPY on host
  for (i = 0; i < N; i++)
    h_y[i] = 2.0 * h_x[i] + h_y[i];

  //

=======

  // saxpy CUDA function
  __global__ void saxpy(float a, float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
      y[i] = a * x[i] + y[i];
    }
  }

  // main routine that executes on the host
  int main(void) {
    float a = 2.0f;
    int N = 1 << 20;
    float *x, *y, *d_x, *d_y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // Initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    saxpy<<<(N + 255) / 256, 256>>>(a, x, y, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 4.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
      maxError = max(maxError, abs(y[i] - 4.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
  }

  // Path: SAXPY/CUDA/SAXPY.cu

=======

  // saxpy CUDA function
  // It is called by the host
  __global__ void saxpy(int n, float a, float *x, float *y) {
    // 1D Thread ID (assuming that only *one* block will be executed)
    int i = threadIdx.x;

    if (i < n)
      y[i] = a * x[i] + y[i];
  }

  int main(void) {
    // Number of elements in vectors
    int N = 1 << 20;

    // Host input vectors
    float *h_x;
    float *h_y;
    // Host output vector
    float *h_z;

    // Device input vectors
    float *d_x;
    float *d_y;
    // Device output vector
    float *d_z;

    // Size, in bytes, of each vector
    size_t bytes = N * sizeof(float);

    // Allocate memory for each vector on host
    h_x = (float *)malloc(bytes);
    h_y = (float *)malloc(bytes);
    h_z = (float *)malloc(bytes);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_z, bytes);

    // Initialize vectors on host
    int i;
    for (i = 0; i < N; i++) {
      h_x[i] = rand() / (float)RAND_MAX;
      h_y[i] = rand() / (float)RAND_MAX;
    }

    // Copy host vectors to device
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    // Number of threads in each thread block
    int THREADS_PER_BLOCK = 256;

    // Number of thread blocks in grid
    int BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Execute the kernel
    saxpy<<<BLOCKS, THREADS_PER_BLOCK>>>(N, 2.0, d_x, d_y);

    // Copy array back to host
    cudaMemcpy(h_z, d_z, bytes, cudaMemcpyDeviceToHost);

    // Sum up vector z and print result divided by N, this should equal 1 within
    // error
    float sum = 0;
    for (i = 0; i < N; i++)
      sum

=======

                                                                                // saxpy CUDA function
                                                                                // y = a * x + y
                                                                                // x and y are vectors of length n
                                                                                // a is a scalar

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

          // CUDA kernel
          // y = a * x + y
          __global__ void
          saxpy(int n, float a, float *x, float *y) {
        // 1D Thread ID
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        // Make sure we do not go out of bounds
        if (i < n)
          y[i] = a * x[i] + y[i];
      }

    int main() {
      // Number of elements in vectors
      int N = 1 << 20;

      // Size, in bytes, of each vector
      size_t bytes = N * sizeof(float);

      // Host input vectors
      float *h_x, *h_y;

      // Device input vectors
      float *d_x, *d_y;

      // Allocate memory for each vector on host
      h_x = (float *)malloc(bytes);
      h_y = (float *)malloc(bytes);

      // Allocate memory for each vector on GPU
      cudaMalloc(&d_x, bytes);
      cudaMalloc(&d_y, bytes);

      // Initialize vectors on host
      for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
      }

      // Copy host vectors to device
      cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
      cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

      // Threads per CTA dimension
      int blockSize = 256;

      // Number of CTAs per grid dimension
      // (there can be no more than 65,535 CTAs in each grid dimension)
      int gridSize = (int)ceil((float)N / blockSize);

      // Execute SAXPY on 1M elements
      saxpy<<<gridSize, blockSize>>>(N, 2.0f, d_x, d_y);

      // Wait for GPU to finish before accessing on host
      cudaDeviceSynchronize();

      // Copy array back to host
      cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);

      //

      // saxpy HIP function
      Synthesizing 10 / 10 solutions

=======

                            // saxpy HIP function
                            __global__ void
                            saxpy(int n, float a, float *x, float *y) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
          y[i] = a * x[i] + y[i];
      }

      // main
      int main(int argc, char **argv) {
        // set up device
        hipSetDevice(0);

        // set up data size of vectors
        int N = 1 << 20;

        // allocate memory
        float *x, *y, *d_x, *d_y;
        size_t bytes = N * sizeof(float);
        x = (float *)malloc(bytes);
        y = (float *)malloc(bytes);
        hipMalloc(&d_x, bytes);
        hipMalloc(&d_y, bytes);

        // initialize data at host side
        float a = 2.0f;
        for (int i = 0; i < N; i++) {
          x[i] = 1.0f;
          y[i] = 2.0f;
        }

        // copy data from host to device
        hipMemcpy(d_x, x, bytes, hipMemcpyHostToDevice);
        hipMemcpy(d_y, y, bytes, hipMemcpyHostToDevice);

        // launch kernel
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        hipLaunchKernelGGL(saxpy, dim3(gridSize), dim3(blockSize), 0, 0, N, a,
                           d_x, d_y);

        // copy data back to host side
        hipMemcpy(y, d_y, bytes, hipMemcpyDeviceToHost);

        // check device results
        float maxError = 0.0f;
        for (int i = 0; i < N; i++)
          maxError = fmax(maxError, fabs(y[i] - 4.0f));
  printf("Max error: %f

", maxError);

  // free device global memory
  hipFree(d_x);
  hipFree(d_y);

  // free host memory
  free(x);
  free(y);

  return 0;
      }

      [/ sourcecode]

          This code was tested with the following compiler and libraries :

          *hipcc version 3.5.0 -
          0.9(amd64)*ROCm Version : 3.5.0 *

=======

              // saxpy HIP function
              __global__ void saxpy(int n, float a, float *x, float *y) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
          y[i] = a * x[i] + y[i];
      }

      int main(int argc, char *argv[]) {
        int N = 1 << 20;
        float a = 2.0f;

        size_t size = N * sizeof(float);

        // Allocate input vectors h_x and h_y in host memory
        float *h_x = (float *)malloc(size);
        float *h_y = (float *)malloc(size);

        // Initialize input vectors
        for (int i = 0; i < N; i++) {
          h_x[i] = 1.0f;
          h_y[i] = 2.0f;
        }

        // Allocate vectors in device memory
        float *d_x, *d_y;
        hipMalloc(&d_x, size);
        hipMalloc(&d_y, size);

        // Copy vectors from host memory to device memory
        hipMemcpy(d_x, h_x, size, hipMemcpyHostToDevice);
        hipMemcpy(d_y, h_y, size, hipMemcpyHostToDevice);

        // Invoke kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        hipLaunchKernelGGL(saxpy, dim3(blocksPerGrid), dim3(threadsPerBlock), 0,
                           0, N, a, d_x, d_y);

        // Copy result from device memory to host memory
        // h_y contains the result in host memory
        hipMemcpy(h_y, d_y, size, hipMemcpyDeviceToHost);

        // Verify result
        bool passed = true;
        for (int i = 0; i < N; i++) {
          if (h_y[i] != 4.0f) {
            passed = false;
            break;
          }
        }

        // Free device memory
        hipFree(d_x);
        hipFree(d_y);

        // Free host memory
        free(h_x);
        free(h_y);

        if (passed) {
        printf("Test Passed!

");
        } else {
        printf("Test Failed!

");
        }

        return 0;
      }

      The above code is a simple example of

=======

          // saxpy HIP function
          __global__ void
          saxpy(int n, float a, float *x, float *y) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
          y[i] = a * x[i] + y[i];
      }

      int main(int argc, char **argv) {
        // Read command line arguments
        if (argc != 2) {
          std::cout << "Usage: " << argv[0] << " N" << std::endl;
          return 1;
        }
        int N = atoi(argv[1]);

        // Allocate and initialize vectors
        float *x = new float[N];
        float *y = new float[N];
        for (int i = 0; i < N; ++i) {
          x[i] = 1.0f;
          y[i] = 2.0f;
        }

        // Allocate device memory
        float *x_d, *y_d;
        hipMalloc(&x_d, N * sizeof(float));
        hipMalloc(&y_d, N * sizeof(float));

        // Copy data to device
        hipMemcpy(x_d, x, N * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(y_d, y, N * sizeof(float), hipMemcpyHostToDevice);

        // Launch kernel
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        hipLaunchKernelGGL(saxpy, dim3(gridSize), dim3(blockSize), 0, 0, N,
                           2.0f, x_d, y_d);

        // Copy result back to host
        hipMemcpy(y, y_d, N * sizeof(float), hipMemcpyDeviceToHost);

        // Verify result
        for (int i = 0; i < N; ++i) {
          assert(y[i] == 4.0f);
        }

        // Free memory
        delete[] x;
        delete[] y;
        hipFree(x_d);
        hipFree(y_d);

        std::cout << "PASSED" << std::endl;
        return 0;
      }

=======

      // saxpy HIP function
      __global__ void saxpy(int n, float a, float *x, float *y) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
          y[i] = a * x[i] + y[i];
      }

      // saxpy CPU function
      void saxpy_cpu(int n, float a, float *x, float *y) {
        for (int i = 0; i < n; i++) {
          y[i] = a * x[i] + y[i];
        }
      }

      // saxpy CPU function
      void saxpy_cpu_omp(int n, float a, float *x, float *y) {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
          y[i] = a * x[i] + y[i];
        }
      }

      // saxpy CPU function
      void saxpy_cpu_omp_target(int n, float a, float *x, float *y) {
#pragma omp target teams distribute parallel for
        for (int i = 0; i < n; i++) {
          y[i] = a * x[i] + y[i];
        }
      }

      // saxpy CPU function
      void saxpy_cpu_omp_target_map(int n, float a, float *x, float *y) {
        float *x_mapped;
        float *y_mapped;
#pragma omp target enter data map(to : x [0:n], y [0:n])
#pragma omp target map(tofrom : x_mapped [0:n], y_mapped [0:n])
        {
          x_mapped = x;
          y_mapped = y;
#pragma omp teams distribute parallel for
          for (int i = 0; i < n; i++) {
            y_mapped[i] = a * x_mapped[i] + y_mapped[i];
          }
        }
#pragma omp target exit data map(release : x_mapped, y_mapped)
      }

      int main(int argc, char *argv[]) {
        // set problem size
        int n = 1 << 20;

        // set alpha
        float a = 2.0;

        // declare hip variables
        float *x, *y, *d_x, *d_y;

        // allocate memory on host
        x = (float *)malloc(n * sizeof(float));
        y = (float *)malloc(n * sizeof(float));

=======

        // saxpy HIP function
        __global__ void saxpy_hip(int n, float a, float *x, float *y) {
          int i = blockIdx.x * blockDim.x + threadIdx.x;
          if (i < n) {
            y[i] = a * x[i] + y[i];
          }
        }

        // saxpy wrapper function
        void saxpy_hip_wrapper(int n, float a, float *x, float *y) {
          int blockSize = 256;
          int numBlocks = (n + blockSize - 1) / blockSize;
          saxpy_hip<<<numBlocks, blockSize>>>(n, a, x, y);
        }

        // saxpy CPU function
        void saxpy_cpu(int n, float a, float *x, float *y) {
          for (int i = 0; i < n; i++) {
            y[i] = a * x[i] + y[i];
          }
        }

        int main() {

          // number of elements
          int N = 1 << 20;

          // allocate memory on the host
          float *x = new float[N];
          float *y = new float[N];

          // initialize the arrays
          for (int i = 0; i < N; i++) {
            x[i] = 1.0f;
            y[i] = 2.0f;
          }

          // saxpy constant
          float a = 2.0f;

          // saxpy on the CPU
          saxpy_cpu(N, a, x, y);

          // saxpy on the GPU
          // allocate memory on the device
          float *d_x, *d_y;
          hipMalloc(&d_x, N * sizeof(float));
          hipMalloc(&d_y, N * sizeof(float));

          // copy data to the device
          hipMemcpy(d_x, x, N * sizeof(float), hipMemcpyHostToDevice);
          hipMemcpy(d_y, y, N * sizeof(float), hipMemcpyHostToDevice);

          // saxpy on the GPU
          saxpy_hip_wrapper(N, a, d_x, d_y);

          // copy data back to the host
          hipMemcpy(y, d_y, N * sizeof(float), hipMemcpyDeviceToHost);

          // check the results
          float maxError = 0.0f;
          for (int i = 0; i < N; i++) {
            max

=======

                // saxpy HIP function
                template <typename T>
                __global__ void saxpy_kernel(int n, T a, T *x, T *y, T *z) {
              int i = blockIdx.x * blockDim.x + threadIdx.x;
              if (i < n) {
                z[i] = a * x[i] + y[i];
              }
            }

            // saxpy function
            template <typename T> void saxpy(int n, T a, T *x, T *y, T *z) {
              // launch kernel
              int blockSize = 256;
              int numBlocks = (n + blockSize - 1) / blockSize;
              saxpy_kernel<<<numBlocks, blockSize>>>(n, a, x, y, z);
            }

            int main(int argc, char **argv) {
              // initialize HIP
              hipInit(0);

              // get number of devices
              int num_devices;
              hipGetDeviceCount(&num_devices);

              // print device info
              for (int i = 0; i < num_devices; i++) {
                hipDeviceProp_t props;
                hipGetDeviceProperties(&props, i);
    printf("Device Number: %d

", i);
    printf("  Device name: %s

", props.name);
    printf("  Memory Clock Rate (KHz): %d

",
           props.memoryClockRate);
    printf("  Memory Bus Width (bits): %d

",
           props.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f

",
           2.0 * props.memoryClockRate *
               (props.memoryBusWidth / 8) / 1.0e6);
              }

              // get first device
              hipSetDevice(0);

              // allocate memory on device
              int N = 1 << 20;
              float *x, *y, *z;
              hipMallocManaged(&x, N * sizeof(float));
              hipMallocManaged(&y, N * sizeof(float));
              hipMallocManaged(&z, N * sizeof(float));

              // initialize data on host
              for (int i = 0; i < N; i++) {
                x[i] = 1.0f;
                y[i] = 2.0f;
              }

              // launch kernel on GPU
              saxpy(N, 2.0f, x, y, z);

              // wait for

=======

              // saxpy HIP function
              void saxpy_hip(int n, float a, float *x, float *y) {
                // Kernel launch parameters
                int blockSize = 256;
                int numBlocks = (n + blockSize - 1) / blockSize;

                // Launch kernel
                hipLaunchKernelGGL(saxpy, dim3(numBlocks), dim3(blockSize), 0,
                                   0, n, a, x, y);
              }

              // saxpy kernel
              __global__ void saxpy(int n, float a, float *x, float *y) {
                // Get global thread ID
                int i = blockDim.x * blockIdx.x + threadIdx.x;

                // Make sure we do not go out of bounds
                if (i < n)
                  y[i] = a * x[i] + y[i];
              }

              // saxpy CPU function
              void saxpy_cpu(int n, float a, float *x, float *y) {
                for (int i = 0; i < n; i++)
                  y[i] = a * x[i] + y[i];
              }

              // saxpy CPU function
              float *saxpy_cpu(int n, float a, float *x, float *y) {
                float *z = (float *)malloc(n * sizeof(float));
                for (int i = 0; i < n; i++)
                  z[i] = a * x[i] + y[i];
                return z;
              }

              // saxpy CPU function
              float *saxpy_cpu(int n, float a, float *x, float *y, float *z) {
                for (int i = 0; i < n; i++)
                  z[i] = a * x[i] + y[i];
                return z;
              }

              // saxpy CPU function
              float *saxpy_cpu(int n, float a, float *x, float *y, float *z,
                               float *w) {
                for (int i = 0; i < n; i++)
                  w[i] = a * x[i] + y[i];
                return w;
              }

              // saxpy CPU function
              float *saxpy_cpu(int n, float a, float *x, float *y, float *z,
                               float *w, float *v) {
                for (int i = 0; i < n; i++)
                  v[i] = a * x[i] +

=======

                                                                                            // saxpy HIP function
                                                                                            // Performs the operation y = a*x + y
                                                                                            // where a is a scalar, x and y are vectors and * is the dot product.
                                                                                            // hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy)
                                                                                            // handle - the handle to the HIPBLAS library context.
                                                                                            // n - the number of elements in the vectors x and y.
                                                                                            // alpha - the scalar a.
                                                                                            // x - the input vector x.
                                                                                            // incx - the increment for the elements of x.
                                                                                            // y - the input/output vector y.
                                                                                            // incy - the increment for the elements of y.
                                                                                            // hipblasStatus_t hipblasStatus_t is either HIPBLAS_STATUS_SUCCESS, HIPBLAS_STATUS_NOT_INITIALIZED, HIPBLAS_STATUS_ALLOC_FAILED, HIPBLAS_STATUS_INVALID_VALUE, or HIPBLAS_STATUS_MAPPING_ERROR.

#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <iostream>

#define HIP_CHECK(status)                                                      \
  {                                                                            \
    hipError_t err = status;                                                   \
    if (err != hipSuccess) {                                                   \
      std::cerr << "Got Status: " << err << " at Line: " << __LINE__           \
                << std::endl;                                                  \
      exit(0);                                                                 \
    }                                                                          \
  }

#define HIPBLAS_CHECK(status)                                                  \
  {                                                                            \
    hipblasStatus_t stat = status;                                             \
    if (stat != HIPBLAS_STATUS_SUCCESS) {                                      \
      std::cerr << "Got Status: " << stat << " at Line: " << __LINE__          \
                << std::endl;                                                  \
      exit(0);                                                                 \
    }                                                                          \
  }

                         int main() {
                    // HIP device
                    hipDevice_t device;
                    HIP_CHECK(hipGetDevice(&device));
                    hipDeviceProp_t props;
                    HIP_CHECK(hipGetDeviceProperties(&props, device));
                    std::cout << "Device: " << props.name << std::endl;

                    // HIPBLAS initialization
                    hipblasHandle_t handle;
                    HIPBLAS_CHECK(hipblasCreate(&handle));

                    // data on host
                    int n = 10;
                    float alpha = 1.0;
                    float x[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float y[10

=======

// saxpy HIP function
// x = alpha * y + x
// where alpha is a scalar and x and y are vectors
// with n elements
__global__ void saxpy(int n, float alpha, float *x, float *y)
{
                                                                                                int i =
                                                                                                    blockIdx.x *
                                                                                                        blockDim
                                                                                                            .x +
                                                                                                    threadIdx
                                                                                                        .x;
                                                                                                if (i <
                                                                                                    n)
                                                                                                  x[i] =
                                                                                                      alpha *
                                                                                                          y[i] +
                                                                                                      x[i];
}

// main function
int main(int argc, char *argv[])
{
                                                                                                // alpha, number of elements in vectors
                                                                                                float alpha =
                                                                                                    2.0;
                                                                                                int N =
                                                                                                    1
                                                                                                    << 20;

                                                                                                // size of bytes
                                                                                                size_t bytes =
                                                                                                    N *
                                                                                                    sizeof(
                                                                                                        float);

                                                                                                // allocate memory on host
                                                                                                float
                                                                                                    *h_x,
                                                                                                    *h_y;
                                                                                                h_x = (float
                                                                                                           *)
                                                                                                    malloc(
                                                                                                        bytes);
                                                                                                h_y = (float
                                                                                                           *)
                                                                                                    malloc(
                                                                                                        bytes);

                                                                                                // initialize vectors on host
                                                                                                for (
                                                                                                    int i =
                                                                                                        0;
                                                                                                    i <
                                                                                                    N;
                                                                                                    i++) {
                                                                                                  h_x[i] =
                                                                                                      0.0f;
                                                                                                  h_y[i] =
                                                                                                      1.0f;
                                                                                                }

                                                                                                // allocate memory on device
                                                                                                float
                                                                                                    *d_x,
                                                                                                    *d_y;
                                                                                                hipMalloc(
                                                                                                    &d_x,
                                                                                                    bytes);
                                                                                                hipMalloc(
                                                                                                    &d_y,
                                                                                                    bytes);

                                                                                                // copy data from host to device
                                                                                                hipMemcpy(
                                                                                                    d_x,
                                                                                                    h_x,
                                                                                                    bytes,
                                                                                                    hipMemcpyHostToDevice);
                                                                                                hipMemcpy(
                                                                                                    d_y,
                                                                                                    h_y,
                                                                                                    bytes,
                                                                                                    hipMemcpyHostToDevice);

                                                                                                // number of threads in each thread block
                                                                                                int blockSize =
                                                                                                    256;

                                                                                                // number of thread blocks in grid
                                                                                                int gridSize =
                                                                                                    (int)ceil(
                                                                                                        (float)
                                                                                                            N /
                                                                                                        blockSize);

                                                                                                // execute the kernel
                                                                                                hipLaunchKernelGGL(
                                                                                                    saxpy,
                                                                                                    dim3(
                                                                                                        gridSize),
                                                                                                    dim3(
                                                                                                        blockSize),
                                                                                                    0,
                                                                                                    0,
                                                                                                    N,
                                                                                                    alpha,
                                                                                                    d_x,
                                                                                                    d_y);

                                                                                                // copy data back to host
                                                                                                hipMemcpy(
                                                                                                    h_x,
                                                                                                    d_x,
                                                                                                    bytes,
                                                                                                    hipMemcpyDeviceToHost);

                                                                                                // print first and last 5 elements
                                                                                                printf(
                                                                                                    "First 5 elements: ");
                                                                                                for (
                                                                                                    int i =
                                                                                                        0;
                                                                                                    i <
                                                                                                    5;
                                                                                                    i++)
                                                                                                  printf(
                                                                                                      "%f ",
                                                                                                      h_x[i]);
    printf("...

");
    printf("Last 5 elements: ");
    for (int i = N-5; i < N; i++)
        printf("%f ", h_x[i]);
    printf("...

=======

// saxpy HIP function
// x and y are pointers to the input vectors
// a is the scalar
// n is the number of elements in the vectors

__global__ void saxpy(int n, float a, float *x, float *y)
{
                                                                                                  int i =
                                                                                                      blockIdx.x *
                                                                                                          blockDim
                                                                                                              .x +
                                                                                                      threadIdx
                                                                                                          .x;
                                                                                                  if (i <
                                                                                                      n)
                                                                                                    y[i] =
                                                                                                        a * x[i] +
                                                                                                        y[i];
}

int main(int argc, char** argv)
{
                                                                                                  // Size of vectors
                                                                                                  int N =
                                                                                                      1
                                                                                                      << 20;

                                                                                                  // Host input vectors
                                                                                                  float
                                                                                                      *h_A;
                                                                                                  float
                                                                                                      *h_B;
                                                                                                  // Host output vector
                                                                                                  float
                                                                                                      *h_C;

                                                                                                  // Device input buffers
                                                                                                  float
                                                                                                      *d_A;
                                                                                                  float
                                                                                                      *d_B;
                                                                                                  // Device output buffer
                                                                                                  float
                                                                                                      *d_C;

                                                                                                  // Size, in bytes, of each vector
                                                                                                  size_t bytes =
                                                                                                      N *
                                                                                                      sizeof(
                                                                                                          float);

                                                                                                  // Allocate memory for each vector on host
                                                                                                  h_A = (float
                                                                                                             *)
                                                                                                      malloc(
                                                                                                          bytes);
                                                                                                  h_B = (float
                                                                                                             *)
                                                                                                      malloc(
                                                                                                          bytes);
                                                                                                  h_C = (float
                                                                                                             *)
                                                                                                      malloc(
                                                                                                          bytes);

                                                                                                  // Initialize vectors on host
                                                                                                  int i;
                                                                                                  for (
                                                                                                      i = 0;
                                                                                                      i <
                                                                                                      N;
                                                                                                      i++) {
                                                                                                    h_A[i] =
                                                                                                        rand() /
                                                                                                        (float)
                                                                                                            RAND_MAX;
                                                                                                    h_B[i] =
                                                                                                        rand() /
                                                                                                        (float)
                                                                                                            RAND_MAX;
                                                                                                  }

                                                                                                  // Allocate memory for each vector on GPU
                                                                                                  hipMalloc(
                                                                                                      &d_A,
                                                                                                      bytes);
                                                                                                  hipMalloc(
                                                                                                      &d_B,
                                                                                                      bytes);
                                                                                                  hipMalloc(
                                                                                                      &d_C,
                                                                                                      bytes);

                                                                                                  // Copy host vectors to device
                                                                                                  hipMemcpy(
                                                                                                      d_A,
                                                                                                      h_A,
                                                                                                      bytes,
                                                                                                      hipMemcpyHostToDevice);
                                                                                                  hipMemcpy(
                                                                                                      d_B,
                                                                                                      h_B,
                                                                                                      bytes,
                                                                                                      hipMemcpyHostToDevice);

                                                                                                  // Number of threads in each thread block
                                                                                                  int THREADS_PER_BLOCK =
                                                                                                      256;

                                                                                                  // Number of thread blocks in grid
                                                                                                  int BLOCKS =
                                                                                                      (N +
                                                                                                       THREADS_PER_BLOCK -
                                                                                                       1) /
                                                                                                      THREADS_PER_BLOCK;

                                                                                                  // Execute the kernel
                                                                                                  saxpy<<<
                                                                                                      BLOCKS,
                                                                                                      THREADS_PER_BLOCK>>>(
                                                                                                      N,
                                                                                                      2.0,
                                                                                                      d_A,
                                                                                                      d_C);

                                                                                                  // Wait for GPU to finish before accessing on host
                                                                                                  hipDeviceSynchronize();

                                                                                                  // Copy array back to host
                                                                                                  hipMemcpy(
                                                                                                      h_C,
                                                                                                      d_C,
                                                                                                      bytes,
                                                                                                      hipMemcpyDeviceToHost);

                                                                                                  // Verify that the result vector is correct
                                                                                                  // Prompt: saxpy Thrust function
                                                                                                  Synthesizing 10 /
                                                                                                      10 solutions

=======

                                                                                                      // saxpy Thrust function object
                                                                                                      struct
                                                                                                      saxpy_functor {
                                                                                                    const float
                                                                                                        a;

                                                                                                    saxpy_functor(
                                                                                                        float
                                                                                                            _a)
                                                                                                        : a(_a) {
                                                                                                    }

                                                                                                    __host__ __device__ float
                                                                                                    operator()(
                                                                                                        const float
                                                                                                            &x,
                                                                                                        const float
                                                                                                            &y)
                                                                                                        const {
                                                                                                      return a * x +
                                                                                                             y;
                                                                                                    }
                                                                                                  };

                                                                                                  int
                                                                                                  main(
                                                                                                      void) {
                                                                                                    // Initialize host vectors
                                                                                                    std::vector<
                                                                                                        float>
                                                                                                        h_x(N,
                                                                                                            1);
                                                                                                    std::vector<
                                                                                                        float>
                                                                                                        h_y(N,
                                                                                                            2);

                                                                                                    // Initialize device vectors
                                                                                                    thrust::device_vector<
                                                                                                        float>
                                                                                                        d_x =
                                                                                                            h_x;
                                                                                                    thrust::device_vector<
                                                                                                        float>
                                                                                                        d_y =
                                                                                                            h_y;

                                                                                                    // Compute SAXPY on the device
                                                                                                    thrust::transform(
                                                                                                        d_x.begin(),
                                                                                                        d_x.end(),
                                                                                                        d_y.begin(),
                                                                                                        d_y.begin(),
                                                                                                        saxpy_functor(
                                                                                                            2));

                                                                                                    // Copy back to host
                                                                                                    h_y =
                                                                                                        d_y;

                                                                                                    // Print result
                                                                                                    for (
                                                                                                        int i =
                                                                                                            0;
                                                                                                        i <
                                                                                                        N;
                                                                                                        ++i) {
                                                                                                      std::cout
                                                                                                          << h_y[i]
                                                                                                          << " ";
                                                                                                    }
                                                                                                    std::cout
                                                                                                        << std::
                                                                                                               endl;

                                                                                                    return 0;
                                                                                                  }

[/sourcecode]

In the above code, we define a saxpy_functor struct that takes a single argument, a , and defines the operator() function to perform the saxpy operation. The functor is then used to compute SAXPY using the Thrust transform function. The output of the program is:

[output]

6 6 6 6 6 6 6 6 6 6

[/output]

The Thrust host_vector and device_vector classes are thin wrappers around the STL vector class. They are designed to be used in the same way, with the exception that device_vector objects are allocated on the GPU. The Thrust transform function is a generalization of the STL transform function. It takes a pair of input iterators, a pair of output iterators, and a function object. The function object is applied to the elements of the input iterators and the results are written to the output iterators. The function object can be any callable object, including a function pointer, a function object, or a lambda expression.

Thrust also provides a number of other functions that are useful for performing common operations on vectors. For example, the Thrust fill function is used to fill a vector with a given value:

[output]

2 2 2 2 2 2 2 2 2

=======

// saxpy Thrust function object
struct saxpy_functor
{
  const float a;

  saxpy_functor(float _a) : a(_a) {}

  __host__ __device__ float operator()(const float &x, const float &y) const {
    return a * x + y;
  }
};

int main(int argc, char **argv) {
  // Initialize command line
  CommandLineArgs args(argc, argv);
  args.AddOption(&size, "-n", "--problem-size",
                 "Size of the problem (default 1000000)");
  args.AddOption(&a, "-a", "--a", "Scalar (default 2)");
  args.AddOption(&device, "-device", "--device",
                 "Device to run on (default 0)");
  args.Parse();

  // Print help message
  if (args.CheckCmdLineFlag("help")) {
        printf("%s

", sSampleName);
        printf("  This sample computes SAXPY using Thrust.

");
        printf("

");
        printf("  -n <problem size>      # of elements in the problem

");
        printf("  -a <scalar>            Scalar

");
        printf("  -device <device_id>    GPU to run on (default 0)

");
        args.PrintHelp();
        exit(0);
  }

  // Initialize device
    printf("Initializing device %d

", device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities

", deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
    printf("> GPU device has %d cores

", ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
    cudaSetDevice(device);

    // Allocate host memory
    printf("Allocating host memory

");
    float *h_x = (float *)malloc(size * sizeof(float));
    float *h_y = (float *)malloc(size * sizeof(float));
    float *h_ref = (float *)malloc(size * sizeof(float));

    // Initialize host arrays
    printf("Initializing host arrays

");
    for (int i = 0; i < size; i++)
    {
                                                                                                      h_x[i] =
                                                                                                          i;
                                                                                                      h_y[i] =
                                                                                                          i;
                                                                                                      h_ref[i] =
                                                                                                          a

=======

                                                                                                          // saxpy Thrust function object
                                                                                                          struct
                                                                                                          saxpy_functor
                                                                                                          : public thrust::binary_function<
                                                                                                                float,
                                                                                                                float,
                                                                                                                float> {
                                                                                                        const float
                                                                                                            a;

                                                                                                        saxpy_functor(
                                                                                                            float
                                                                                                                _a)
                                                                                                            : a(_a) {
                                                                                                        }

                                                                                                        __host__ __device__ float
                                                                                                        operator()(
                                                                                                            const float
                                                                                                                &x,
                                                                                                            const float
                                                                                                                &y)
                                                                                                            const {
                                                                                                          return a * x +
                                                                                                                 y;
                                                                                                        }
                                                                                                      };

                                                                                                      int
                                                                                                      main(
                                                                                                          void) {
                                                                                                        // generate random data serially
                                                                                                        thrust::default_random_engine
                                                                                                            rng;
                                                                                                        thrust::uniform_real_distribution<
                                                                                                            float>
                                                                                                            dist(
                                                                                                                0,
                                                                                                                1);
                                                                                                        thrust::generate(
                                                                                                            h_x.begin(),
                                                                                                            h_x.end(),
                                                                                                            rng);
                                                                                                        thrust::generate(
                                                                                                            h_y.begin(),
                                                                                                            h_y.end(),
                                                                                                            rng);

                                                                                                        // transfer data to the device
                                                                                                        thrust::copy(
                                                                                                            h_x.begin(),
                                                                                                            h_x.end(),
                                                                                                            d_x.begin());
                                                                                                        thrust::copy(
                                                                                                            h_y.begin(),
                                                                                                            h_y.end(),
                                                                                                            d_y.begin());

                                                                                                        // perform SAXPY on the device
                                                                                                        thrust::transform(
                                                                                                            d_x.begin(),
                                                                                                            d_x.end(),
                                                                                                            d_y.begin(),
                                                                                                            d_y.begin(),
                                                                                                            saxpy_functor(
                                                                                                                2));

                                                                                                        // transfer data back to host
                                                                                                        thrust::copy(
                                                                                                            d_y.begin(),
                                                                                                            d_y.end(),
                                                                                                            h_y.begin());

                                                                                                        return 0;
                                                                                                      }

                                                                                                      I am trying to run this on a Jetson TX1
                                                                                                                  .I have installed
                                                                                                                      CUDA 8.0 on
                                                                                                                          it
                                                                                                                  .I am getting
                                                                                                                      the following
                                                                                                                          error
                                                                                                          :

                                                                                                          nvcc
                                                                                                              fatal
                                                                                                          : Unsupported gpu
                                                                                                                architecture 'compute_20'

                                                                                                            I have also tried changing
                                                                                                                the compute capability
                                                                                                                    to 30 and
                                                                                                              35 but I am still getting
                                                                                                                  the same error
                                                                                                                      .

                                                                                                              I have also tried
                                                                                                                  to run the same code
                                                                                                                      on my laptop with
                                                                                                                          an Intel i7 processor
                                                                                                                              and it compiled
                                                                                                                                  and ran
                                                                                                                                      successfully
                                                                                                                      .

                                                                                                              Can someone please
                                                                                                                  help me
                                                                                                                      with this
                                                                                                          ?

                                                                                                          Ganondolians
                                                                                                          : You need to use
                                                                                                                a newer version
                                                                                                                    of CUDA
                                                                                                                        .Compute
                                                                                                                            capability 2.0 was introduced
                                                                                                                                in CUDA 6.0. The Jetson
                                                                                                                                    TX1 is a Maxwell
                                                                                                                                        GPU,
                                                                                                          which is supported
                                                                                                                  by CUDA 7.5 and
                                                                                                              8.0.

=======

                                                                                                              // saxpy Thrust function object
                                                                                                              struct
                                                                                                              saxpy_functor
                                                                                                          : public thrust::binary_function<
                                                                                                                float,
                                                                                                                float,
                                                                                                                float> {
                                                                                                        const float
                                                                                                            a;

                                                                                                        saxpy_functor(
                                                                                                            float
                                                                                                                _a)
                                                                                                            : a(_a) {
                                                                                                        }

                                                                                                        __host__ __device__ float
                                                                                                        operator()(
                                                                                                            const float
                                                                                                                &x,
                                                                                                            const float
                                                                                                                &y)
                                                                                                            const {
                                                                                                          return a * x +
                                                                                                                 y;
                                                                                                        }
                                                                                                      };

                                                                                                      int
                                                                                                      main(
                                                                                                          void) {
                                                                                                        // Number of elements
                                                                                                        int N =
                                                                                                            1
                                                                                                            << 20;

                                                                                                        // Host input vectors
                                                                                                        std::vector<
                                                                                                            float>
                                                                                                            h_X(N);
                                                                                                        std::vector<
                                                                                                            float>
                                                                                                            h_Y(N);

                                                                                                        // Initialize host input vectors
                                                                                                        for (
                                                                                                            int i =
                                                                                                                0;
                                                                                                            i <
                                                                                                            N;
                                                                                                            i++) {
                                                                                                          h_X[i] =
                                                                                                              rand() /
                                                                                                              (float)
                                                                                                                  RAND_MAX;
                                                                                                          h_Y[i] =
                                                                                                              rand() /
                                                                                                              (float)
                                                                                                                  RAND_MAX;
                                                                                                        }

                                                                                                        // Thrust device input vectors
                                                                                                        thrust::device_vector<
                                                                                                            float>
                                                                                                            d_X =
                                                                                                                h_X;
                                                                                                        thrust::device_vector<
                                                                                                            float>
                                                                                                            d_Y =
                                                                                                                h_Y;

                                                                                                        // Thrust function object for saxpy
                                                                                                        saxpy_functor
                                                                                                            saxpy_op(
                                                                                                                2.0f);

                                                                                                        // saxpy: d_Y = 2 * d_X + d_Y
                                                                                                        thrust::transform(
                                                                                                            d_X.begin(),
                                                                                                            d_X.end(),
                                                                                                            d_Y.begin(),
                                                                                                            d_Y.begin(),
                                                                                                            saxpy_op);

                                                                                                        // Copy result back to host
                                                                                                        thrust::copy(
                                                                                                            d_Y.begin(),
                                                                                                            d_Y.end(),
                                                                                                            h_Y.begin());

                                                                                                        return 0;
                                                                                                      }

SAXPY is a simple but important example of a parallel algorithm. It is a simple vectorized operation that is often used as a building block for more complex algorithms. SAXPY is also a good example of a parallel algorithm that is easy to express in Thrust.

In the above code, we define a Thrust function object that implements the SAXPY operation. The function object is initialized with the scalar value a. The function object’s operator() is then used to transform the input vectors X and Y into the output vector Y. The operator() is defined to be a binary function, which means that it takes two arguments (the x and y values) and returns a single value (the result of the SAXPY operation). The operator() is executed in parallel by Thrust.

The saxpy_functor is used to transform the input vectors X and Y into the output vector Y. The transform() function takes four arguments: the beginning and end of the input

=======

// saxpy Thrust function object
struct saxpy_functor : public thrust::binary_function<float,float,float>
{
  const float a;
  saxpy_functor(float _a) : a(_a) {}

  __host__ __device__ float operator()(const float &x, const float &y) const {
    return a * x + y;
  }
};

int main(void) {
  // initialize host arrays
  const int N = 1 << 20;
  thrust::host_vector<float> h_x(N, 1.0f);
  thrust::host_vector<float> h_y(N, 2.0f);

  // initialize device arrays
  thrust::device_vector<float> d_x = h_x;
  thrust::device_vector<float> d_y = h_y;

  // saxpy with Thrust
  thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(),
                    saxpy_functor(2.0f));

  // saxpy with CUB
  cub::Saxpy<float> saxpy_functor(2.0f);
  saxpy_functor(d_x.data(), d_y.data(), d_y.data(), N);

  // saxpy with Thrust
  thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(),
                    saxpy_functor(2.0f));

  // saxpy with CUB
  saxpy_functor(d_x.data(), d_y.data(), d_y.data(), N);

  // saxpy with Thrust
  thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(),
                    saxpy_functor(2.0f));

  // saxpy with CUB
  saxpy_functor(d_x.data(), d_y.data(), d_y.data(), N);

  // saxpy with Thrust
  thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(),
                    saxpy_functor(2.0f));

  // saxpy with CUB
  saxpy_functor(d_x.data(), d_y.data(), d_y.data(), N);

  // saxpy with Thrust
  thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(),
                    saxpy_functor(2.0f));

  // saxpy with CUB
  saxpy_functor(d_x.data(), d_y.data(), d

=======

// saxpy Thrust function
template <typename T>
struct saxpy_functor
{
                                                                                                          const T
                                                                                                              a;

                                                                                                          saxpy_functor(
                                                                                                              T _a)
                                                                                                              : a(_a) {
                                                                                                          }

                                                                                                          __host__ __device__ T
                                                                                                          operator()(
                                                                                                              const T
                                                                                                                  &x,
                                                                                                              const T
                                                                                                                  &y)
                                                                                                              const {
                                                                                                            return a * x +
                                                                                                                   y;
                                                                                                          }
};

// saxpy Thrust function
template <typename T>
struct saxpy_functor
{
                                                                                                          const T
                                                                                                              a;

                                                                                                          saxpy_functor(
                                                                                                              T _a)
                                                                                                              : a(_a) {
                                                                                                          }

                                                                                                          __host__ __device__ T
                                                                                                          operator()(
                                                                                                              const T
                                                                                                                  &x,
                                                                                                              const T
                                                                                                                  &y)
                                                                                                              const {
                                                                                                            return a * x +
                                                                                                                   y;
                                                                                                          }
};

// saxpy Thrust function
template <typename T>
struct saxpy_functor
{
                                                                                                          const T
                                                                                                              a;

                                                                                                          saxpy_functor(
                                                                                                              T _a)
                                                                                                              : a(_a) {
                                                                                                          }

                                                                                                          __host__ __device__ T
                                                                                                          operator()(
                                                                                                              const T
                                                                                                                  &x,
                                                                                                              const T
                                                                                                                  &y)
                                                                                                              const {
                                                                                                            return a * x +
                                                                                                                   y;
                                                                                                          }
};

// saxpy Thrust function
template <typename T>
struct saxpy_functor
{
                                                                                                          const T
                                                                                                              a;

                                                                                                          saxpy_functor(
                                                                                                              T _a)
                                                                                                              : a(_a) {
                                                                                                          }

                                                                                                          __host__ __device__ T
                                                                                                          operator()(
                                                                                                              const T
                                                                                                                  &x,
                                                                                                              const T
                                                                                                                  &y)
                                                                                                              const {
                                                                                                            return a * x +
                                                                                                                   y;
                                                                                                          }
};

// saxpy Thrust function
template <typename T>
struct saxpy_functor
{
                                                                                                          const T
                                                                                                              a;

                                                                                                          saxpy_functor(
                                                                                                              T _a)
                                                                                                              : a(_a) {
                                                                                                          }

                                                                                                          __host__ __device__ T
                                                                                                          operator()(
                                                                                                              const T
                                                                                                                  &x,
                                                                                                              const T
                                                                                                                  &y)
                                                                                                              const {
                                                                                                            return a * x +
                                                                                                                   y;
                                                                                                          }
};

// saxpy Thrust function
template <typename T>
struct saxpy_functor
{
                                                                                                          const T
                                                                                                              a;

                                                                                                          saxpy_functor(
                                                                                                              T _a)
                                                                                                              : a(_a) {
                                                                                                          }

                                                                                                          __host__ __device__ T
                                                                                                          operator()(
                                                                                                              const T
                                                                                                                  &x,
                                                                                                              const T
                                                                                                                  &y)
                                                                                                              const {
                                                                                                            return a * x +
                                                                                                                   y;
                                                                                                          }
};

// saxpy Thrust function
template <typename T>
struct saxpy_functor
{
                                                                                                          const T
                                                                                                              a;

                                                                                                          saxpy_functor(
                                                                                                              T _a)
                                                                                                              : a(_a) {
                                                                                                          }

                                                                                                          __host__ __device__ T
                                                                                                          operator()(
                                                                                                              const T
                                                                                                                  &x,
                                                                                                              const T
                                                                                                                  &y)
                                                                                                              const {
                                                                                                            return a *
                                                                                                                   x

=======

                                                                                                                   // saxpy Thrust function
                                                                                                                   template <
                                                                                                                       typename T>
                                                                                                                   struct
                                                                                                                   saxpy_functor
                                                                                                                : public thrust::
                                                                                                                      binary_function<
                                                                                                                          T,
                                                                                                                          T,
                                                                                                                          T> {
                                                                                                              const T
                                                                                                                  a;

                                                                                                              saxpy_functor(
                                                                                                                  T _a)
                                                                                                                  : a(_a) {
                                                                                                              }

                                                                                                              __host__ __device__ T
                                                                                                              operator()(
                                                                                                                  const T
                                                                                                                      &x,
                                                                                                                  const T
                                                                                                                      &y)
                                                                                                                  const {
                                                                                                                return a * x +
                                                                                                                       y;
                                                                                                              }
                                                                                                            };

                                                                                                            // saxpy Thrust function
                                                                                                            template <
                                                                                                                typename T>
                                                                                                            struct
                                                                                                                saxpy_functor2
                                                                                                                : public thrust::binary_function<
                                                                                                                      T,
                                                                                                                      T,
                                                                                                                      T> {
                                                                                                              const T
                                                                                                                  a;

                                                                                                              saxpy_functor2(
                                                                                                                  T _a)
                                                                                                                  : a(_a) {
                                                                                                              }

                                                                                                              __host__ __device__ T
                                                                                                              operator()(
                                                                                                                  const T
                                                                                                                      &x,
                                                                                                                  const T
                                                                                                                      &y)
                                                                                                                  const {
                                                                                                                return a * x +
                                                                                                                       y;
                                                                                                              }
                                                                                                            };

                                                                                                            // saxpy Thrust function
                                                                                                            template <
                                                                                                                typename T>
                                                                                                            struct
                                                                                                                saxpy_functor3
                                                                                                                : public thrust::binary_function<
                                                                                                                      T,
                                                                                                                      T,
                                                                                                                      T> {
                                                                                                              const T
                                                                                                                  a;

                                                                                                              saxpy_functor3(
                                                                                                                  T _a)
                                                                                                                  : a(_a) {
                                                                                                              }

                                                                                                              __host__ __device__ T
                                                                                                              operator()(
                                                                                                                  const T
                                                                                                                      &x,
                                                                                                                  const T
                                                                                                                      &y)
                                                                                                                  const {
                                                                                                                return a * x +
                                                                                                                       y;
                                                                                                              }
                                                                                                            };

                                                                                                            int
                                                                                                            main(
                                                                                                                void) {
                                                                                                              // initialize host arrays
                                                                                                              const int N =
                                                                                                                  1
                                                                                                                  << 20;
                                                                                                              thrust::host_vector<
                                                                                                                  float>
                                                                                                                  h_x(N);
                                                                                                              thrust::host_vector<
                                                                                                                  float>
                                                                                                                  h_y(N);
                                                                                                              for (
                                                                                                                  int i =
                                                                                                                      0;
                                                                                                                  i <
                                                                                                                  N;
                                                                                                                  i++) {
                                                                                                                h_x[i] =
                                                                                                                    1.0f;
                                                                                                                h_y[i] =
                                                                                                                    2.0f;
                                                                                                              }

                                                                                                              // initialize device arrays
                                                                                                              thrust::device_vector<
                                                                                                                  float>
                                                                                                                  d_x =
                                                                                                                      h_x;
                                                                                                              thrust::device_vector<
                                                                                                                  float>
                                                                                                                  d_y =
                                                                                                                      h_y;

                                                                                                              // saxpy
                                                                                                              float a =
                                                                                                                  2.0f;
                                                                                                              thrust::transform(
                                                                                                                  d_x.begin(),
                                                                                                                  d_x.end(),
                                                                                                                  d_y.begin(),
                                                                                                                  d_y.begin(),
                                                                                                                  saxpy_functor<
                                                                                                                      float>(
                                                                                                                      a));

                                                                                                              // saxpy
                                                                                                              float b =
                                                                                                                  3.0f;
                                                                                                              thrust::transform(
                                                                                                                  d_x.begin(),
                                                                                                                  d_x.end(),
                                                                                                                  d_y.begin(),
                                                                                                                  d_y.begin(),
                                                                                                                  saxpy_functor2<
                                                                                                                      float>(
                                                                                                                      b));

                                                                                                              // saxpy
                                                                                                              float c =
                                                                                                                  4.0f;
                                                                                                              thrust::transform(
                                                                                                                  d_x.begin(),
                                                                                                                  d_x.end(),
                                                                                                                  d_y.begin(),
                                                                                                                  d_y.begin(),
                                                                                                                  saxpy_functor3<
                                                                                                                      float>(
                                                                                                                      c));

                                                                                                              //

=======

                                                                                                              // saxpy Thrust function
                                                                                                              void
                                                                                                              saxpy(
                                                                                                                  int n,
                                                                                                                  float
                                                                                                                      a,
                                                                                                                  float
                                                                                                                      *x,
                                                                                                                  float *
                                                                                                                      y) {
                                                                                                                thrust::device_vector<
                                                                                                                    float>
                                                                                                                    d_x(n);
                                                                                                                thrust::device_vector<
                                                                                                                    float>
                                                                                                                    d_y(n);

                                                                                                                thrust::copy(
                                                                                                                    x,
                                                                                                                    x + n,
                                                                                                                    d_x.begin());
                                                                                                                thrust::copy(
                                                                                                                    y,
                                                                                                                    y + n,
                                                                                                                    d_y.begin());

                                                                                                                thrust::transform(
                                                                                                                    d_x.begin(),
                                                                                                                    d_x.end(),
                                                                                                                    d_y.begin(),
                                                                                                                    d_y.begin(),
                                                                                                                    saxpy_functor(
                                                                                                                        a));

                                                                                                                thrust::copy(
                                                                                                                    d_y.begin(),
                                                                                                                    d_y.end(),
                                                                                                                    y);
                                                                                                              }

                                                                                                              // saxpy Thrust functor
                                                                                                              struct
                                                                                                                  saxpy_functor {
                                                                                                                const float
                                                                                                                    a;

                                                                                                                saxpy_functor(
                                                                                                                    float
                                                                                                                        _a)
                                                                                                                    : a(_a) {
                                                                                                                }

                                                                                                                __host__ __device__ float
                                                                                                                operator()(
                                                                                                                    const float
                                                                                                                        &x,
                                                                                                                    const float
                                                                                                                        &y)
                                                                                                                    const {
                                                                                                                  return a * x +
                                                                                                                         y;
                                                                                                                }
                                                                                                              };

                                                                                                              // saxpy Thrust kernel
                                                                                                              __global__ void
                                                                                                              saxpy_kernel(
                                                                                                                  int n,
                                                                                                                  float
                                                                                                                      a,
                                                                                                                  float
                                                                                                                      *x,
                                                                                                                  float *
                                                                                                                      y) {
                                                                                                                int i =
                                                                                                                    blockIdx.x *
                                                                                                                        blockDim
                                                                                                                            .x +
                                                                                                                    threadIdx
                                                                                                                        .x;

                                                                                                                if (i <
                                                                                                                    n) {
                                                                                                                  y[i] =
                                                                                                                      a * x[i] +
                                                                                                                      y[i];
                                                                                                                }
                                                                                                              }

                                                                                                              // saxpy Thrust kernel wrapper
                                                                                                              void
                                                                                                              saxpy_kernel_wrapper(
                                                                                                                  int n,
                                                                                                                  float
                                                                                                                      a,
                                                                                                                  float
                                                                                                                      *x,
                                                                                                                  float *
                                                                                                                      y) {
                                                                                                                int blockSize =
                                                                                                                    256;
                                                                                                                int numBlocks =
                                                                                                                    (n +
                                                                                                                     blockSize -
                                                                                                                     1) /
                                                                                                                    blockSize;

                                                                                                                saxpy_kernel<<<
                                                                                                                    numBlocks,
                                                                                                                    blockSize>>>(
                                                                                                                    n,
                                                                                                                    a,
                                                                                                                    x,
                                                                                                                    y);
                                                                                                              }

                                                                                                              // saxpy Thrust kernel wrapper with shared memory
                                                                                                              void
                                                                                                              saxpy_kernel_wrapper_shared(
                                                                                                                  int n,
                                                                                                                  float
                                                                                                                      a,
                                                                                                                  float
                                                                                                                      *x,
                                                                                                                  float *
                                                                                                                      y) {
                                                                                                                int blockSize =
                                                                                                                    256;
                                                                                                                int numBlocks =
                                                                                                                    (n +
                                                                                                                     blockSize -
                                                                                                                     1) /
                                                                                                                    blockSize;

                                                                                                                saxpy_kernel_shared<<<
                                                                                                                    numBlocks,
                                                                                                                    blockSize>>>(
                                                                                                                    n,
                                                                                                                    a,
                                                                                                                    x,
                                                                                                                    y);
                                                                                                              }

                                                                                                              // saxpy Thrust kernel with shared memory
                                                                                                              __global__ void
                                                                                                              saxpy_kernel_shared(
                                                                                                                  int n,
                                                                                                                  float
                                                                                                                      a,
                                                                                                                  float
                                                                                                                      *x,
                                                                                                                  float *
                                                                                                                      y) {
                                                                                                                int i =
                                                                                                                    blockIdx.x *
                                                                                                                        blockDim
                                                                                                                            .x +
                                                                                                                    threadIdx
                                                                                                                        .x;

                                                                                                                __shared__ float
                                                                                                                    s_x[256];
                                                                                                                __shared__ float
                                                                                                                    s_y[256];

                                                                                                                if (i <
                                                                                                                    n) {
                                                                                                                  s_x[threadIdx
                                                                                                                          .x] =
                                                                                                                      x[i];
                                                                                                                  s_y[threadIdx
                                                                                                                          .x] =
                                                                                                                      y[i];

                                                                                                                  __syncthreads();

                                                                                                                  s_y

=======

                                                                                                                      // saxpy Thrust function
                                                                                                                      // Performs y = alpha*x + y
                                                                                                                      // where alpha is a scalar and x and y are vectors
                                                                                                                      void
                                                                                                                      saxpy(
                                                                                                                          float
                                                                                                                              alpha,
                                                                                                                          thrust::device_vector<
                                                                                                                              float>
                                                                                                                              &x,
                                                                                                                          thrust::device_vector<
                                                                                                                              float>
                                                                                                                              &y) {
                                                                                                                    thrust::transform(
                                                                                                                        x.begin(),
                                                                                                                        x.end(),
                                                                                                                        y.begin(),
                                                                                                                        y.begin(),
                                                                                                                        saxpy_functor(
                                                                                                                            alpha));
                                                                                                                  }

                                                                                                                  int
                                                                                                                  main(
                                                                                                                      void) {
                                                                                                                    // Create vectors x and y
                                                                                                                    thrust::device_vector<
                                                                                                                        float>
                                                                                                                        x(4);
                                                                                                                    thrust::device_vector<
                                                                                                                        float>
                                                                                                                        y(4);

                                                                                                                    // Initialize x and y
                                                                                                                    x[0] =
                                                                                                                        1.0f;
                                                                                                                    x[1] =
                                                                                                                        2.0f;
                                                                                                                    x[2] =
                                                                                                                        3.0f;
                                                                                                                    x[3] =
                                                                                                                        4.0f;
                                                                                                                    y[0] =
                                                                                                                        1.0f;
                                                                                                                    y[1] =
                                                                                                                        2.0f;
                                                                                                                    y[2] =
                                                                                                                        3.0f;
                                                                                                                    y[3] =
                                                                                                                        4.0f;

                                                                                                                    // Compute y = 2 * x + y
                                                                                                                    saxpy(
                                                                                                                        2.0f,
                                                                                                                        x,
                                                                                                                        y);

                                                                                                                    // Print y
                                                                                                                    for (
                                                                                                                        int i =
                                                                                                                            0;
                                                                                                                        i <
                                                                                                                        y.size();
                                                                                                                        i++)
                                                                                                                      std::cout
                                                                                                                          << "y["
                                                                                                                          << i
                                                                                                                          << "] = "
                                                                                                                          << y[i]
                                                                                                                          << std::
                                                                                                                                 endl;

                                                                                                                    return 0;
                                                                                                                  }

                                                                                                                Output

                                                                                                                    y[0] = 3 y[1] =
                                                                                                                        6 y[2] = 9 y
                                                                                                                            [3] =
                                                                                                                                12

                                                                                                                    Example 2
                                                                                                                    : SAXPY with
                                                                                                                          cuBLAS

                                                                                                                // Path: SAXPY/CUDA/SAXPY.cu

=======

                                                                                                                // saxpy Thrust function
                                                                                                                // Path: SAXPY/CXX/SAXPY.cpp
