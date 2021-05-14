/*
 * Cooley-Tukey Algorithm for Fast Fourier Transform
 */

#include "fft.h"
#include "fft_ispc.h"
#include "pthread.h"
#include "stdbool.h"
#include "stdio.h"
#include "stdlib.h"

#include "omp.h"

#define GRANULARITY 0

double LUTS4[16];
double LUTC4[16];

double *twiddles_r;
double *twiddles_c;

double x8_r[8];
double x8_c[8];

void fft(double complex *A, int N) {
  if (N <= 1) {
    return;
  }

  // split based on parity of index
  double complex *E = (double complex *)calloc(N / 2, sizeof(double complex));
  double complex *O = (double complex *)calloc(N / 2, sizeof(double complex));
  for (int i = 0; i < N / 2; i++) {
    E[i] = A[2 * i];
    O[i] = A[2 * i + 1];
  }

  fft(E, N / 2);
  fft(O, N / 2);

  for (int k = 0; k < N / 2; k++) {
    double complex t = O[k] * cexp(I * ((-2.0 * M_PI * k) / N));
    A[k] = E[k] + t;
    A[k + (N / 2)] = E[k] - t;
  }
  free(O);
  free(E);
}

// SIMD version with near in place implemenation
void fft_simd_out_body(double *x_r, double *x_c, double *X_r, double *X_c, int s, int N) {
  if (N == 4) {
    fft_ispc_four_out(X_r, X_c, x_r, x_c, LUTS4, LUTC4, s);
    return;
  }

  fft_simd_out_body(x_r, x_c, X_r, X_c, 2 * s, N / 2);
  fft_simd_out_body(x_r + s, x_c + s, X_r + (N / 2), X_c + (N / 2), 2 * s, N / 2);

  if (s >= (1 << GRANULARITY)) {
    fft_ispc_out(X_r, X_c, twiddles_r + (N / 2), twiddles_c + (N / 2), N);
  } else {
    fft_ispc_out_raw(X_r, X_c, N);
  }
}

// SIMD version with near in place implemenation
void fft_simd_out_body_par(double *x_r, double *x_c, double *X_r, double *X_c, int s, int N, int nproc) {
  if (N == 4) {
    fft_ispc_four_out(X_r, X_c, x_r, x_c, LUTS4, LUTC4, s);
    return;
  }

  if (nproc == 1) {
    fft_simd_out_body(x_r, x_c, X_r, X_c, s, N);
    return;
  }

#pragma omp task
  {
    fft_simd_out_body_par(x_r, x_c, X_r, X_c, 2 * s, N / 2, nproc / 2);
  }
#pragma omp task
  {
    fft_simd_out_body_par(x_r + s, x_c + s, X_r + (N / 2), X_c + (N / 2), 2 * s, N / 2, nproc / 2);
  }

#pragma omp taskwait

  if (s >= (1 << GRANULARITY)) {
    fft_ispc_out(X_r, X_c, twiddles_r + (N / 2), twiddles_c + (N / 2), N);
  } else {
    fft_ispc_out_raw(X_r, X_c, N);
  }
}

// SIMD version with near in place implemenation cached better
void fft_simd_out_body_cache(int start, double *x_r, double *x_c, double *X_r, double *X_c, int s, int N) {
  if (N == 4) {
    fft_ispc_four_out_cache(X_r, X_c, x_r, x_c, LUTS4, LUTC4, start);
    return;
  }

  fft_simd_out_body_cache(start, x_r, x_c, X_r, X_c, 2 * s, N / 2);
  fft_simd_out_body_cache(start + s, x_r, x_c, X_r + (N / 2), X_c + (N / 2), 2 * s, N / 2);

  if (s >= (1 << GRANULARITY)) {
    fft_ispc_out(X_r, X_c, twiddles_r + (N / 2), twiddles_c + (N / 2), N);
  } else {
    fft_ispc_out_raw(X_r, X_c, N);
  }
}

// SIMD version with near in place implemenation with parallelism, cached better
void fft_simd_out_body_par_cache(int start, double *x_r, double *x_c, double *X_r, double *X_c, int s, int N, int nproc) {
  if (N == 4) {
    fft_ispc_four_out_cache(X_r, X_c, x_r, x_c, LUTS4, LUTC4, start);
    return;
  }

  if (nproc == 1) {
    fft_simd_out_body_cache(start, x_r, x_c, X_r, X_c, s, N);
    return;
  }

#pragma omp task
  {
    fft_simd_out_body_par_cache(start, x_r, x_c, X_r, X_c, 2 * s, N / 2, nproc / 2);
  }
#pragma omp task
  {
    fft_simd_out_body_par_cache(start + s, x_r, x_c, X_r + (N / 2), X_c + (N / 2), 2 * s, N / 2, nproc / 2);
  }

#pragma omp taskwait

  if (s >= (1 << GRANULARITY)) {
    fft_ispc_out(X_r, X_c, twiddles_r + (N / 2), twiddles_c + (N / 2), N);
  } else {
    fft_ispc_out_raw(X_r, X_c, N);
  }
}

// wrapper function to fft_simd_out_body
void fft_simd_out(double *A_r, double *A_c, double *O_r, double *O_c, int N, bool cached) {

  if (cached) {
#pragma omp parallel
#pragma omp single
    fft_simd_out_body_par_cache(0, A_r, A_c, O_r, O_c, 1, N, 2);
  } else {
#pragma omp parallel
#pragma omp single
    fft_simd_out_body_par(A_r, A_c, O_r, O_c, 1, N, 2);
  }
}

// Exported functions.

void fft_simd_prep(double *A_r, double *A_c, int N) {
  // produce sin, cos LUT for fft codlets
  for (int n = 0; n < 4; n++) {
    for (int k = 0; k < 4; k++) {
      LUTC4[4 * n + k] = cos(2 * M_PI * k * n / 4);
      LUTS4[4 * n + k] = -sin(2 * M_PI * k * n / 4);
    }
  }

  twiddles_c = malloc(sizeof(double) * N);
  twiddles_r = malloc(sizeof(double) * N);

  if (twiddles_c == NULL || twiddles_r == NULL) {
    printf("malloc fail for twiddles LUT\n");
    exit(0);
  }

  double *local_r;
  double *local_c;
  for (int n = N >> GRANULARITY; n > 4; n /= 2) {
    local_r = twiddles_r + (n / 2);
    local_c = twiddles_c + (n / 2);
    local_r[0] = 1;
    local_c[0] = 0;
    double complex rou = cexp(I * ((-2.0 * M_PI) / n));
    local_r[1] = creal(rou);
    local_c[1] = cimag(rou);
    for (int k = 2; k < n / 2; k++) {
      local_r[k] = local_r[k - 1] * local_r[1] - local_c[k - 1] * local_c[1];
      local_c[k] = local_c[k - 1] * local_r[1] + local_r[k - 1] * local_c[1];
    }

    // #pragma omp parallel for
    //     for (int k = 0; k < n / 2; k++) {
    //       local_r[k] = cos(2 * k * M_PI / n);
    //       local_c[k] = -sin(2 * k * M_PI / n);
    //     }
  }
}

void fft_simd(double *A_r, double *A_c, double *O_r, double *O_c, int N, bool cached) {
  fft_simd_out(A_r, A_c, O_r, O_c, N, cached);

  free(twiddles_c);
  free(twiddles_r);
}
// basic fft with omp pragmas applied
void fft_omp(double complex *A, int N, int nproc) {
  if (N <= 1) {
    return;
  }

  if (nproc == 1) {
    fft(A, N);
    return;
  }

  // split based on parity of index
  double complex *E = (double complex *)calloc(N / 2, sizeof(double complex));
  double complex *O = (double complex *)calloc(N / 2, sizeof(double complex));
#pragma omp parallel for schedule(static) num_threads(nproc)
  for (int i = 0; i < N / 2; i++) {
    E[i] = A[2 * i];
    O[i] = A[2 * i + 1];
  }

#pragma omp parallel num_threads(2)
  {
#pragma omp sections
    {
#pragma omp section
      fft_omp(E, N / 2, nproc / 2);
#pragma omp section
      fft_omp(O, N / 2, nproc / 2);
    }
  }

#pragma omp parallel for schedule(static) num_threads(nproc)
  for (int k = 0; k < N / 2; k++) {
    double complex t = O[k] * cexp(I * ((-2.0 * M_PI * k) / N));
    A[k] = E[k] + t;
    A[k + (N / 2)] = E[k] - t;
  }
  free(O);
  free(E);
}

// Out of place implementation of FFT Cooley Tukey, no optimizations

void fft_out_place(double complex *x, double complex *X, int s, int N) {
  if (N == 1) {
    X[0] = x[0];
    return;
  }

  fft_out_place(x, X, 2 * s, N / 2);
  fft_out_place(x + s, X + (N / 2), 2 * s, N / 2);

  for (int k = 0; k < N / 2; k++) {
    double complex p = X[k];
    double complex q = X[k + (N / 2)] * cexp(I * ((-2.0 * M_PI * k) / N));
    X[k] = p + q;
    X[k + (N / 2)] = p - q;
  }
}

void fft_out(double complex *A, int N) {
  double complex *out = (double complex *)calloc(N, sizeof(double complex));
  fft_out_place(A, out, 1, N);

  for (int i = 0; i < N; i++) {
    A[i] = out[i];
  }
  free(out);
}

// OMP enhanced version of fft_out_place

void fft_out_place_omp(double complex *x, double complex *X, int s, int N, int nproc) {
  if (N == 1) {
    X[0] = x[0];
    return;
  }

  if (nproc == 1) {
    fft_out_place(x, X, s, N);
  }

#pragma omp parallel num_threads(2)
  {
#pragma omp sections
    {
#pragma omp section
      fft_out_place_omp(x, X, 2 * s, N / 2, nproc / 2);
#pragma omp section
      fft_out_place_omp(x + s, X + (N / 2), 2 * s, N / 2, nproc / 2);
    }
  }

  // #pragma omp parallel for num_threads(nproc)
  for (int k = 0; k < N / 2; k++) {
    double complex p = X[k];
    double complex q = X[k + (N / 2)] * cexp(I * ((-2.0 * M_PI * k) / N));
    X[k] = p + q;
    X[k + (N / 2)] = p - q;
  }
}

void fft_out_omp(double complex *A, int N) {
  double complex *out = (double complex *)calloc(N, sizeof(double complex));
  fft_out_place_omp(A, out, 1, N, 8);

  for (int i = 0; i < N; i++) {
    A[i] = out[i];
  }
  free(out);
}

// Pthreads implementation.

// typedef struct {
//   double complex *A;
//   int N;
//   int nproc;
// } args_t;

// void *fft_par_pthread(void *args) {
//   args_t *my_args = (args_t *)args;
//   fft_par(my_args->A, my_args->N, my_args->nproc);
//   return NULL;
// }

// // Assumes that nproc is a power of two.
// void fft_par(double complex *A, int N, int nproc) {
//   if (nproc == 1) {
//     fft(A, N);
//     return;
//   }

//   // split based on parity of index
//   double complex *E = (double complex *)calloc(N / 2, sizeof(double complex));
//   double complex *O = (double complex *)calloc(N / 2, sizeof(double complex));
//   for (int i = 0; i < N / 2; i++) {
//     E[i] = A[2 * i];
//     O[i] = A[2 * i + 1];
//   }

//   // split with pthreads.
//   pthread_t thread_id;
//   args_t args;

//   args.A = E;
//   args.N = N / 2;
//   args.nproc = nproc / 2;

//   pthread_create(&thread_id, NULL, fft_par_pthread, &args);

//   fft_par(O, N / 2, nproc / 2);

//   pthread_join(thread_id, NULL);

//   for (int k = 0; k < N / 2; k++) {
//     double complex t = O[k] * cexp(I * ((-2.0 * M_PI * k) / N));
//     A[k] = E[k] + t;
//     A[k + N / 2] = E[k] - t;
//   }
//   free(O);
//   free(E);
// }