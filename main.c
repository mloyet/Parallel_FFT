/*
 * Command Line interface for testing and performance
 * tracking of fft.c
 */

#include "fft.h"
#include "stdbool.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "unistd.h"

#include "fftw3.h"
#include "omp.h"

// used for if verify flag set.
#define EPSILON 1e-3

int length; // number of input points given.

void print_help() {
  printf("Arguments:\n-f infile\n-o outfile\n\n");
  printf("-r randlen");
  printf("-V validate");
  printf("-t time");
}

void print_fftw(double prep_cost, double cost) {
  printf("========== FFTW TIME ==========\n");
  printf("prep time: %.2f ms (%.2f s)\n", prep_cost, prep_cost / 1000);
  printf("exec time: %.2f ms (%.2f s)\n", cost, cost / 1000);
  printf("\n");
}

void print_fft(double prep_cost, double cost) {
  printf("========== OUR TIME ===========\n");
  printf("prep time: %.2f ms (%.2f s)\n", prep_cost, prep_cost / 1000);
  printf("exec time: %.2f ms (%.2f s)\n", cost, cost / 1000);
  printf("\n");
}

void print_csv(double prep_cost, double cost) {
  printf("%.2f, ", prep_cost * 1000);
  printf("%.2f", cost * 1000);
  printf("\n");
}

static inline int index4(int i, int N) {
  return ((i * 4) + ((i * 4) / N)) % N;
}

void create_random_cache(fftw_complex *input_fftw, double **A_r, double **A_c, int length) {

  for (int i = 0; i < length; i++) {
    (*A_c)[index4(i, length)] = 0;
    (*A_r)[index4(i, length)] = ((double)rand()) / RAND_MAX;
    input_fftw[i] = (*A_r)[index4(i, length)];
  }
}

void create_random(fftw_complex *input_fftw, double **A_r, double **A_c, int length) {

  for (int i = 0; i < length; i++) {
    (*A_c)[i] = 0;
    (*A_r)[i] = ((double)rand()) / RAND_MAX;
    input_fftw[i] = (*A_r)[i];
  }
}

void read_infile(char *filename, double *A_r, double *A_c) {
  FILE *f = fopen(filename, "r");
  fscanf(f, "%d\n", &length);

  double x;
  for (int i = 0; i < length; i++) {
    fscanf(f, "%lf\n", &x);
    A_r[i] = x;
    A_c[i] = 0;
  }

  fclose(f);
}

fftw_complex *read_infile_fftw(char *filename) {
  FILE *f = fopen(filename, "r");

  fscanf(f, "%d\n", &length);

  fftw_complex *input = fftw_malloc(length * sizeof(double complex));
  double x;

  for (int i = 0; i < length; i++) {
    fscanf(f, "%lf\n", &x);
    input[i] = x;
  }

  fclose(f);

  return input;
}

bool double_close(double a, double b) {
  return fabs(a - b) < EPSILON;
}

int main(int argc, char **argv) {

  char *infile = NULL;
  char *outfile = NULL;
  FILE *out;
  bool fftw = false;
  bool validate = false;
  bool report_time = false;
  bool write_out = false;
  bool rand_in = false;
  int nproc = -1;

  bool caching = false;

  struct timespec before, after;

  omp_set_num_threads(8);

  int c;
  while ((c = getopt(argc, argv, "f:o:wtVp:r:c")) != -1) {
    switch (c) {
    case 'f':
      infile = optarg;
      break;

    case 'o':
      outfile = optarg;
      write_out = true;
      break;

    case 'p':
      nproc = atoi(optarg);
      if (nproc <= 1) {
        printf("Enter a positive number of processors.\n");
        exit(1);
      }
      break;

    case 'r':
      rand_in = true;
      length = atoi(optarg);
      if (length <= 0) {
        printf("Enter a positive random length.\n");
        exit(1);
      }
      break;

    case 'w':
      fftw = true;
      break;

    case 'V':
      validate = true;
      fftw = true;
      break;

    case 't':
      report_time = true;
      // fftw = true;
      break;

    case 'c':
      caching = true;
      break;

    default:
      print_help();
      exit(1);
    }
  }
  if (infile == NULL && !rand_in) {
    print_help();
    exit(1);
  }

  if (outfile == NULL) {
    out = stdout;
  } else {
    out = fopen(outfile, "w");
  }

  // END command line parsing

  fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * length);
  fftw_complex *o = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * length);
  double *A_r = calloc(length, sizeof(double));
  double *A_c = calloc(length, sizeof(double));
  double *O_r = calloc(length, sizeof(double));
  double *O_c = calloc(length, sizeof(double));

  if (rand_in && !caching) {
    create_random(in, &A_r, &A_c, length);
  } else if (rand_in) {
    create_random_cache(in, &A_r, &A_c, length);
  } else {
    in = read_infile_fftw(infile);
  }

  if (fftw) {

    clock_gettime(CLOCK_REALTIME, &before);

    // IMPORTANT: if you plan to use any other prep than estimate, in order
    // to actually compute the result, the data must be loaded after making
    // the plan.
    fftw_plan p = fftw_plan_dft_1d(length, in, o, FFTW_FORWARD, FFTW_ESTIMATE);
    clock_gettime(CLOCK_REALTIME, &after);
    double prep_cost = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;

    clock_gettime(CLOCK_REALTIME, &before);
    fftw_execute(p);
    clock_gettime(CLOCK_REALTIME, &after);
    double fftw_cost = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;

    if (report_time) {
      print_fftw(prep_cost, fftw_cost);
    }

    fftw_destroy_plan(p);
  }

  clock_gettime(CLOCK_REALTIME, &before);
  fft_simd_prep(A_r, A_c, length);
  clock_gettime(CLOCK_REALTIME, &after);
  double prep_cost = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;

  clock_gettime(CLOCK_REALTIME, &before);
  fft_simd(A_r, A_c, O_r, O_c, length, caching);
  clock_gettime(CLOCK_REALTIME, &after);

  double cost = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;

  if (report_time) {
    print_fftw(prep_cost, cost);
  }

  if (write_out) {
    for (int i = 0; i < length; i++) {
      fprintf(out, "%f, %f\n", creal(in[i]), cimag(in[i]));
    }
  }

  if (validate) {
    for (int i = 0; i < length; i++) {
      if (!(double_close(creal(o[i]), O_r[i]) &&
            double_close(cimag(o[i]), O_c[i]))) {
        printf("Missmatch at %d: (%f, %f) != (%f, %f)\n",
               i, creal(o[i]), cimag(o[i]), O_r[i], O_c[i]);
        exit(1);
      }
    }
    printf("Validation successful. Comparisons passed.\n");
  }

  fclose(out);
  free(A_r);
  free(A_c);
  fftw_free(in);
  fftw_free(o);
  return 0;
}
