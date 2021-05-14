


#ifndef FFT_H
#define FFT_H

#include "complex.h"
#include "math.h"
#include "stdbool.h"

void fft_simd (double *A_r, double *A_c, double *O_r, double *O_c, int N, bool cached);
void fft_simd_prep (double *A_r, double *A_c, int N);

#endif