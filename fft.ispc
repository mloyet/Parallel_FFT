
export void fft_twiddle(uniform double E_r[], uniform double E_c[],
                        uniform double O_r[], uniform double O_c[], 
                        uniform double twiddle_r[], uniform double twiddle_c[], 
                        uniform double A_r[], uniform double A_c[],
                        uniform int N) 
{
  
  foreach(k = 0 ... N / 2)
  {
    double t_r = O_r[k] * twiddle_r[k] - O_c[k] * twiddle_c[k];
    double t_c = O_c[k] * twiddle_r[k] + O_r[k] * twiddle_c[k];
    // A[k]
    A_r[k] = E_r[k] + t_r;
    A_c[k] = E_c[k] + t_c;
    // A[k + N / 2]
    A_r[k + N / 2] = E_r[k] - t_r; 
    A_c[k + N / 2] = E_c[k] - t_c;
  }
}


export void fft_ispc_out (uniform double X_r[], uniform double X_c[], 
                          uniform double twiddle_r[], uniform double twiddle_c[],
                          uniform int N) 
{
  foreach (k = 0 ... N / 2) {
    double p_r = X_r[k];
    double p_c = X_c[k];

    double q_r = X_r[k + N / 2] * twiddle_r[k] - X_c[k + N / 2] * twiddle_c[k];
    double q_c = X_c[k + N / 2] * twiddle_r[k] + X_r[k + N / 2] * twiddle_c[k];

    X_r[k] = p_r + q_r;
    X_c[k] = p_c + q_c;

    X_r[k + (N / 2)] = p_r - q_r;
    X_c[k + (N / 2)] = p_c - q_c;
  }
}

export void fft_ispc_out_raw (uniform double X_r[], uniform double X_c[], 
                              uniform int N) 
{
  foreach (k = 0 ... N / 2) {
    double p_r = X_r[k];
    double p_c = X_c[k];

    double twiddle_r = cos (2 * k * PI / N);
    double twiddle_c = -sin (2 * k * PI / N);

    double q_r = X_r[k + N / 2] * twiddle_r - X_c[k + N / 2] * twiddle_c;
    double q_c = X_c[k + N / 2] * twiddle_r + X_r[k + N / 2] * twiddle_c;

    X_r[k] = p_r + q_r;
    X_c[k] = p_c + q_c;

    X_r[k + (N / 2)] = p_r - q_r;
    X_c[k + (N / 2)] = p_c - q_c;
  }
}

export void fft_ispc_out_split (uniform double X_r[], uniform double X_c[], 
                                uniform double twiddle_r[], uniform double twiddle_c[],
                                uniform int N, uniform int start, uniform int end) 
{
  foreach (k = start ... end) {
    double p_r = X_r[k];
    double p_c = X_c[k];

    double q_r = X_r[k + N / 2] * twiddle_r[k] - X_c[k + N / 2] * twiddle_c[k];
    double q_c = X_c[k + N / 2] * twiddle_r[k] + X_r[k + N / 2] * twiddle_c[k];

    X_r[k] = p_r + q_r;
    X_c[k] = p_c + q_c;

    X_r[k + (N / 2)] = p_r - q_r;
    X_c[k + (N / 2)] = p_c - q_c;
  }
}


export void fft_ispc_eight (uniform double X_r[], uniform double X_c[], 
                            uniform double LUTS[], uniform double LUTC[]) {
  uniform double x_r[8];
  uniform double x_c[8];

  foreach (i = 0 ... 8) {
    x_r[i] = X_r[i];
    x_c[i] = X_c[i];
    X_r[i] = 0;
    X_c[i] = 0;
  }

  for (int n = 0; n < 8; n++) {
    foreach (k = 0 ... 8) {
      // double t_r = cos (2 * PI * k * n / 8);
      // double t_c = -sin (2 * PI * k * n / 8); 
      double t_r = LUTC[8*n + k];
      double t_c = LUTS[8*n + k]; 
      X_r[k] += x_r[n] * t_r - x_c[n] * t_c;
      X_c[k] += x_c[n] * t_r + x_r[n] * t_c; 
    }
  }
}

export void fft_ispc_eight_out (uniform double X_r[], uniform double X_c[], 
                                uniform double x_r[], uniform double x_c[],
                                uniform double LUTS[], uniform double LUTC[],
                                uniform int s) {

  uniform double x8_r[8];
  uniform double x8_c[8];

  foreach (i = 0 ... 8) {
    x8_r[i] = x_r[s * i];
    x8_c[i] = x_c[s * i];
  }

  for (int n = 0; n < 8; n++) {
    foreach (k = 0 ... 8) {
      // double t_r = cos (2 * PI * k * n / 8);
      // double t_c = -sin (2 * PI * k * n / 8); 
      double t_r = LUTC[8*n + k];
      double t_c = LUTS[8*n + k]; 
      X_r[k] += x8_r[n] * t_r - x8_c[n] * t_c;
      X_c[k] += x8_c[n] * t_r + x8_r[n] * t_c; 
    }
  }
}

export void fft_ispc_four_out (uniform double X_r[], uniform double X_c[], 
                                uniform double x_r[], uniform double x_c[],
                                uniform double LUTS[], uniform double LUTC[],
                                uniform int s) {

  uniform double x8_r[4];
  uniform double x8_c[4];

  foreach (i = 0 ... 4) {
    x8_r[i] = x_r[s * i];
    x8_c[i] = x_c[s * i];
  }

  for (int n = 0; n < 4; n++) {
    foreach (k = 0 ... 4) {
      // double t_r = cos (2 * PI * k * n / 4);
      // double t_c = -sin (2 * PI * k * n / 4); 
      double t_r = LUTC[4*n + k];
      double t_c = LUTS[4*n + k]; 
      X_r[k] += x8_r[n] * t_r - x8_c[n] * t_c;
      X_c[k] += x8_c[n] * t_r + x8_r[n] * t_c; 
    }
  }
}

export void fft_ispc_four_out_cache (uniform double X_r[], uniform double X_c[], 
                                     uniform double x_r[], uniform double x_c[],
                                     uniform double LUTS[], uniform double LUTC[],
                                     uniform int start) {

  uniform double *x8_r = x_r + (start * 4);
  uniform double *x8_c = x_c + (start * 4);

  for (int n = 0; n < 4; n++) {
    foreach (k = 0 ... 4) {
      // double t_r = cos (2 * PI * k * n / 4);
      // double t_c = -sin (2 * PI * k * n / 4); 
      double t_r = LUTC[4*n + k];
      double t_c = LUTS[4*n + k]; 
      X_r[k] += x8_r[n] * t_r - x8_c[n] * t_c;
      X_c[k] += x8_c[n] * t_r + x8_r[n] * t_c; 
    }
  }
}