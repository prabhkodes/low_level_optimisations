#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* ================= Complex num def ================= */

typedef struct {
    double re;
    double im;
} Complex;

Complex c_make(double re, double im) {
    Complex c = { re, im };
    return c;
}

Complex c_add(Complex a, Complex b) {
    return c_make(a.re + b.re, 
                  a.im + b.im
                 );
}

Complex c_sub(Complex a, Complex b) {
    return c_make(a.re - b.re, 
                  a.im - b.im
                 );
}

Complex c_mul(Complex a, Complex b) {
    return c_make(
        a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re
    );
}

double c_abs(Complex a) {
    return sqrt(a.re * a.re + 
                a.im * a.im
               );
}

Complex c_exp_i(double theta) {
    return c_make(cos(theta), 
                  sin(theta)
                 );
}

/* ================= FFT implementation ================= */

unsigned int bit_reverse(unsigned int x, int log2n) {
    unsigned int result = 0;
    for (int i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

void fft_1d(Complex *data, int N) {
    int log2n = 0;
    for (int temp = N; temp > 1; temp >>= 1)
        log2n++;

    /* Bit-reversal permutation */
    for (unsigned int i = 0; i < (unsigned int)N; i++) {
        unsigned int j = bit_reverse(i, log2n);
        if (i < j) {
            Complex tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }

    /* Cooley-Tukey */
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s;
        int m2 = m >> 1;

        Complex w = c_exp_i(-2.0 * M_PI / m);

        for (int k = 0; k < N; k += m) {
            Complex wn = c_make(1.0, 0.0);

            for (int j = 0; j < m2; j++) {
                Complex t = c_mul(wn, data[k + j + m2]);
                Complex u = data[k + j];

                data[k + j]       = c_add(u, t);
                data[k + j + m2]  = c_sub(u, t);

                wn = c_mul(wn, w);
            }
        }
    }
}

/* ================= Helpers ================= */

void init_test_signal(Complex *data, int N) {
    for (int i = 0; i < N; i++) {
        double t = (double)i / N;
        double v = sin(2 * M_PI * 5 * t) + 0.5 * sin(2 * M_PI * 10 * t);
        data[i]  = c_make(v, 0.0);
    }
}

void save_fft_results(const char *filename, Complex *data, int N) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Could not open file\n");
        return;
    }

    for (int i = 0; i < N; i++) {
        fprintf(fp, "%.16e %.16e\n", data[i].re, data[i].im);
    }
    fclose(fp);
}

/* ================= main ================= */

int main(int argc, char **argv) {
    int N = 1024;

    if (argc > 1) {
        N = atoi(argv[1]);
        if ((N & (N - 1)) != 0 || N <= 0) {
            fprintf(stderr, "N must be a power of 2\n");
            return 1;
        }
    }

    printf("1D FFT (user-defined complex)\n");
    printf("N = %d\n", N);

    Complex *data = malloc(N * sizeof(Complex));
    if (!data) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    init_test_signal(data, N);

    clock_t start = clock();
    fft_1d(data, N);
    clock_t end = clock();

    double ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("FFT time: %.3f ms\n", ms);

    char filename[256]; 
    snprintf(filename, sizeof(filename), "fft_result_%d.txt", N); 
    save_fft_results(filename, data, N); 
    printf("Results saved to %s\n", filename);

    free(data);
    return 0;
}