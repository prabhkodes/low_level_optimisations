/*
 * 1D 3-point Jacobi stencil, plus two deliberately worse variants used to
 * show what the compiler (and the LoopStride pass) can and cannot fix:
 *
 *   stencil_unit    - unit-stride sweep, vectorises cleanly
 *   stencil_strided - same maths but walks the array with stride S,
 *                     kills spatial locality and vectorisation
 *   stencil_gather  - indexes through a permutation table, SCEV cannot
 *                     compute a stride at all (gather pattern)
 *
 * Build via scripts/remarks.sh to see what clang says about each loop.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef N
#define N (1 << 24)
#endif
#define SWEEPS 50
#define STRIDE 8

static double wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void stencil_unit(const double *restrict in, double *restrict out, long n) {
    for (long i = 1; i < n - 1; i++)
        out[i] = 0.25 * in[i - 1] + 0.5 * in[i] + 0.25 * in[i + 1];
}

void stencil_strided(const double *restrict in, double *restrict out, long n) {
    /* Same update, but iterate in STRIDE-spaced passes. Every load pulls a
     * whole cache line for one useful element. */
    for (long s = 0; s < STRIDE; s++)
        for (long i = (s == 0 ? STRIDE : s); i < n - 1; i += STRIDE)
            out[i] = 0.25 * in[i - 1] + 0.5 * in[i] + 0.25 * in[i + 1];
}

void stencil_gather(const double *restrict in, double *restrict out,
                    const long *restrict idx, long n) {
    for (long i = 1; i < n - 1; i++) {
        long j = idx[i];
        out[j] = 0.25 * in[j - 1] + 0.5 * in[j] + 0.25 * in[j + 1];
    }
}

static double checksum(const double *a, long n) {
    double s = 0.0;
    for (long i = 0; i < n; i++) s += a[i];
    return s;
}

int main(void) {
    long n = N;
    double *in  = malloc(n * sizeof *in);
    double *out = malloc(n * sizeof *out);
    long  *idx  = malloc(n * sizeof *idx);
    if (!in || !out || !idx) { perror("malloc"); return 1; }

    srand(42);
    for (long i = 0; i < n; i++) {
        in[i]  = (double)rand() / RAND_MAX;
        out[i] = 0.0;
    }
    /* Identity permutation: gather variant does the same work, the only
     * difference is that the compiler can no longer prove it. */
    for (long i = 0; i < n; i++) idx[i] = i;

    struct {
        const char *name;
        int variant;
    } cases[] = { {"unit", 0}, {"strided", 1}, {"gather", 2} };

    printf("%-10s %12s %14s %16s\n", "variant", "time (s)", "GB/s", "checksum");
    for (int c = 0; c < 3; c++) {
        memset(out, 0, n * sizeof *out);
        double t0 = wtime();
        for (int sweep = 0; sweep < SWEEPS; sweep++) {
            switch (cases[c].variant) {
            case 0: stencil_unit(in, out, n); break;
            case 1: stencil_strided(in, out, n); break;
            case 2: stencil_gather(in, out, idx, n); break;
            }
        }
        double dt = wtime() - t0;
        /* 3 doubles read + 1 written per point per sweep */
        double gbs = (double)SWEEPS * n * 4 * sizeof(double) / dt / 1e9;
        printf("%-10s %12.4f %14.2f %16.8f\n",
               cases[c].name, dt, gbs, checksum(out, n));
    }

    free(in); free(out); free(idx);
    return 0;
}
