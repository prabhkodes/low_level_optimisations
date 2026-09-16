/*
 * Control experiment for the gather result in stencil.c.
 *
 * stencil.c argued: the gather uses an identity permutation, so it touches memory in the same
 * order as the unit-stride loop; therefore the whole gap is lost vectorisation. That does not
 * follow. stencil_gather also reads idx[i], which is 8 more bytes per point than stencil_unit
 * moves -- a 25% traffic increase in a loop the report itself calls DRAM-bandwidth-bound. The
 * two variants are not traffic-matched, so the gap cannot be attributed to codegen alone.
 *
 * These five variants separate the effects, holding one fixed at a time:
 *
 *   unit            out[i] from in[i-1..i+1]                        vectorised, 32 B/point
 *   unit_novec      same loop, vectoriser + interleaver disabled     scalar,     32 B/point
 *   unit_idx        unit loop that ALSO sums idx[i]                  vectorised, 40 B/point
 *   unit_idx_novec  unit_idx, vectoriser + interleaver disabled      scalar,     40 B/point
 *   gather          addresses through idx (identity permutation)     scalar,     40 B/point
 *
 *   unit_novec / unit          vectorisation alone, at matched traffic
 *   unit_idx   / unit          traffic alone, at matched codegen
 *   gather / unit_idx_novec    whatever the gather costs beyond both
 *
 * All five print the same checksum. Build with clang (the pragmas are clang's):
 *
 *   clang -O3 -mcpu=native -std=c11 stencil_control.c -o stencil_control && ./stencil_control
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N (1L << 24)
#define SWEEPS 50
#define REPEATS 3

static double wtime(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

__attribute__((noinline))
static void unit(const double *restrict in, double *restrict out, long n) {
  for (long i = 1; i < n - 1; i++)
    out[i] = 0.25 * in[i - 1] + 0.5 * in[i] + 0.25 * in[i + 1];
}

__attribute__((noinline))
static void unit_novec(const double *restrict in, double *restrict out, long n) {
  #pragma clang loop vectorize(disable) interleave(disable)
  for (long i = 1; i < n - 1; i++)
    out[i] = 0.25 * in[i - 1] + 0.5 * in[i] + 0.25 * in[i + 1];
}

__attribute__((noinline))
static long unit_idx(const double *restrict in, double *restrict out, const long *restrict idx, long n) {
  long acc = 0;
  for (long i = 1; i < n - 1; i++) {
    acc += idx[i];
    out[i] = 0.25 * in[i - 1] + 0.5 * in[i] + 0.25 * in[i + 1];
  }
  return acc;
}

__attribute__((noinline))
static long unit_idx_novec(const double *restrict in, double *restrict out, const long *restrict idx, long n) {
  long acc = 0;
  #pragma clang loop vectorize(disable) interleave(disable)
  for (long i = 1; i < n - 1; i++) {
    acc += idx[i];
    out[i] = 0.25 * in[i - 1] + 0.5 * in[i] + 0.25 * in[i + 1];
  }
  return acc;
}

__attribute__((noinline))
static void gather(const double *restrict in, double *restrict out, const long *restrict idx, long n) {
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
  double *in = malloc(N * sizeof *in);
  double *out = malloc(N * sizeof *out);
  long *idx = malloc(N * sizeof *idx);
  if (!in || !out || !idx) { perror("malloc"); return 1; }

  srand(42);
  for (long i = 0; i < N; i++) { in[i] = (double)rand() / RAND_MAX; out[i] = 0.0; idx[i] = i; }

  const char *names[] = {"unit", "unit_novec", "unit_idx", "unit_idx_novec", "gather"};
  const int bytes_per_point[] = {32, 32, 40, 40, 40};
  double best[5];

  for (int v = 0; v < 5; v++) {
    best[v] = 1e300;
    double cs = 0.0;
    long acc = 0;
    for (int r = 0; r < REPEATS; r++) {
      memset(out, 0, N * sizeof *out);
      double t0 = wtime();
      for (int s = 0; s < SWEEPS; s++) {
        switch (v) {
          case 0: unit(in, out, N); break;
          case 1: unit_novec(in, out, N); break;
          case 2: acc += unit_idx(in, out, idx, N); break;
          case 3: acc += unit_idx_novec(in, out, idx, N); break;
          case 4: gather(in, out, idx, N); break;
        }
      }
      double dt = wtime() - t0;
      if (dt < best[v]) best[v] = dt;
      cs = checksum(out, N);
    }
    double touched = (double)SWEEPS * N * bytes_per_point[v] / best[v] / 1e9;
    printf("%-15s %8.4f s  %6.2fx unit   %4d B/pt  %7.1f GB/s touched  checksum %.8f%s\n", names[v], best[v],
           best[v] / (v == 0 ? best[0] : best[0]), bytes_per_point[v], touched, cs, acc ? "  (idx summed)" : "");
  }

  printf("\nInterpretation\n");
  printf("  vectorisation benefit at equal traffic   unit_novec / unit          = %.2fx\n", best[1] / best[0]);
  printf("  extra-traffic cost at equal codegen      unit_idx / unit            = %.2fx\n", best[2] / best[0]);
  printf("  both effects together                    unit_idx_novec / unit      = %.2fx\n", best[3] / best[0]);
  printf("  measured gather                          gather / unit              = %.2fx\n", best[4] / best[0]);
  printf("  gather beyond traffic+no-vectorisation   gather / unit_idx_novec    = %.2fx\n", best[4] / best[3]);

  free(in); free(out); free(idx);
  return 0;
}
