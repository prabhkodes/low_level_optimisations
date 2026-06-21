#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#define t_time unsigned long int
static inline t_time getmicros(){
			struct timeval tv;
			gettimeofday(&tv, NULL);
			return tv.tv_sec*1e6 + tv.tv_usec;
}

#if 0
void blocked_gemm(int N, double * restrict A, double * restrict B, double * restrict C, int BLOCK_SIZE) {
	for (int i = 0; i < N; i += BLOCK_SIZE) {
		for (int k = 0; k < N; k += BLOCK_SIZE) {
			for (int j = 0; j < N; j += BLOCK_SIZE) {
				// Compute block multiplication
				for (int ii = i; ii < i + BLOCK_SIZE && ii < N; ++ii) {
					for (int kk = k; kk < k + BLOCK_SIZE && kk < N; ++kk) {
						double a_val = A[ii * N + kk];
						for (int jj = j; jj < j + BLOCK_SIZE && jj < N; ++jj) {
							C[ii * N + jj] += a_val * B[kk * N + jj];
						}
					}
				}
			}
		}
	}
}
#else
void blocked_gemm(int N, double * restrict A, double * restrict B, double * restrict C, int BLOCK_SIZE) {
	for (int i = 0; i < N; i += BLOCK_SIZE) {
		for (int j = 0; j < N; j += BLOCK_SIZE) {
			for (int k = 0; k < N; k += BLOCK_SIZE) {
				// Compute block multiplication
				for (int ii = i; ii < i + BLOCK_SIZE && ii < N; ++ii) {
					for (int jj = j; jj < j + BLOCK_SIZE && jj < N; ++jj) {
						double sum = C[ii * N + jj];
						for (int kk = k; kk < k + BLOCK_SIZE && kk < N; ++kk) {
							sum += A[ii * N + kk] * B[kk * N + jj];
						}
						C[ii * N + jj] = sum;
					}
				}
			}
		}
	}
}
#endif

int main(int argc, char * argv[]){
	if (argc < 3){
		printf("Usage: %s N BLOCK_SIZE\n",argv[0]);
		exit(-1);
	}
	int N = atoi(argv[1]);
	int BLOCK_SIZE = atoi(argv[2]);

	double N_kb=N*N*3.0*sizeof(double)/1024.0;
	double B_kb=BLOCK_SIZE*BLOCK_SIZE*3.0*sizeof(double)/1024.0;
	printf("Using %.2f KB of data\n", N_kb);
	printf("In a Block: %.2f KB\n", B_kb);

	double * A = (double *)malloc(sizeof(double)*N*N);
	double * B = (double *)malloc(sizeof(double)*N*N);
	double * C = (double *)malloc(sizeof(double)*N*N);

	char hostname[32];
	gethostname(hostname, 32);
	int user = getuid();

	int reps=3;
	t_time ini = getmicros();
	for(int r=0; r<reps; ++r){
		blocked_gemm(N, A, B, C, BLOCK_SIZE);
	}
	t_time fin = getmicros();
	double time = (double)(fin-ini)/reps;
	double flops = 2.0*N*N*N;
	printf("User\tMachine\tsize(N)\tsize(KB)\t\tBlockSize\tBlocSize(KB)\ttime(micros)\tPerformance(MFLOPs)\n");
	printf("%d\t%s\t%d\t%.2f\t%d\t%.2f\t%.2f\t%.2f\n",user,hostname,N,N_kb,BLOCK_SIZE,B_kb,time,flops/time);

	volatile double no_opt = C[0];
	return 0;
}