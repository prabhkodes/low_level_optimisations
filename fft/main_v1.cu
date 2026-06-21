#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

typedef struct {
    double re;
    double im;
} Complex;

/* ================================================================== */
/* HOST / CPU HELPERS                                                 */
/* ================================================================== */

Complex c_make(double re, double im) { Complex c; c.re = re; c.im = im; return c; }
Complex c_add(Complex a, Complex b) { return c_make(a.re + b.re, a.im + b.im); }
Complex c_sub(Complex a, Complex b) { return c_make(a.re - b.re, a.im - b.im); }
Complex c_mul(Complex a, Complex b) { return c_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
Complex c_exp_i(double theta) { return c_make(cos(theta), sin(theta)); }

/* ================================================================== */
/* DEVICE HELPERS                                                     */
/* ================================================================== */

__device__ Complex dev_c_add(Complex a, Complex b) { Complex r; r.re = a.re + b.re; r.im = a.im + b.im; return r; }
__device__ Complex dev_c_sub(Complex a, Complex b) { Complex r; r.re = a.re - b.re; r.im = a.im - b.im; return r; }
__device__ Complex dev_c_mul(Complex a, Complex b) { Complex r; r.re = a.re * b.re - a.im * b.im; r.im = a.re * b.im + a.im * b.re; return r; }
__device__ Complex dev_c_exp_i(double theta) { Complex r; r.re = cos(theta); r.im = sin(theta); return r; }

/* ================================================================== */
/* KERNELS                                                            */
/* ================================================================== */

// 1. Bit Reverse (Shared by both versions)
__global__ void bit_reverse_kernel(Complex *data, int N, int log2n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int n = i, j = 0;
        for (int b = 0; b < log2n; b++) { j = (j << 1) | (n & 1); n >>= 1; }
        if (i < j) { Complex tmp = data[i]; data[i] = data[j]; data[j] = tmp; }
    }
}

// 2. V1 Kernel: Global Memory Butterfly (Naive)
__global__ void fft_butterfly_kernel(Complex *data, int N, int m, int m2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N / 2) {
        int k = tid / m2;
        int j = tid % m2;
        int idx1 = k * m + j;
        int idx2 = idx1 + m2;

        Complex w = dev_c_exp_i(-2.0 * M_PI * j / m);
        Complex u = data[idx1];
        Complex t = dev_c_mul(w, data[idx2]);

        data[idx1] = dev_c_add(u, t);
        data[idx2] = dev_c_sub(u, t);
    }
}

// 3. V2 Kernel: Shared Memory Fused (Optimized)
__global__ void fft_shared_kernel(Complex *data, int N, int log2n) {
    extern __shared__ Complex s_data[];
    int tid = threadIdx.x;
    int idx1 = tid;
    int idx2 = tid + blockDim.x;

    if (idx1 < N) s_data[idx1] = data[idx1];
    if (idx2 < N) s_data[idx2] = data[idx2];
    __syncthreads();

    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s;
        int m2 = m >> 1;
        int k = tid / m2;
        int j = tid % m2;
        int s_idx1 = k * m + j;
        int s_idx2 = s_idx1 + m2;

        Complex w = dev_c_exp_i(-2.0 * M_PI * (double)j / (double)m);
        Complex u = s_data[s_idx1];
        Complex t = dev_c_mul(w, s_data[s_idx2]);

        s_data[s_idx1] = dev_c_add(u, t);
        s_data[s_idx2] = dev_c_sub(u, t);
        __syncthreads();
    }

    if (idx1 < N) data[idx1] = s_data[idx1];
    if (idx2 < N) data[idx2] = s_data[idx2];
}

/* ================================================================== */
/* WRAPPERS                                                           */
/* ================================================================== */

// --- V1 WRAPPER (Global Memory) ---
void fft_1d_gpu_v1(Complex *h_data, int N) {
    Complex *d_data;
    size_t size = N * sizeof(Complex);
    int log2n = 0;
    for (int temp = N; temp > 1; temp >>= 1) log2n++;

    cudaMalloc((void **)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Added Timing for Fair Comparison
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Bit Reversal
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    bit_reverse_kernel<<<blocks, threadsPerBlock>>>(d_data, N, log2n);
    cudaDeviceSynchronize();

    // FFT Stages (Loop on Host, Launch Kernel per Stage)
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s;
        int m2 = m >> 1;
        int total_threads = N / 2;
        int grid_size = (total_threads + threadsPerBlock - 1) / threadsPerBlock;
        fft_butterfly_kernel<<<grid_size, threadsPerBlock>>>(d_data, N, m, m2);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU V1 Time (Global): %.4f ms\n", ms);

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

// --- V2 WRAPPER (Shared Memory) ---
void fft_1d_gpu_v2(Complex *h_data, int N) {
    Complex *d_data;
    size_t size = N * sizeof(Complex);
    int log2n = 0;
    for (int temp = N; temp > 1; temp >>= 1) log2n++;

    cudaMalloc((void **)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    // Config for V2
    int threads = N / 2;
    size_t shared_mem_size = N * sizeof(Complex);

    cudaEventRecord(start);

    bit_reverse_kernel<<<(N + 255) / 256, 256>>>(d_data, N, log2n);
    // Single Kernel Launch
    fft_shared_kernel<<<1, threads, shared_mem_size>>>(d_data, N, log2n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU V2 Time (Shared): %.4f ms\n", ms);

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

/* ================================================================== */
/* CPU & MAIN                                                         */
/* ================================================================== */

void fft_1d_cpu(Complex *data, int N) {
    // (Collapsed CPU implementation for brevity, same logic as before)
    int log2n = 0;
    for (int temp = N; temp > 1; temp >>= 1) log2n++;
    // Bit Rev
    for (unsigned int i = 0; i < N; i++) {
        unsigned int j = 0, n = i;
        for (int b = 0; b < log2n; b++) { j = (j << 1) | (n & 1); n >>= 1; }
        if (i < j) { Complex tmp = data[i]; data[i] = data[j]; data[j] = tmp; }
    }
    // Stages
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s, m2 = m >> 1;
        Complex w_step = c_exp_i(-2.0 * M_PI / m);
        for (int k = 0; k < N; k += m) {
            Complex wn = c_make(1.0, 0.0);
            for (int j = 0; j < m2; j++) {
                Complex t = c_mul(wn, data[k + j + m2]);
                Complex u = data[k + j];
                data[k + j] = c_add(u, t);
                data[k + j + m2] = c_sub(u, t);
                wn = c_mul(wn, w_step);
            }
        }
    }
}

void init_signal(Complex *data, int N) {
    for (int i = 0; i < N; i++) {
        double t = (double)i / N;
        data[i] = c_make(sin(2 * M_PI * 5 * t) + 0.5 * sin(2 * M_PI * 10 * t), 0.0);
    }
}

void save_file(const char *name, Complex *data, int N) {
    FILE *fp = fopen(name, "w");
    if(fp) { for(int i=0; i<N; i++) fprintf(fp, "%.16e %.16e\n", data[i].re, data[i].im); fclose(fp); }
}

int main(int argc, char **argv) {
    int N = 1024;
    if (argc > 1) N = atoi(argv[1]);
    
    printf("Benchmarking FFT (N=%d)\n", N);
    size_t size = N * sizeof(Complex);
    
    // Allocate 3 arrays
    Complex *h_cpu = (Complex*)malloc(size);
    Complex *h_v1 = (Complex*)malloc(size);
    Complex *h_v2 = (Complex*)malloc(size);

    init_signal(h_cpu, N);
    init_signal(h_v1, N);
    init_signal(h_v2, N);

    // 1. Run CPU
    clock_t start = clock();
    fft_1d_cpu(h_cpu, N);
    printf("CPU Time:         %.4f ms\n", (double)(clock() - start) * 1000.0 / CLOCKS_PER_SEC);
    save_file("fft_cpu.txt", h_cpu, N);

    // 2. Run GPU V1 (Global Memory)
    fft_1d_gpu_v1(h_v1, N);
    save_file("fft_v1.txt", h_v1, N);

    // 3. Run GPU V2 (Shared Memory)
    fft_1d_gpu_v2(h_v2, N);
    save_file("fft_v2.txt", h_v2, N);

    free(h_cpu); free(h_v1); free(h_v2);
    return 0;
}