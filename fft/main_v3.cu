#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cufft.h>


#define PAD(x) (x + (x >> 5)) 

typedef struct {
    double re;
    double im;
} Complex;

/* ================================================================== */
/* COMPLEX NUMBER HELPERS (Unified for CPU and OpenACC)               */
/* ================================================================== */

// #pragma acc routine seq
// Complex c_make(double re, double im) { Complex c; c.re = re; c.im = im; return c; }
// #pragma acc routine seq
// Complex c_add(Complex a, Complex b) { return c_make(a.re + b.re, a.im + b.im); }
// #pragma acc routine seq
// Complex c_sub(Complex a, Complex b) { return c_make(a.re - b.re, a.im - b.im); }
// #pragma acc routine seq
// Complex c_mul(Complex a, Complex b) { return c_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
// #pragma acc routine seq
// Complex c_exp_i(double theta) { return c_make(cos(theta), sin(theta)); }

// __forceinline__ Complex c_make(double re, double im) { Complex c; c.re = re; c.im = im; return c; }
// __forceinline__ Complex c_add(Complex a, Complex b) { return c_make(a.re + b.re, a.im + b.im); }
// __forceinline__ Complex c_sub(Complex a, Complex b) { return c_make(a.re - b.re, a.im - b.im); }
// __forceinline__ Complex c_mul(Complex a, Complex b) { return c_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
// __forceinline__ Complex c_exp_i(double theta) { return c_make(cos(theta), sin(theta)); }

__forceinline__ Complex c_make(double re, double im) { Complex c; c.re = re; c.im = im; return c; }
#define c_add(a, b) (c_make(a.re + b.re, a.im + b.im) )
#define c_sub(a, b) (c_make(a.re - b.re, a.im - b.im) )
#define c_mul(a, b) (c_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re) )
#define c_exp_i(theta) (c_make(cos(theta), sin(theta)) )


/* ================================================================== */
/* DEVICE COMPLEX NUMBER HELPERS (Specifically for CUDA)              */
/* ================================================================== */

__device__ Complex dev_c_add(Complex a, Complex b) { Complex r; r.re = a.re + b.re; r.im = a.im + b.im; return r; }
__device__ Complex dev_c_sub(Complex a, Complex b) { Complex r; r.re = a.re - b.re; r.im = a.im - b.im; return r; }
__device__ Complex dev_c_mul(Complex a, Complex b) { Complex r; r.re = a.re * b.re - a.im * b.im; r.im = a.re * b.im + a.im * b.re; return r; }
__device__ Complex dev_c_exp_i(double theta) { Complex r; r.re = cos(theta); r.im = sin(theta); return r; }

/* ================================================================== */
/* GPU KERNELS (CUDA)                                                 */
/* ================================================================== */

__global__ void bit_reverse_kernel(Complex *data, int N, int log2n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int n = i, j = 0;
        for (int b = 0; b < log2n; b++) { j = (j << 1) | (n & 1); n >>= 1; }
        if (i < j) { Complex tmp = data[i]; data[i] = data[j]; data[j] = tmp; }
    }
}

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

__global__ void fft_padded_kernel(Complex *data, int N, int log2n) {
    extern __shared__ Complex s_data[]; 
    int tid = threadIdx.x;
    int idx1 = tid;
    int idx2 = tid + blockDim.x;

    if (idx1 < N) s_data[PAD(idx1)] = data[idx1]; 
    if (idx2 < N) s_data[PAD(idx2)] = data[idx2];
    __syncthreads();

    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s;
        int m2 = m >> 1;
        int k = tid / m2;
        int j = tid % m2;
        int s_idx1 = k * m + j;
        int s_idx2 = s_idx1 + m2;
        Complex w = dev_c_exp_i(-2.0 * M_PI * (double)j / (double)m);
        Complex u = s_data[PAD(s_idx1)];
        Complex t = dev_c_mul(w, s_data[PAD(s_idx2)]);
        s_data[PAD(s_idx1)] = dev_c_add(u, t);
        s_data[PAD(s_idx2)] = dev_c_sub(u, t);
        __syncthreads();
    }

    if (idx1 < N) data[idx1] = s_data[PAD(idx1)];
    if (idx2 < N) data[idx2] = s_data[PAD(idx2)];
}

/* ================================================================== */
/* WRAPPERS                                                           */
/* ================================================================== */

float fft_1d_gpu_v1(Complex *h_data, int N) {
    Complex *d_data;
    size_t size = N * sizeof(Complex);
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int total_threads = N / 2;
    int grid_size = (total_threads + threadsPerBlock - 1) / threadsPerBlock;
    int log2n = 0;
    for (int temp = N; temp > 1; temp >>= 1) log2n++;
    cudaMalloc((void **)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    bit_reverse_kernel<<<blocks, threadsPerBlock>>>(d_data, N, log2n);
    cudaDeviceSynchronize();
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s; 
        int m2 = m >> 1;
        fft_butterfly_kernel<<<grid_size, threadsPerBlock>>>(d_data, N, m, m2);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    return ms;
}

float fft_1d_gpu_v2(Complex *h_data, int N) {
    Complex *d_data;
    size_t size = N * sizeof(Complex);
    int log2n = 0;
    for (int temp = N; temp > 1; temp >>= 1) log2n++;
    cudaMalloc((void **)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int threads = N / 2;
    size_t shared_mem_size = N * sizeof(Complex);
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    cudaEventRecord(start);
    bit_reverse_kernel<<<blocks, threadsPerBlock>>>(d_data, N, log2n);
    fft_shared_kernel<<<1, threads, shared_mem_size>>>(d_data, N, log2n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    return ms;
}

float fft_1d_gpu_v3(Complex *h_data, int N) {
    Complex *d_data;
    size_t size = N * sizeof(Complex);
    int log2n = 0;
    for (int temp = N; temp > 1; temp >>= 1) log2n++;
    cudaMalloc((void **)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int threads = N / 2;
    size_t padded_elements = N + (N >> 5);
    size_t shared_mem_size = padded_elements * sizeof(Complex);
    cudaEventRecord(start);
    bit_reverse_kernel<<<blocks, threadsPerBlock>>>(d_data, N, log2n);
    fft_padded_kernel<<<1, threads, shared_mem_size>>>(d_data, N, log2n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    return ms;
}

void fft_1d_cpu(Complex *data, int N) {
    int log2n = 0;
    for (int temp = N; temp > 1; temp >>= 1) log2n++;
    for (unsigned int i = 0; i < N; i++) {
        unsigned int j = 0, n = i;
        for (int b = 0; b < log2n; b++) { j = (j << 1) | (n & 1); n >>= 1; }
        if (i < j) { Complex tmp = data[i]; data[i] = data[j]; data[j] = tmp; }
    }
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s, m2 = m >> 1;
        for (int k = 0; k < N; k += m) {
            for (int j = 0; j < m2; j++) {
                Complex w = c_exp_i(-2.0 * M_PI * j / m);
                Complex t = c_mul(w, data[k + j + m2]);
                Complex u = data[k + j];
                data[k + j] = c_add(u, t);
                data[k + j + m2] = c_sub(u, t);
            }
        }
    }
}

/* ================================================================== */
/* CuFFT                                                              */
/* ================================================================== */

float fft_1d_cufft(Complex *h_data, int N) {
    cufftHandle plan;
    cufftDoubleComplex *d_data;
    size_t size = N * sizeof(Complex);

    // 1. Allocate & Copy to GPU
    cudaMalloc((void**)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // 2. Create Plan & Execute
    if (cufftPlan1d(&plan, N, CUFFT_Z2Z, 1) != CUFFT_SUCCESS) return 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // 3. Copy Back & Cleanup
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cufftDestroy(plan);
    cudaFree(d_data);

    return ms;
}

/* ================================================================== */
/* OPENACC IMPLEMENTATIONS                                            */
/* ================================================================== */

float fft_1d_gpu_acc_v1(Complex *data, int N) {
    int log2n = 0;
    for (int temp = N; temp > 1; temp >>= 1) log2n++;
    clock_t start, stop;
    
    #pragma acc data copy(data[0:N], start, stop)
    {
        start = clock();
        #pragma acc parallel loop independent
        for (unsigned int i = 0; i < N; i++) {
            unsigned int n = i, j = 0;
            for (int b = 0; b < log2n; b++) { j = (j << 1) | (n & 1); n >>= 1; }
            if (i < j) { Complex tmp = data[i]; data[i] = data[j]; data[j] = tmp; }
        }
        
        for (int s = 1; s <= log2n; s++) {
            int m = 1 << s, m2 = m >> 1;
            #pragma acc parallel loop independent collapse(2)
            for (int k = 0; k < N; k += m) {
                // #pragma acc loop independent vector
                for (int j = 0; j < m2; j++) {
                    Complex w = c_exp_i(-2.0 * M_PI * j / m);
                    Complex u = data[k + j];
                    Complex t = c_mul(w, data[k + j + m2]);
                    data[k + j] = c_add(u, t);
                    data[k + j + m2] = c_sub(u, t);
                }
            }
        }
        stop = clock();
    }
    
    return (float)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
}

/* ================================================================== */
/* UTILITIES & MAIN                                                   */
/* ================================================================== */

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

/* ================================================================== */
/* MAIN FUNCTION                                                      */
/* ================================================================== */
int main(int argc, char **argv) {
    // 1. SETUP PROBLEM SIZE
    int N = 1024;
    if (argc > 1) N = atoi(argv[1]);
    
    size_t size = N * sizeof(Complex);
    printf("========================================\n");
    printf("BENCHMARKING FFT (N = %d)\n", N);
    printf("========================================\n");

    // 2. CALCULATE METRICS CONSTANTS
    double log2n = log2((double)N);
    double total_ops = 5.0 * (double)N * log2n;
    
    // BW constants: 
    // V1/ACC_V1 (Global) read/write log2n times.
    // V2/V3 (Shared) read/write only once (start/end).
    double bytes_global = (double)size * (log2n + 1.0) * 2.0; 
    double bytes_shared = (double)size * 2.0; 

    // --- 1. CPU REFERENCE ---
    {
        Complex *h_cpu = (Complex*)malloc(size);
        init_signal(h_cpu, N);
        
        clock_t start = clock();
        fft_1d_cpu(h_cpu, N);
        double cpu_ms = (double)(clock() - start) * 1000.0 / CLOCKS_PER_SEC;
        
        printf("CPU Time:           %8.4f ms | GFLOPS: %6.2f\n", 
               cpu_ms, (total_ops / (cpu_ms * 1e6)));
        
        // save_file("fft_cpu.txt", h_cpu, N);
        free(h_cpu);
    }

    // --- 2. GPU V1 (Global Memory) ---
    {
        Complex *h_v1 = (Complex*)malloc(size);
        init_signal(h_v1, N);
        
        float ms = fft_1d_gpu_v1(h_v1, N);
        
        printf("GPU V1 Time:        %8.4f ms | GFLOPS: %6.2f | BW: %7.2f GB/s\n", 
               ms, (total_ops / (ms * 1e6)), (bytes_global / (ms * 1e6)));
        
        // save_file("fft_gpu_v1.txt", h_v1, N);
        free(h_v1);
    }

    // --- 3. GPU V2 (Shared Memory) ---
    {
        Complex *h_v2 = (Complex*)malloc(size);
        init_signal(h_v2, N);
        
        float ms = fft_1d_gpu_v2(h_v2, N);
        
        printf("GPU V2 Time:        %8.4f ms | GFLOPS: %6.2f | BW: %7.2f GB/s\n", 
               ms, (total_ops / (ms * 1e6)), (bytes_shared / (ms * 1e6)));
        
        // save_file("fft_gpu_v2.txt", h_v2, N);
        free(h_v2);
    }

    // --- 4. GPU V3 (Shared + Padded) ---
    {
        Complex *h_v3 = (Complex*)malloc(size);
        init_signal(h_v3, N);
        
        float ms = fft_1d_gpu_v3(h_v3, N);
        
        printf("GPU V3 Time:        %8.4f ms | GFLOPS: %6.2f | BW: %7.2f GB/s\n", 
               ms, (total_ops / (ms * 1e6)), (bytes_shared / (ms * 1e6)));
        
        // save_file("fft_gpu_v3.txt", h_v3, N);
        free(h_v3);
    }

    // --- 5. OPENACC V1 (Global Memory Access) ---
    {
        Complex *h_acc = (Complex*)malloc(size);
        init_signal(h_acc, N);
        
        float ms = fft_1d_gpu_acc_v1(h_acc, N);
        
        printf("ACC V1 Time:        %8.4f ms | GFLOPS: %6.2f | BW: %7.2f GB/s\n", 
               ms, (total_ops / (ms * 1e6)), (bytes_global / (ms * 1e6)));
        
        // save_file("fft_acc_v1.txt", h_acc, N);
        free(h_acc);
    }

    // --- 6. cuFFT (Reference Library) ---
    {
        Complex *h_cufft = (Complex*)malloc(size);
        init_signal(h_cufft, N);
        
        float ms = fft_1d_cufft(h_cufft, N);
        
        if (ms > 0.0f) {
            printf("cuFFT Time:         %8.4f ms | GFLOPS: %6.2f | BW: %7.2f GB/s\n", 
                   ms, (total_ops / (ms * 1e6)), (bytes_shared / (ms * 1e6)));
            // save_file("fft_cufft.txt", h_cufft, N);
        }
        free(h_cufft);
    }

    return 0;
}