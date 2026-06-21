#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define PAD(x) (x + (x >> 5)) // Macro to get padded index for avoiding bank conflict

typedef struct {
    double re;
    double im;
} Complex;


/* ================================================================== */
/* CPU COMPLEX NUMBER HELPERS                                         */
/* ================================================================== */

Complex c_make(double re, double im) { Complex c; c.re = re; c.im = im; return c; }
Complex c_add(Complex a, Complex b) { return c_make(a.re + b.re, a.im + b.im); }
Complex c_sub(Complex a, Complex b) { return c_make(a.re - b.re, a.im - b.im); }
Complex c_mul(Complex a, Complex b) { return c_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
Complex c_exp_i(double theta) { return c_make(cos(theta), sin(theta)); }

/* ================================================================== */
/* DEVICE COMPLEX NUMBER HELPERS                                      */
/* ================================================================== */

__device__ Complex dev_c_add(Complex a, Complex b) { Complex r; r.re = a.re + b.re; r.im = a.im + b.im; return r; }
__device__ Complex dev_c_sub(Complex a, Complex b) { Complex r; r.re = a.re - b.re; r.im = a.im - b.im; return r; }
__device__ Complex dev_c_mul(Complex a, Complex b) { Complex r; r.re = a.re * b.re - a.im * b.im; r.im = a.re * b.im + a.im * b.re; return r; }
__device__ Complex dev_c_exp_i(double theta) { Complex r; r.re = cos(theta); r.im = sin(theta); return r; }

/* ================================================================== */
/* GPU KERNELS                                                        */
/* ================================================================== */

__global__ void bit_reverse_kernel(Complex *data, int N, int log2n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int n = i, j = 0;
        for (int b = 0; b < log2n; b++) { j = (j << 1) | (n & 1); n >>= 1; }
        if (i < j) { Complex tmp = data[i]; data[i] = data[j]; data[j] = tmp; }
    }
}

// V1 Kernel
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

// V2 Kernel (Shared, Unpadded)
__global__ void fft_shared_kernel(Complex *data, int N, int log2n) {
    extern __shared__ Complex s_data[]; // dynamic allocation of shared memory based on kernel args
    int tid = threadIdx.x; // local thread id
    
    int idx1 = tid; // element 1 the thread will handle
    int idx2 = tid + blockDim.x; // element 2 the thread will handle

    // load from global to shared memory
    if (idx1 < N) s_data[idx1] = data[idx1]; 
    if (idx2 < N) s_data[idx2] = data[idx2];
    __syncthreads();

    // FFT 
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s; // sub-problem size
        int m2 = m >> 1; // stride
        int k = tid / m2; // butterfly group idx
        int j = tid % m2; // butterfly item idx

        int s_idx1 = k * m + j; // top part, in shared mem
        int s_idx2 = s_idx1 + m2; // bottom part, in shared mem

        // Butterfly Math (u = even, t = odd * twiddle)
        Complex w = dev_c_exp_i(-2.0 * M_PI * (double)j / (double)m);
        Complex u = s_data[s_idx1];
        Complex t = dev_c_mul(w, s_data[s_idx2]);

        // write back to shared mem
        s_data[s_idx1] = dev_c_add(u, t);
        s_data[s_idx2] = dev_c_sub(u, t);

        __syncthreads();
    }

    // write back to global mem
    if (idx1 < N) data[idx1] = s_data[idx1];
    if (idx2 < N) data[idx2] = s_data[idx2];
}

// V3 Kernel (Shared + Padding Optimization)
__global__ void fft_padded_kernel(Complex *data, int N, int log2n) {
    extern __shared__ Complex s_data[]; // dynamically allocated shared memory with padding
    
    int tid = threadIdx.x;
    int idx1 = tid;
    int idx2 = tid + blockDim.x;

    // use padded index
    // load data from global to shared memory
    // with padding of one element
    if (idx1 < N) s_data[PAD(idx1)] = data[idx1]; 
    if (idx2 < N) s_data[PAD(idx2)] = data[idx2];
    
    __syncthreads();

    // FFT
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s;
        int m2 = m >> 1;
        int k = tid / m2;
        int j = tid % m2;

        int s_idx1 = k * m + j;
        int s_idx2 = s_idx1 + m2;

        Complex w = dev_c_exp_i(-2.0 * M_PI * (double)j / (double)m);
        
        // use padded memory location for no bank conflict
        Complex u = s_data[PAD(s_idx1)];
        Complex t = dev_c_mul(w, s_data[PAD(s_idx2)]);

        // write the data in correct place in shared memory
        s_data[PAD(s_idx1)] = dev_c_add(u, t);
        s_data[PAD(s_idx2)] = dev_c_sub(u, t);
        
        __syncthreads();
    }

    // write data back from padded shared memory into global memory
    if (idx1 < N) data[idx1] = s_data[PAD(idx1)];
    if (idx2 < N) data[idx2] = s_data[PAD(idx2)];
}

/* ================================================================== */
/* WRAPPERS FOR KERNELS AND HOST FUNCTIONS                            */
/* ================================================================== */


/**
 * Naive implementation of fft_1d
 */
float fft_1d_gpu_v1(Complex *h_data, int N) {
    Complex *d_data; // device pointer
    size_t size = N * sizeof(Complex); // problem size

    // cuda config
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int total_threads = N / 2; // We need N/2 butterflies per stage
    int grid_size = (total_threads + threadsPerBlock - 1) / threadsPerBlock;
    
    int log2n = 0;
    for (int temp = N; temp > 1; temp >>= 1) log2n++; // total FFT stages

    cudaMalloc((void **)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start); // start timer


    bit_reverse_kernel<<<blocks, threadsPerBlock>>>(d_data, N, log2n);
    cudaDeviceSynchronize();

    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s; // Current sub-problem size (2, 4, 8...)
        int m2 = m >> 1; // Stride/Offset for the butterfly (1, 2, 4...)
        fft_butterfly_kernel<<<grid_size, threadsPerBlock>>>(d_data, N, m, m2);
    }

    cudaEventRecord(stop); // stop timer
    cudaEventSynchronize(stop);

    // time evaluation
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU V1 Time (Global):   %.4f ms\n", ms);

    // memory cleanup
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    return ms;
}

/**
 * Shared memory implementation of fft_1d
 */
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

    // cuda config
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    bit_reverse_kernel<<<blocks, threadsPerBlock>>>(d_data, N, log2n);
    fft_shared_kernel<<<1, threads, shared_mem_size>>>(d_data, N, log2n); // shared memory, hence 1 huge block
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU V2 Time (Shared):   %.4f ms\n", ms);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    return ms;
}


/**
 * Shared memory + fixing bank conflicts
 */
float fft_1d_gpu_v3(Complex *h_data, int N) {
    Complex *d_data;
    size_t size = N * sizeof(Complex);
    int log2n = 0;
    for (int temp = N; temp > 1; temp >>= 1) log2n++;
    cudaMalloc((void **)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // cuda config
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int threads = N / 2;
    size_t padded_elements = N + (N >> 5); // 32 threads in each warp)
    size_t shared_mem_size = padded_elements * sizeof(Complex);


    cudaEventRecord(start);
    bit_reverse_kernel<<<blocks, threadsPerBlock>>>(d_data, N, log2n);
    fft_padded_kernel<<<1, threads, shared_mem_size>>>(d_data, N, log2n);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU V3 Time (Padded):   %.4f ms\n", ms);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    return ms;
}

/**
 * Host method for FFT
 */
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

/* ================================================================== */
/* UTILITIES                                                          */
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
/* MAIN                                                               */
/* ================================================================== */

int main(int argc, char **argv) {
    int N = 1024;
    if (argc > 1) N = atoi(argv[1]);
    
    // Fix 1: Declare 'size' FIRST so metrics can use it
    size_t size = N * sizeof(Complex);
    
    // Fix 2: Constants for metrics
    double log2n = log2((double)N);
    double total_ops = 5.0 * (double)N * log2n;
    
    // Estimated Bytes for Bandwidth
    // V1 hits VRAM log2(N) times for bit-rev + stages
    double bytes_v1 = (double)size * (log2n + 1.0) * 2.0; 
    // V2/V3 only hit VRAM for initial load and final store
    double bytes_shared = (double)size * 2.0; 

    printf("Benchmarking FFT (N=%d)\n", N);
    
    Complex *h_cpu = (Complex*)malloc(size);
    Complex *h_v1 = (Complex*)malloc(size);
    Complex *h_v2 = (Complex*)malloc(size);
    Complex *h_v3 = (Complex*)malloc(size);

    init_signal(h_cpu, N);
    init_signal(h_v1, N);
    init_signal(h_v2, N);
    init_signal(h_v3, N);

    // --- CPU ---
    clock_t start = clock();
    fft_1d_cpu(h_cpu, N);
    double cpu_ms = (double)(clock() - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("CPU Time:           %.4f ms | GFLOPS: %6.2f\n", 
            cpu_ms, (total_ops / (cpu_ms * 1e6)));
    save_file("fft_cpu.txt", h_cpu, N);

    // --- GPU V1 (Global) ---
    float ms_v1 = fft_1d_gpu_v1(h_v1, N); 
    printf("GPU V1 Time:        %8.4f ms | GFLOPS: %6.2f | BW: %7.2f GB/s\n", 
            ms_v1, (total_ops / (ms_v1 * 1e6)), (bytes_v1 / (ms_v1 * 1e6)));
    save_file("fft_gpu_v1.txt", h_v1, N);

    // --- GPU V2 (Shared) ---
    float ms_v2 = fft_1d_gpu_v2(h_v2, N);
    printf("GPU V2 Time:        %8.4f ms | GFLOPS: %6.2f | BW: %7.2f GB/s\n", 
            ms_v2, (total_ops / (ms_v2 * 1e6)), (bytes_shared / (ms_v2 * 1e6)));
    save_file("fft_gpu_v2.txt", h_v2, N);

    // --- GPU V3 (Shared + Padded) ---
    float ms_v3 = fft_1d_gpu_v3(h_v3, N);
    printf("GPU V3 Time:        %8.4f ms | GFLOPS: %6.2f | BW: %7.2f GB/s\n", 
            ms_v3, (total_ops / (ms_v3 * 1e6)), (bytes_shared / (ms_v3 * 1e6)));
    save_file("fft_gpu_v3.txt", h_v3, N);

    free(h_cpu); free(h_v1); free(h_v2); free(h_v3);
    return 0;
}