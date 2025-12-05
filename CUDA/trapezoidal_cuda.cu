
// Compile with:
// nvcc trapezoidal_cuda.cu -O2 -arch=sm_75 -o trap_cuda

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Device function f(x)
__device__ inline double f(double x) {
    return sin(x);
}

// Kernel: each thread walks with stride and accumulates a local double sum.
// At the end each thread does one atomicAdd to the global result.
__global__ void trapezoid_kernel(uint64_t n, double a, double h, double *d_result) {
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * (uint64_t)gridDim.x;

    double local_sum = 0.0;
    // Interior points: i = 1 .. n-1
    for (uint64_t i = idx + 1ULL; i < n; i += stride) {
        double x = a + (double)i * h;
        local_sum += f(x);
    }

    // Do one atomicAdd per thread if non-zero
    if (local_sum != 0.0) {
        atomicAdd(d_result, local_sum);
    }
}

static inline void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    // Defaults
    uint64_t n = 100000000ULL; // subdivisions
    int threadsPerBlock = 256;

    if (argc >= 2) n = strtoull(argv[1], nullptr, 10);
    if (argc >= 3) threadsPerBlock = atoi(argv[2]);
    if (threadsPerBlock <= 0) threadsPerBlock = 256;

    // Query device
    int dev = 0;
    checkCuda(cudaGetDevice(&dev), "cudaGetDevice");
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");

    // Check for double atomic support (compute capability >= 6.0 recommended)
    if (prop.major < 6) {
        fprintf(stderr,
            "Warning: device \"%s\" (compute capability %d.%d) may NOT support atomicAdd(double).\n"
            "This program will exit. Fallback: implement per-block reduction + single atomic per block.\n",
            prop.name, prop.major, prop.minor);
        return 1;
    }

    // Interval geometry
    double a = 0.0;
    double b = PI;
    double h = (b - a) / (double)n;

    // Determine blocks: need enough threads to cover n (threads process interior points with stride)
    // Compute rough number of needed threads, then set blocks = ceil(neededThreads / threadsPerBlock)
    uint64_t neededThreads = (n > 0ULL ? n : 1ULL);
    uint64_t blocks_u64 = (neededThreads + (uint64_t)threadsPerBlock - 1ULL) / (uint64_t)threadsPerBlock;
    if (blocks_u64 < 1ULL) blocks_u64 = 1ULL;

    // Cap blocks to device's max grid capacity (product of maxGridSize dims)
    uint64_t maxBlocks = 1ULL
        * (uint64_t)prop.maxGridSize[0]
        * (uint64_t)prop.maxGridSize[1]
        * (uint64_t)prop.maxGridSize[2];

    if (blocks_u64 > maxBlocks) blocks_u64 = maxBlocks;

    // Fit into int for kernel launch (CUDA accepts up to 2^31-1 in host API); split to reasonable size if needed
    // If blocks exceed 2^31-1, cap to maxBlocks (already enforced). Cast to int for launch.
    int blocks = (int)blocks_u64;
    if (blocks < 1) blocks = 1;

    // Allocate device result and initialize
    double *d_result = nullptr;
    checkCuda(cudaMalloc((void**)&d_result, sizeof(double)), "cudaMalloc d_result");
    checkCuda(cudaMemset(d_result, 0, sizeof(double)), "cudaMemset d_result");

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    // Start GPU timer
    checkCuda(cudaEventRecord(start, 0), "cudaEventRecord start");

    // Launch kernel
    trapezoid_kernel<<<blocks, threadsPerBlock>>>(n, a, h, d_result);

    // Check launch error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        // Clean up before exit
        cudaFree(d_result);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return 1;
    }

    // Synchronize and stop timer
    checkCuda(cudaEventRecord(stop, 0), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");

    // Copy result back
    double gpu_sum = 0.0;
    checkCuda(cudaMemcpy(&gpu_sum, d_result, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy d_result");

    // Clean up
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Add endpoints and scale by h: trapezoid rule = h*( (f(a)+f(b))/2 + sum_{i=1..n-1} f(x_i) )
    double endpoints = 0.5 * (sin(a) + sin(b));
    double result = h * (endpoints + gpu_sum);

    // Print exactly the requested format:
    // n=<n> blocks=<b> threadsPerBlock=<t> result=<result> time=<time_in_seconds> s GPU=<gpu_name>
    printf("n=%llu blocks=%d threadsPerBlock=%d result=%.12f time=%.6f s GPU=%s\n",
           (unsigned long long)n, blocks, threadsPerBlock, result, (double)(ms / 1000.0), prop.name);

    return 0;
}
