#include <cuda_runtime.h>
#include "linalg.h"

__global__ void test_linalg_kernel() {
    test_ludcmp();
    test_lubksb();
    test_luminv();
    test_leastsq();
    test_leastsq_kkt();
}

void run_linalg_tests() {
    test_linalg_kernel<<<1, 256>>>();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel execution: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}
