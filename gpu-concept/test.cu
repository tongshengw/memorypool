#include <stdio.h>
#include <cuda_runtime.h>
#include "poolalloc.cuh"

__global__ void allocate_and_write(int **ptrs, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    poolinit(idx);
    if (idx < n) {
        int *mem = (int*)poolmalloc(4 * sizeof(int));
        if (mem != NULL) {
            for (int i = 0; i < 4; ++i) {
                mem[i] = idx * 10 + i;
            }
            ptrs[idx] = mem;
        } else {
            ptrs[idx] = NULL;
        }
    }
}

__global__ void read_and_free(int **ptrs, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && ptrs[idx] != NULL) {
        for (int i = 0; i < 4; ++i) {
            printf("%d ", ptrs[idx][i]);
        }
        poolfree(ptrs[idx]);
    }
}

int main() {
    int n = 8;
    int **d_ptrs;
    cudaMalloc(&d_ptrs, n * sizeof(int*));

    allocatePools(n);
    allocate_and_write<<<1, n>>>(d_ptrs, n);
    cudaDeviceSynchronize();

    read_and_free<<<1, n>>>(d_ptrs, n);
    cudaDeviceSynchronize();
    freePools();

    cudaFree(d_ptrs);
    return 0;
}