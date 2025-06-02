#include <stdio.h>
#include <cuda_runtime.h>

__global__ void allocate_and_write(int **ptrs, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int *mem = (int*)malloc(4 * sizeof(int));
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
        printf("Thread %d: ", idx);
        for (int i = 0; i < 4; ++i) {
            printf("%d ", ptrs[idx][i]);
        }
        printf("\n");
        free(ptrs[idx]);
    }
}

int main() {
    int n = 8;
    int **d_ptrs;
    printf("here\n");
    cudaMalloc(&d_ptrs, n * sizeof(int*));

    allocate_and_write<<<1, n>>>(d_ptrs, n);
    cudaDeviceSynchronize();

    read_and_free<<<1, n>>>(d_ptrs, n);
    cudaDeviceSynchronize();

    cudaFree(d_ptrs);
    return 0;
}