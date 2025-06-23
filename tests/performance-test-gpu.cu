#include<stdio.h>
#include<iostream>
#include<random>
#include<cuda_runtime.h>
#include<memorypool/alloc.h>
#include<memorypool/math/linalg.h>
#include<memorypool/gpu/poolalloc.cuh>

void cpu_generate_matrices(double *output, unsigned int number, unsigned int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 100);
    for(unsigned int i = 0; i < number; i++) {
        for(unsigned int j = 0; j < size * size; j++) {
            output[i * size * size + j] = dis(gen);
        }
    }
}

__global__ void init_all_pools(void *poolMemoryBlock, unsigned int number) {
    unsigned int threadInd = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadInd < number) {
        poolinit(poolMemoryBlock, threadInd);
    }
}

// __global__ void test_ludcmp_kernel(double *matrices, int *results, unsigned int number, unsigned int size, double *buf 
__global__ void test_ludcmp_kernel(double *matrices, int *results, unsigned int number, unsigned int size
                                   #ifdef USE_MEMORY_POOL
                                   , void *ptr
                                   #endif
                                   ) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < number) {
        for (unsigned int i = 0; i < 10000; i++) {
            ludcmp(matrices + (index * size * size), results + (index * size), size);
            // ludcmp_buffered(matrices + (index * size * size), results + (index * size), size, buf + (index * size));
        }
    }
}

void test_ludcmp(double *h_input, unsigned int number, unsigned int size) {
    double *d_input;
    cudaError_t err = cudaMalloc(&d_input, number * size * size * sizeof(double));
    if (err != cudaSuccess) {
        printf("Error allocating device memory: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaMemcpy(d_input, h_input, number * size * size * sizeof(double), cudaMemcpyHostToDevice);

    int *d_idx;
    err = cudaMalloc(&d_idx, number * size * sizeof(int));
    if (err != cudaSuccess) {
        printf("Error allocating device memory for index array: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        return;
    }


    int blockSize = 1024;
    int gridSize = (number + blockSize - 1) / blockSize;

    #ifdef USE_MEMORY_POOL
    void *poolMemoryBlock = allocatePools(number);
    init_all_pools<<<gridSize, blockSize>>>(poolMemoryBlock, number);
    cudaDeviceSynchronize();
    #endif
    
    // double *buf;
    // cudaMalloc(&buf, number * size * sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    #ifdef USE_MEMORY_POOL
    test_ludcmp_kernel<<<gridSize, blockSize>>>(d_input, d_idx, number, size, poolMemoryBlock);
    #else
    // test_ludcmp_kernel<<<gridSize, blockSize>>>(d_input, d_idx, number, size, buf);
    test_ludcmp_kernel<<<gridSize, blockSize>>>(d_input, d_idx, number, size);
    #endif
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%f\n", milliseconds);

    cudaFree(d_input);
    cudaFree(d_idx);
}

__global__ void test_luminv_kernel(double *matrices, double *output, int *idx, unsigned int number, unsigned int size
                                   #ifdef USE_MEMORY_POOL
                                   , void *ptr
                                   #endif
                                   ) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < number) {
        for (unsigned int i = 0; i < 10000; i++) {
            ludcmp(matrices + (index * size * size), idx + (index * size), size);
            luminv(output + (index * size * size), matrices + (index * size * size), idx + (index * size), size);
        }
    }
}

void test_luminv(double *h_input, unsigned int number, unsigned int size) {
    double *d_input;
    cudaMalloc(&d_input, number * size * size * sizeof(double));
    cudaMemcpy(d_input, h_input, number * size * size * sizeof(double), cudaMemcpyHostToDevice);

    double *d_output;
    cudaMalloc(&d_output, number * size * size * sizeof(double));

    int *d_idx;
    cudaMalloc(&d_idx, number * size * sizeof(int));

    int blockSize = 1024;
    int gridSize = (number + blockSize - 1) / blockSize;

    #ifdef USE_MEMORY_POOL
    void *poolMemoryBlock = allocatePools(number);
    init_all_pools<<<gridSize, blockSize>>>(poolMemoryBlock, number);
    cudaDeviceSynchronize();
    #endif

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    #ifdef USE_MEMORY_POOL
    test_luminv_kernel<<<gridSize, blockSize>>>(d_input, d_output, d_idx, number, size, poolMemoryBlock);
    #else
    test_luminv_kernel<<<gridSize, blockSize>>>(d_input, d_output, d_idx, number, size);
    #endif
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%f\n", milliseconds);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char **argv) {
    if(argc != 4) {
        printf("Usage: %s <function number> <number of matrices> <size of matrices>\n", argv[0]);
        printf("Function numbers:\n");
        printf("0: ludcmp\n");
        printf("1: luminv\n");
        return 1;
    }

    int function_number = atoi(argv[1]);
    unsigned int number_of_matrices = atoi(argv[2]);
    unsigned int size_of_matrices = atoi(argv[3]);

    double *h_input = (double *)malloc(number_of_matrices * size_of_matrices * size_of_matrices * sizeof(double));
    cpu_generate_matrices(h_input, number_of_matrices, size_of_matrices);
    
    // FIXME: comment out to use python script
    // #ifdef USE_MEMORY_POOL
    // printf("Using memory pool\n");
    // #else
    // printf("Not using memory pool\n");
    // #endif

    switch(function_number) {
        case 0:
            test_ludcmp(h_input, number_of_matrices, size_of_matrices);
            break;
        case 1:
            test_luminv(h_input, number_of_matrices, size_of_matrices);
        default:
            printf("Invalid function number\n");
            free(h_input);
            return 1;
    }

    free(h_input);
    return 0;
}
