#include<stdio.h>
#include<iostream>
#include<random>
#include<cuda_runtime.h>
#include<memorypool/alloc.h>

// cannot be reorganised
#include<memorypool/math/linalg.h>

unsigned int global_blocksize;

#define MATRIX_CALCULATION_REPEATS 100

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
__global__ void test_ludcmp_kernel(double *matrices, int *results, unsigned int number, unsigned int size) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < number) {
        for (unsigned int i = 0; i < MATRIX_CALCULATION_REPEATS; i++) {
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
    if (poolMemoryBlock == nullptr) {
        printf("Error allocating memory pool\n");
        cudaFree(d_input);
        cudaFree(d_idx);
        return;
    }
    
    init_all_pools<<<gridSize, blockSize>>>(poolMemoryBlock, number);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("init pools error: %s\n", cudaGetErrorString(err));
    }
    #endif
    
    // double *buf;
    // cudaMalloc(&buf, number * size * sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // test_ludcmp_kernel<<<gridSize, blockSize>>>(d_input, d_idx, number, size, buf);
    test_ludcmp_kernel<<<gridSize, blockSize>>>(d_input, d_idx, number, size);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel run error: %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%f\n", milliseconds);

    cudaFree(d_input);
    cudaFree(d_idx);
}

__global__ void test_luminv_kernel(double *matrices, double *output, int *idx, unsigned int number, unsigned int size) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < number) {
        for (unsigned int i = 0; i < MATRIX_CALCULATION_REPEATS; i++) {
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

    test_luminv_kernel<<<gridSize, blockSize>>>(d_input, d_output, d_idx, number, size);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%f\n", milliseconds);

    cudaFree(d_input);
    cudaFree(d_output);
}

__global__ void test_leastsq_kernel(double *vectors, double *inputs, unsigned int number, unsigned int size) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < number) {
        for (unsigned int i = 0; i < MATRIX_CALCULATION_REPEATS; i++) {
            leastsq(vectors + (index * size), inputs + (index * size * size), size, size);
        }
    }
}

void test_leastsq(double *h_matrices, unsigned int number, unsigned int size) {
    // am lazy, generate size * size inputs, but only size will be used
    double *h_vectors = (double *)malloc(number * size * size * sizeof(double));
    cpu_generate_matrices(h_vectors, number, size);
    // for (unsigned int i = 0; i < number; i++) {
    //     for (unsigned int j = 0; j < size; j++) {
    //         h_vectors[i * size + j] = (double)(j);
    //     }
    // }

    double *d_vectors;
    cudaMalloc(&d_vectors, number * size * sizeof(double));
    cudaMemcpy(d_vectors, h_vectors, number * size * sizeof(double), cudaMemcpyHostToDevice);
    
    double *d_matrices;
    cudaMalloc(&d_matrices, number * size * size * sizeof(double));
    cudaMemcpy(d_matrices, h_matrices, number * size * size * sizeof(double), cudaMemcpyHostToDevice);

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

    test_leastsq_kernel<<<gridSize, blockSize>>>(d_vectors, d_matrices, number, size);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%f\n", milliseconds);

    cudaFree(d_vectors);
    cudaFree(d_vectors);
}

int main(int argc, char **argv) {
    if(argc != 4 && argc != 5) {
        printf("Usage: %s <function number> <number of matrices> <size of matrices> optional: <block size>\n", argv[0]);
        printf("Function numbers:\n");
        printf("0: ludcmp\n");
        printf("1: luminv\n");
        printf("2: leastsq\n");
        return 1;
    }

    int function_number = atoi(argv[1]);
    unsigned int number_of_matrices = atoi(argv[2]);
    unsigned int size_of_matrices = atoi(argv[3]);
    if (argc == 5) {
        global_blocksize = atoi(argv[4]);
    } else {
        global_blocksize = 1024;
    }
    // int function_number = 0;
    // unsigned int number_of_matrices = 100000;
    // unsigned int size_of_matrices = 10;

    double *h_input = (double *)malloc(number_of_matrices * size_of_matrices * size_of_matrices * sizeof(double));
    cpu_generate_matrices(h_input, number_of_matrices, size_of_matrices);
    
    // FIXME: comment out to use python script
    // #ifdef USE_MEMORY_POOL
    // printf("Using memory pool\n");
    // #else
    // printf("Not using memory pool\n");
    // #endif
    
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 6 * 1024 * 1024 * 1024);

    switch(function_number) {
        case 0:
            test_ludcmp(h_input, number_of_matrices, size_of_matrices);
            break;
        case 1:
            test_luminv(h_input, number_of_matrices, size_of_matrices);
            break;
        case 2:
            test_leastsq(h_input, number_of_matrices, size_of_matrices);
            break;
        default:
            printf("Invalid function number\n");
            free(h_input);
            return 1;
    }

    free(h_input);
    return 0;
}
