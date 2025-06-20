#include<stdio.h>
#include<iostream>
#include<random>
#include<cuda_runtime.h>
#include<memorypool/math/linalg.h>

// #define SEED 1235
#define APPROX_EQUAL_DIFF 1e-6

bool approx_equal(double a, double b) {
    return fabs(a - b) < APPROX_EQUAL_DIFF;
}

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

__global__ void test_ludcmp_kernel(double *matrices, int *results, unsigned int number, unsigned int size) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < number) {
        ludcmp(matrices + (index * size * size), results + (index * size), size);
    }
}

void test_ludcmp(double *h_input, unsigned int number, unsigned int size) {
    double *d_input;
    cudaMalloc(&d_input, number * size * size * sizeof(double));
    cudaMemcpy(d_input, h_input, number * size * size * sizeof(double), cudaMemcpyHostToDevice);

    int *d_idx;
    cudaMalloc(&d_idx, number * size * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Calculate grid size to handle all matrices
    int blockSize = 256;
    int gridSize = (number + blockSize - 1) / blockSize;
    test_ludcmp_kernel<<<gridSize, blockSize>>>(d_input, d_idx, number, size);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // double *gpu_input_modified = (double *)malloc(number * size * size * sizeof(double));
    // cudaMemcpy(gpu_input_modified, d_input, number * size * size * sizeof(double), cudaMemcpyDeviceToHost);

    // int *gpu_idx_modified = (int *)malloc(number * size * sizeof(int));
    // cudaMemcpy(gpu_idx_modified, d_idx, number * size * sizeof(int), cudaMemcpyDeviceToHost);

    // int *ref_idx = (int *)malloc(number * size * sizeof(int));

    // // printf("Calculating CPU reference...\n");
    // for (unsigned int i = 0; i < number; i++) {
    //     ludcmp(h_input + (i * size * size), ref_idx + (i * size), size);
    // }

    // // checking idx array
    // // printf("Checking idx array correctness...\n");
    // for (unsigned int i = 0; i < number; i++) {
    //     for (unsigned int j = 0; j < size; j++) {
    //         int idx_pos = i * size + j;
    //         if(gpu_idx_modified[idx_pos] != ref_idx[idx_pos]) {
    //             printf("Error at idx index %d (matrix %d, element %d): %d != %d\n", idx_pos, i, j, gpu_idx_modified[idx_pos], ref_idx[idx_pos]);
    //             exit(1);
    //         }
    //     }
    // }
    
    // // checking input array
    // // printf("Checking input array correctness...\n");
    // for (unsigned int i = 0; i < number; i++) {
    //     for (unsigned int j = 0; j < size * size; j++) {
    //         if(!approx_equal(h_input[i * size * size + j], gpu_input_modified[i * size * size + j])) {
    //             printf("Error at input index %d: %f != %f\n", i * size * size + j, gpu_input_modified[i * size * size + j], h_input[i * size * size + j]);
    //             exit(1);
    //         }
    //     }
    // }

    printf("%f\n", milliseconds);

    // cudaFree(d_input);
    // cudaFree(d_idx);
    // free(gpu_input_modified);
    // free(gpu_idx_modified);
    // free(ref_idx);
}

int main(int argc, char **argv) {
    if(argc != 4) {
        printf("Usage: %s <function number> <number of matrices> <size of matrices>\n", argv[0]);
        printf("Function numbers:\n");
        printf("0: ludcmp\n");
        return 1;
    }

    int function_number = atoi(argv[1]);
    unsigned int number_of_matrices = atoi(argv[2]);
    unsigned int size_of_matrices = atoi(argv[3]);

    double *h_input = (double *)malloc(number_of_matrices * size_of_matrices * size_of_matrices * sizeof(double));
    cpu_generate_matrices(h_input, number_of_matrices, size_of_matrices);
    // printf("h_input values:\n");
    // for (int i = 0; i < number_of_matrices; i++) {
    //     printf("Matrix %d:\n", i);
    //     for (int j = 0; j < size_of_matrices; j++) {
    //         for (int k = 0; k < size_of_matrices; k++) {
    //             printf("%f ", h_input[i * size_of_matrices * size_of_matrices + j * size_of_matrices + k]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    switch(function_number) {
        case 0:
            test_ludcmp(h_input, number_of_matrices, size_of_matrices);
            break;
        default:
            printf("Invalid function number\n");
            free(h_input);
            return 1;
    }

    free(h_input);
    return 0;
}