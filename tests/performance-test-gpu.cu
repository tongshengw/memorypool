#include<stdio.h>
#include<iostream>
#include<random>
#include<cuda_runtime.h>
#include<memorypool/math/linalg.h>

#define SEED 2
#define APPROX_EQUAL_DIFF 1e-6

bool approx_equal(double a, double b) {
    return fabs(a - b) < APPROX_EQUAL_DIFF;
}

void cpu_generate_matrices(double *output, int number, int size) {
    std::mt19937 gen(SEED);
    std::uniform_real_distribution<double> dis(0, 100);
    for(int i = 0; i < number; i++) {
        for(int j = 0; j < size * size; j++) {
            output[i * size * size + j] = dis(gen);
        }
    }
}

__global__ void test_ludcmp_kernel(double *matrices, int *results, int number, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < number) {
        ludcmp(matrices + (index * size * size), results + (index * size * size), size);
    }
}

void test_ludcmp(double *h_input, int number, int size) {
    double *d_input;
    cudaMalloc(&d_input, number * size * size * sizeof(double));
    cudaMemcpy(d_input, h_input, number * size * size * sizeof(double), cudaMemcpyHostToDevice);

    int *d_idx;
    cudaMalloc(&d_idx, number * (size/2) * (size/2) * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    test_ludcmp_kernel<<<1, 10>>>(d_input, d_idx, number, size);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    double *gpu_input_modified = (double *)malloc(number * size * size * sizeof(double));
    cudaMemcpy(gpu_input_modified, d_input, number * size * size * sizeof(double), cudaMemcpyDeviceToHost);

    int *h_idx = (int *)malloc(number * (size/2) * (size/2) * sizeof(int));
    cudaMemcpy(h_idx, d_idx, number * (size/2) * (size/2) * sizeof(int), cudaMemcpyDeviceToHost);

    int *ref_idx = (int *)malloc(number * (size/2) * (size/2) * sizeof(int));

    printf("Calculating CPU reference...\n");
    for (int i = 0; i < number; i++) {
        ludcmp(h_input + (i * size * size), ref_idx + (i * size * size), size);
    }

    // checking idx array
    printf("Checking idx array correctness...\n");
    for (int i = 0; i < number; i++) {
        for (int j = 0; j < (size/2) * (size/2); j++) {
            if(h_idx[i] != ref_idx[i]) {
                printf("Error at idx index %d: %d != %d\n", i, h_idx[i], ref_idx[i]);
                exit(1);
            }
        }
    }
    
    // checking input array
    printf("Checking input array correctness...\n");
    for (int i = 0; i < number; i++) {
        for (int j = 0; j < size * size; j++) {
            if(!approx_equal(h_input[i * size * size + j], gpu_input_modified[i * size * size + j])) {
                printf("Error at input index %d: %f != %f\n", i * size * size + j, gpu_input_modified[i * size * size + j], h_input[i * size * size + j]);
                exit(1);
            }
        }
    }

    printf("Time taken: %f milliseconds\n", milliseconds);

    cudaFree(d_input);
    cudaFree(d_idx);
    free(h_idx);
    free(ref_idx);
}

int main(int argc, char **argv) {
    if(argc != 4) {
        printf("Usage: %s <function number> <number of matrices> <size of matrices>\n", argv[0]);
        printf("Function numbers:\n");
        printf("0: ludcmp\n");
        return 1;
    }

    int function_number = atoi(argv[1]);
    int number_of_matrices = atoi(argv[2]);
    int size_of_matrices = atoi(argv[3]);

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
            return 1;
    }
}