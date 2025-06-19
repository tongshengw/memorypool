#include<stdio.h>
#include<iostream>
#include<random>
#include<cuda_runtime.h>
#include<memorypool/math/linalg.h>

#define SEED 0

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
        printf("HERE: %f", matrices[index]);
        ludcmp(matrices + (index * size * size), results + (index * size * size), size);
    }
}

void test_ludcmp(double *h_input, int number, int size) {
    double *d_input;
    cudaMalloc(&d_input, number * size * size * sizeof(double));
    cudaMemcpy(d_input, h_input, number * size * size * sizeof(double), cudaMemcpyHostToDevice);

    int *d_results;
    cudaMalloc(&d_results, number * (size/2) * (size/2) * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    test_ludcmp_kernel<<<1, 10>>>(d_input, d_results, number, size);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    int *h_results = (int *)malloc(number * (size/2) * (size/2) * sizeof(int));
    cudaMemcpy(h_results, d_results, number * (size/2) * (size/2) * sizeof(int), cudaMemcpyDeviceToHost);

    int *ref_results = (int *)malloc(number * (size/2) * (size/2) * sizeof(int));

    for (int i = 0; i < number; i++) {
        ref_results[i] = ludcmp(h_input + (i * size), ref_results + (i * size), size);
    }

    for (int i = 0; i < number; i++) {
        for (int j = 0; j < (size/2) * (size/2); j++) {
            // if(h_results[i] != ref_results[i]) {
            //     printf("Error at index %d: %d != %d\n", i, h_results[i], ref_results[i]);
            //     exit(1);
            // }
            std::cout << "(" << h_results[i] << ", " << ref_results[i] << ")\n";
        }
    }

    printf("Time taken: %f milliseconds\n", milliseconds);

    cudaFree(d_input);
    cudaFree(d_results);
    free(h_results);
    free(ref_results);
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