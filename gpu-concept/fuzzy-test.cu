#include <stdio.h>
#include <cuda_runtime.h>
#include <unordered_set>
#include <getopt.h>
#include <assert.h>

#include "poolalloc.cuh"

#define NUM_THREADS 4
#define OPS_PER_THREAD 10

struct TestOperation {
    bool isAlloc;
    unsigned long numBytes;
    int corrospondingAlloc;
};

void generateRandomOperations(TestOperation *ops) {
    for (int i = 0; i < NUM_THREADS; i++) {
        std::unordered_set<size_t> allocationIndices;
        size_t startingInd = i * OPS_PER_THREAD;
        for (int j = 0; j < OPS_PER_THREAD; j++) {
            if (allocationIndices.size() == 0) {
                ops[startingInd + j].isAlloc = true;
                ops[startingInd + j].numBytes = 4 * sizeof(int);
                ops[startingInd + j].corrospondingAlloc = -1;
                allocationIndices.insert(j);
            } else if (rand() % 2 == 0) {
                ops[startingInd + j].isAlloc = true;
                ops[startingInd + j].numBytes = 4 * sizeof(int);
                ops[startingInd + j].corrospondingAlloc = -1;
                allocationIndices.insert(startingInd + j);
            } else {
                ops[startingInd + j].isAlloc = false;
                ops[startingInd + j].numBytes = 0;
                auto it = allocationIndices.begin();
                std::advance(it, rand() % allocationIndices.size());
                ops[startingInd + j].corrospondingAlloc = *it;
                allocationIndices.erase(it);
            }
        }
    }
}

// Simple global spinlock for serialization
__device__ int global_lock = 0;

__device__ void acquire_lock(int *lock) {
    while (atomicCAS(lock, 0, 1) != 0) {
        // spin
    }
    __threadfence();
}

__device__ void release_lock(int *lock) {
    __threadfence();
    atomicExch(lock, 0);
}

__global__ void runTests(TestOperation *ops, void *poolMemoryBlock) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    poolinit(poolMemoryBlock, idx);

    void *allocatedPtrs[OPS_PER_THREAD];

    // acquire_lock(&global_lock);

    for (unsigned int i = 0; i < OPS_PER_THREAD; i++) {
        unsigned int opIndex = idx * OPS_PER_THREAD + i;
        if (ops[opIndex].isAlloc) {
            void *ptr = poolmalloc(ops[opIndex].numBytes);
            allocatedPtrs[i] = ptr;
            for (unsigned long j = 0; j < ops[opIndex].numBytes; j++) {
                ((char*)ptr)[j] = 'a';
            }
            printf("Thread %d: Allocated %lu bytes at %p\n", idx, ops[opIndex].numBytes, ptr);
        } else {
            char *ptrToFree = (char*)allocatedPtrs[ops[opIndex].corrospondingAlloc];
            for (unsigned long j = 0; j < ops[ops[opIndex].corrospondingAlloc].numBytes; j++) {
                if (ptrToFree[j] != 'a') {
                    printf("Thread %d: failed to free allocation at index %d\n", idx, ops[opIndex].corrospondingAlloc);
                    assert(ptrToFree[j] == 'a');
                }
            }
            poolfree(ptrToFree);
            printf("Thread %d: Freed allocation at index %d\n", idx, ops[opIndex].corrospondingAlloc);
        }
    }

    // release_lock(&global_lock);
}

int main(int argc, char **argv) {
    int opt;
    int seed = 0;
    while ((opt = getopt(argc, argv, "s:")) != -1) {
        switch (opt) {
            case 's':
                seed = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Usage: %s [-s seed]\n", argv[0]);
                exit(1);
        }
    }
    srand(seed);
    printf("Random seed: %d\n", seed);

    TestOperation ops[NUM_THREADS * OPS_PER_THREAD];
    generateRandomOperations(ops);
    
    TestOperation *d_ops;
    cudaMalloc(&d_ops, sizeof(TestOperation) * NUM_THREADS * OPS_PER_THREAD);
    cudaMemcpy(d_ops, ops, sizeof(TestOperation) * NUM_THREADS * OPS_PER_THREAD, cudaMemcpyHostToDevice);

    runTests<<<1, NUM_THREADS>>>(d_ops, allocatePools(NUM_THREADS));
    cudaDeviceSynchronize();
    cudaFree(d_ops);
}