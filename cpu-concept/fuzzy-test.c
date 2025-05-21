#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "./poolalloc.h"

#define MAX_TEST_SIZE 2
#define MAX_BYTES_PER_ALLOC 20
#define NUM_TESTS 1

/*
This file generates random testing sequences for poolmalloc and poolfree.

genRandomTestOperationArr() generates an array of valid TestOperations.

executeTestOperationArr() executes the array of TestOperations.
*/

// TestOperation:
// isAlloc - true for allocations
// numBytes - number of bytes to allocate
// corrospondingAlloc - index of allocate operation to free
typedef struct TestOperation {
    bool isAlloc;
    unsigned long numBytes;
    int corrospondingAlloc;
} TestOperation;

void printOperationArr(TestOperation *ops, int N) {
    for (int i = 0; i < N; i++) {
        printf("(alloc: %d, bytes %lu, corrosp: %d), ", ops[i].isAlloc,
               ops[i].numBytes, ops[i].corrospondingAlloc);
    }
    printf("\n");
}

void genRandomTestOperationArr(TestOperation *ops, int N) {
    int curAllocs = 0;
    bool allocated[MAX_TEST_SIZE];
    for (int i = 0; i < MAX_TEST_SIZE; i++) {
        allocated[i] = false;
    }

    for (int i = 0; i < N; i++) {
        bool setAlloc;
        if (curAllocs == 0) {
            setAlloc = true;
        } else {
            if (rand() % 2 == 0) {
                setAlloc = false;
            } else {
                setAlloc = true;
            }
        }

        if (setAlloc) {
            ops[i].isAlloc = true;
            ops[i].numBytes = rand() % MAX_BYTES_PER_ALLOC;
            allocated[i] = true;
            curAllocs++;
        } else {
            int allocToFree = rand() % curAllocs;
            int j = 0;
            int seen = 0;
            while (j < MAX_TEST_SIZE) {
                if (allocated[j] && seen == allocToFree) {
                    ops[i].isAlloc = false;
                    ops[i].corrospondingAlloc = j;
                    allocated[j] = false;
                    curAllocs--;
                    break;
                } else if (allocated[j]) {
                    seen++;
                    j++;
                } else {
                    j++;
                }
            }
        }
    }
    printOperationArr(ops, N);
}

void executeTestOperationArr(TestOperation *ops, int N) {
    void *allocatedAddrs[MAX_TEST_SIZE];
    unsigned long bytesAllocated[MAX_TEST_SIZE];
    for (int i = 0; i < MAX_TEST_SIZE; i++) {
        allocatedAddrs[i] = NULL;
        bytesAllocated[i] = 0;
    }

    printf("============================\n");
    for (int i = 0; i < N; i++) {
        printf("Operation %d: ", i);
        if (ops[i].isAlloc) {
            allocatedAddrs[i] = poolmalloc(ops[i].numBytes);
            bytesAllocated[i] = ops[i].numBytes;
            assert(allocatedAddrs[i] != NULL);
            printf("Allocated %lu bytes at (%p)\n", ops[i].numBytes,
                   allocatedAddrs[i]);
        } else {
            void *freedPtr = allocatedAddrs[ops[i].corrospondingAlloc];
            poolfree(freedPtr);
            allocatedAddrs[ops[i].corrospondingAlloc] = NULL;
            printf("Freed %lu bytes at (%p)\n",
                   bytesAllocated[ops[i].corrospondingAlloc], freedPtr);
        }
        printlayout();
        printf("============================\n");
    }

    for (int i = 0; i < N; i++) {
        if (allocatedAddrs[i] != NULL) {
            poolfree(allocatedAddrs[i]);
        }
    }
    printlayout();
}

int main(void) {
    poolinit();

    for (int i = 0; i < NUM_TESTS; i++) {
        TestOperation *ops = malloc(MAX_TEST_SIZE * sizeof(TestOperation));
        printf("TEST %d:\n", i);
        genRandomTestOperationArr(ops, MAX_TEST_SIZE);
        executeTestOperationArr(ops, MAX_TEST_SIZE);
        free(ops);
    }
}