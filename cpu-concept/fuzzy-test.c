#include "./poolalloc.h"
#include "stdlib.h"
#include "stdio.h"
#include <assert.h>

// #define MAX_TEST_SIZE 200
// #define MAX_BYTES_PER_ALLOC 256
// #define NUM_TESTS 1

#define MAX_TEST_SIZE 8
#define MAX_BYTES_PER_ALLOC 256
#define NUM_TESTS 1

typedef struct TestOperation {
    bool isAlloc;
    unsigned long numBytes;
    int corrospondingAlloc;
} TestOperation;

void printOperationArr(TestOperation *ops, int N) {
    for (int i = 0; i < N; i++) {
        printf("(alloc: %d, bytes %lu, corrosp: %d), ", ops[i].isAlloc, ops[i].numBytes, ops[i].corrospondingAlloc);
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

int linearSearchArr(void **arr, int N, void *target) {
    for (int i = 0; i < N; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}

// NOTE: make sure every free corrosponds to a previous allocate when generating ops
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
            printf("Allocated %lu bytes at (%p)\n", ops[i].numBytes, allocatedAddrs[i]);
        } else {
            void *freedPtr = allocatedAddrs[ops[i].corrospondingAlloc];
            poolfree(freedPtr);
            allocatedAddrs[ops[i].corrospondingAlloc] = NULL;
            printf("Freed %lu bytes at (%p)\n", bytesAllocated[ops[i].corrospondingAlloc], freedPtr);
        }
        printlayout();
        printf("============================\n");
    }
    
    for (int i = 0; i < N; i++) {
        if (allocatedAddrs[i] != NULL) {
            poolfree(allocatedAddrs[i]);
        }    
    }
}

int main(void) {
    poolinit();
    // TestOperation ops[3];
    // ops[0].isAlloc = true;
    // ops[0].numBytes = 100;
    // ops[1].isAlloc = true;
    // ops[1].numBytes = 2;
    // ops[2].isAlloc = false;
    // ops[2].corrospondingAlloc = 0;
    // executeTestOperationArr(ops, 3);
    
    for (int i = 0; i < NUM_TESTS; i++) {
        TestOperation *ops = malloc(MAX_TEST_SIZE * sizeof(TestOperation));
        printf("TEST %d:\n", i);
        genRandomTestOperationArr(ops, MAX_TEST_SIZE);
        executeTestOperationArr(ops, MAX_TEST_SIZE);
        free(ops);
    }
}