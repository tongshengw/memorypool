#include "./poolalloc.h"
#include "stdlib.h"
#include "stdio.h"

#define MAX_TEST_SIZE 100

void randTests() {
    int sizes[100];
    sizes[0] = 0;
    for (int i = 1; i < 100; i++) {
       sizes[i] = sizes[i-1] + i; 
    }
    
    void *allocatedPtrs[MAX_TEST_SIZE];
    int allocatedPtrsSize = 0;
    
    
}

int main(void) {

}