#include "./poolalloc.h"
#include "stdlib.h"
#include "stdio.h"
#include "assert.h"

void test_alloc_ints() {
    int *ptrs[14];
    for (int i = 0; i < 14; i++) {
        int *ptr = (int*) poolmalloc(sizeof(int));
        *ptr = i;
        ptrs[i] = ptr;
        if (i == 7) {
            poolfree(ptrs[6]);
            ptrs[6] = NULL;
        }
    }

    for (int i = 0; i < 14; i++) {
        if (ptrs[i] != NULL) {
            poolfree(ptrs[i]);
        }
    }
}

void test_alloc_chars() {
    char *ptrs[20];
    for (int i = 0; i < 20; i++) {
        char *ptr = (char*) poolmalloc(sizeof(int));
        *ptr = i;
        ptrs[i] = ptr;
    }
    for (int i = 0; i < 20; i++) {
        poolfree(ptrs[i]);
    }
}

int main(void) {
    poolinit();

    test_alloc_ints();
    test_alloc_chars();
}
