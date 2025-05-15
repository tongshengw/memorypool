#include "./poolalloc.h"
#include "stdlib.h"
#include "stdio.h"
#include <assert.h>
#include <vector>

void test_alloc_ints() {
    std::vector<int*> ptrs;
    for (int i = 0; i < 14; i++) {
        int *ptr = (int*) poolmalloc(sizeof(int));
        *ptr = i;
        ptrs.push_back(ptr);
        if (i == 7) {
            poolfree(ptrs[6]);
            ptrs[6] = nullptr;
        }
    }

    for (int *ptr : ptrs) {
        if (ptr != nullptr) {
            poolfree(ptr);
        }
    }
}

void test_alloc_chars() {
    std::vector<char*> ptrs;
    for (int i = 0; i < 20; i++) {
        char *ptr = (char*) poolmalloc(sizeof(int));
        *ptr = i;
        ptrs.push_back(ptr);
    }
    for (char *ptr : ptrs) {
        poolfree(ptr);
    }
}

int main(void) {
    poolinit();

    test_alloc_ints();
    test_alloc_chars();
}
