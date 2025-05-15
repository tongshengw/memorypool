#include "./poolalloc.h"
#include "stdlib.h"
#include "stdio.h"
#include <assert.h>
#include <vector>

int main(void) {
    poolinit();

    std::vector<int*> ptrs;
    for (int i = 0; i < 5; i++) {
        int *ptr = (int*) poolmalloc(sizeof(int));
        *ptr = i;
        ptrs.push_back(ptr);
    }
}
