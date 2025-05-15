#include "poolalloc.h"
#include "stdlib.h"

#define MEM_POOL_SIZE 1024

static char memPool[MEM_POOL_SIZE];

void *poolmalloc(unsigned long size) {
    return memPool;
}

void poolfree(void *ptr) {
    return;
}
