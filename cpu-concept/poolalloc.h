#ifndef POOL_ALLOC
#define POOL_ALLOC

#define MEMORY_BLOCK_SIZE 64

#include "stdbool.h"

typedef struct BlockHeader {
    bool free;
    bool size;
    struct BlockHeader *prev;
    struct BlockHeader *next;
} BlockHeader;

typedef struct BlockFooter {
    BlockHeader *header;
} BlockFooter;

typedef struct MemoryBlock {
    BlockHeader header;
    char *data;
    BlockFooter footer;
} MemoryBlock;

void *poolmalloc(unsigned long size);

void poolfree(void *ptr);

#endif
