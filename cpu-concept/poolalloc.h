#ifndef POOL_ALLOC
#define POOL_ALLOC

typedef struct BlockHeader {
    unsigned long size;
} BlockHeader;

// typedef struct BlockFooter {
//     BlockHeader *header;
// } BlockFooter;

typedef struct MemoryBlock {
    BlockHeader header;
    char *data;
} MemoryBlock;

void poolinit();

void *poolmalloc(unsigned long size);

void poolfree(void *ptr);

#endif
