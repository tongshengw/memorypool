#ifndef POOL_ALLOC
#define POOL_ALLOC

#include <stdbool.h>

typedef struct BlockHeader {
    struct BlockHeader *prev;
    struct BlockHeader *next;
    unsigned long size;
    bool free;
} BlockHeader;

void poolinit();

void *poolmalloc(unsigned long size);

void poolfree(void *ptr);

void printlayout();

int dataBytes(BlockHeader *head);

int headerBytes(BlockHeader *head);

#endif
