#ifndef POOL_ALLOC
#define POOL_ALLOC

#include <stdbool.h>

typedef struct BlockHeader {
    unsigned long size;
    bool free;
    struct BlockHeader *prev;
    struct BlockHeader *next;
} BlockHeader;

void poolinit();

void *poolmalloc(unsigned long size);

void poolfree(void *ptr);

void printlayout();

int dataBytes(BlockHeader *head);

int headerBytes(BlockHeader *head);

#endif
