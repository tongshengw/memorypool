#pragma once

#ifdef __cplusplus
extern "C" {
#endif 

#include <stdbool.h>

typedef struct BlockHeader {
    struct BlockHeader *prev;
    struct BlockHeader *next;
    unsigned long size;
    bool free;
} BlockHeader;

typedef struct BlockFooter {
    BlockHeader *headerPtr;
} BlockFooter;

void poolinit();

void *poolmalloc(unsigned long size);

void poolfree(void *ptr);

void printlayout();

void printbytes();

int dataBytes(BlockHeader *head);

int headerBytes(BlockHeader *head);

#ifdef __cplusplus
} /* extern "C" */
#endif 

