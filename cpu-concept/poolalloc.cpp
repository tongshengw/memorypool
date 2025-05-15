#include "poolalloc.h"
#include "stdio.h"
#include <cassert>
#include <deque>

#define MEM_POOL_SIZE 32

static char memPool[MEM_POOL_SIZE];

struct BlockHeaderComp {
    bool operator()(BlockHeader *a, BlockHeader *b) {
        assert(a != nullptr && b != nullptr);
        return a->size < b->size;
    }
};
static std::deque<BlockHeader *> freeList;

static std::deque<BlockHeader *> usedList;

void poolinit() {
    BlockHeader *header = (BlockHeader *)memPool;
    // create a block that fills entire pool
    header->size = MEM_POOL_SIZE - sizeof(BlockHeader);
    freeList.push_back(header);
    for (int i = 0; i < MEM_POOL_SIZE; i++) {
        printf("%d ", memPool[i]);
    }
}

void *poolmalloc(unsigned long size) {
    if (freeList.front()->size < size) {
        return nullptr;
    }

    unsigned long oldBlockSize = freeList.front()->size;
    BlockHeader *newAllocatedHeader = freeList.front();
    newAllocatedHeader->size = size;
    freeList.pop_front();
    usedList.push_back(newAllocatedHeader);

    BlockHeader *newFreeHeader =
        (BlockHeader*) ((char*) newAllocatedHeader + (sizeof(BlockHeader) + size));
    newFreeHeader->size = oldBlockSize - (size + sizeof(BlockHeader));
    freeList.push_back(newFreeHeader);

    std::sort(freeList.begin(), freeList.end());

    return newAllocatedHeader + (sizeof(BlockHeader));
}

void poolfree(void *ptr) {
    char *poolPtr = (char*) ptr;
    BlockHeader *freedHeader = (BlockHeader*)(poolPtr - sizeof(BlockHeader));
    freeList.push_back(freedHeader);
    // TODO: coalesing
}
