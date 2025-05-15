#include "poolalloc.h"
#include <algorithm>
#include <cassert>
#include <deque>

#define MEM_POOL_SIZE 2048

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
}

void *poolmalloc(unsigned long size) {
    // all pointer arithmetic is done on char* for clarity
    if (freeList.front()->size < size) {
        return nullptr;
    }

    unsigned long oldBlockSize = freeList.front()->size;
    BlockHeader *newAllocatedHeader = freeList.front();
    unsigned long sizeToAllocate = std::max<unsigned long>(size, 8);
    newAllocatedHeader->size = sizeToAllocate;
    freeList.pop_front();
    usedList.push_back(newAllocatedHeader);

    BlockHeader *newFreeHeader =
        (BlockHeader*) ((char*) newAllocatedHeader + (sizeof(BlockHeader) + sizeToAllocate));
    newFreeHeader->size = oldBlockSize - (sizeToAllocate + sizeof(BlockHeader));
    freeList.push_back(newFreeHeader);

    std::sort(freeList.begin(), freeList.end());

    void *res = (char*) newAllocatedHeader + (sizeof(BlockHeader));
    return res;
}

void poolfree(void *ptr) {
    char *poolPtr = (char*) ptr;
    BlockHeader *freedHeader = (BlockHeader*)(poolPtr - sizeof(BlockHeader));
    freeList.push_back(freedHeader);
    // TODO: coalesing
}
