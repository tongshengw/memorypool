#include "poolalloc.h"
#include "assert.h"
#include "stdlib.h"

#define MEM_POOL_SIZE 2048

static char memPool[MEM_POOL_SIZE];

static BlockHeader *freeList;
static BlockHeader *usedList;

// NOTE: input pointer to data, will automatically calculate BlockHeader pointer
// address by subtracting sizeof(BlockHeader)
static BlockHeader *listLinearFind(BlockHeader *head, void *target) {
    BlockHeader *start = head;
    while (start != NULL) {
        if (start == target) {
            return start;
        }
        start = start->next;
    }
    assert(false);
}

static void listRemove(BlockHeader *target) {
    BlockHeader *front = target->next;
    BlockHeader *back = target->prev;

    front->prev = back;
    back->next = front;
}

static void listPrepend(BlockHeader *head, BlockHeader *added) {
    head->prev = added;
    added->next = head;
    added->prev = NULL;
    head = added;
}

// NOTE: only use on lists with length greater than or equal 2
static void listSwapHeadSort(BlockHeader *head) {
    assert(head->next->next != NULL);
    BlockHeader *front = head->next->next;
    BlockHeader *back = head->next;
    unsigned long targetSize = head->size;

    while (front != NULL) {
        unsigned long frontSize = front->size;
        unsigned long backSize = back->size;
        if (frontSize > targetSize && targetSize > backSize) {
            front->prev = head;
            back->next = head;
            BlockHeader *tmp = head->next;
            head->prev = back;
            head->next = front;
            head = tmp;
            return;
        }
    }

    // not found, insert at end
    back->next = head;
    BlockHeader *tmp = head;
    head->prev = back;
    head = tmp;
}

unsigned long max(unsigned long a, unsigned long b) { return a > b ? a : b; }

void poolinit() {
    BlockHeader *header = (BlockHeader *)memPool;
    // create a block that fills entire pool
    header->size = MEM_POOL_SIZE - sizeof(BlockHeader);
    listPrepend(freeList, header);
}

void *poolmalloc(unsigned long size) {
    // all pointer arithmetic is done on char* for clarity
    if (freeList->size < size) {
        return NULL;
    }

    unsigned long oldBlockSize = freeList->size;
    BlockHeader *newAllocatedHeader = freeList;
    unsigned long sizeToAllocate = max(size, 8);
    newAllocatedHeader->size = sizeToAllocate;
    listRemove(freeList);
    listPrepend(usedList, newAllocatedHeader);

    BlockHeader *newFreeHeader =
        (BlockHeader *)((char *)newAllocatedHeader +
                        (sizeof(BlockHeader) + sizeToAllocate));
    newFreeHeader->size = oldBlockSize - (sizeToAllocate + sizeof(BlockHeader));
    listPrepend(freeList, newFreeHeader);

    listSwapHeadSort(freeList);

    void *res = (char *)newAllocatedHeader + (sizeof(BlockHeader));
    return res;
}

void poolfree(void *ptr) {
    char *poolPtr = (char *)ptr;
    BlockHeader *freedHeader = (BlockHeader *)(poolPtr - sizeof(BlockHeader));
    BlockHeader *freedUsedListPtr = listLinearFind(usedList, freedHeader);
    listRemove(freedUsedListPtr);
    listPrepend(freeList, freedHeader);
}
