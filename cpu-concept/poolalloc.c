#include "poolalloc.h"
#include "assert.h"
#include "stdlib.h"

#define MEM_POOL_SIZE 2048

static char memPool[MEM_POOL_SIZE];

static BlockHeader *freeList;
static BlockHeader *usedList;

// NOTE:: Debug functions
static int debugListSize(BlockHeader *head) {
    int size = 0;
    BlockHeader *current = head;
    while (current != NULL) {
        size++;
        current = current->next;
    }
    return size;
}

static void assertListValid(BlockHeader *head) {
    BlockHeader *current = head;
    while (current != NULL) {
        if (current->next != NULL) {
            assert(current->next->prev == current);
        }
        if (current->prev != NULL) {
            assert(current->prev->next == current);
        } else {
            assert(current == head);
        }
        current = current->next;
    }
}

static void assertFreeListSorted(BlockHeader *head) {
    while (head && head->next) {
        assert(head->size <= head->next->size);
        assert(head->next->prev == head);
        head = head->next;
    }
}

// NOTE: input pointer to data, will automatically calculate BlockHeader pointer
// address by subtracting sizeof(BlockHeader)
// FIXME: double pointer head
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

static void listPrepend(BlockHeader **head, BlockHeader *added) {
    if (*head == NULL) {
        *head = added;
        added->prev = NULL;
        added->next = NULL;
        return;
    }
    (*head)->prev = added;
    added->next = *head;
    added->prev = NULL;
    *head = added;
}

// NOTE: only use on lists with length greater than or equal 2
// FIXME: double pointer head
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

static unsigned long max(unsigned long a, unsigned long b) {
    return a > b ? a : b;
}

void poolinit() {
    BlockHeader *header = (BlockHeader *)memPool;
    // create a block that fills entire pool
    header->size = MEM_POOL_SIZE - sizeof(BlockHeader);
    listPrepend(&freeList, header);
}

void *poolmalloc(unsigned long size) {
    // all pointer arithmetic is done on char* for clarity
    // TODO: handle edge case where last alloc takes space of last header
    if (freeList->size < size) {
        return NULL;
    }

    int initFreeListSize = debugListSize(freeList);
    int initUsedListSize = debugListSize(usedList);

    unsigned long oldBlockSize = freeList->size;
    BlockHeader *newAllocatedHeader = freeList;
    unsigned long sizeToAllocate = max(size, 8);
    newAllocatedHeader->size = sizeToAllocate;
    listRemove(freeList);
    listPrepend(&usedList, newAllocatedHeader);
    BlockHeader *newFreeHeader =
        (BlockHeader *)((char *)newAllocatedHeader +
                        (sizeof(BlockHeader) + sizeToAllocate));
    newFreeHeader->size = oldBlockSize - (sizeToAllocate + sizeof(BlockHeader));
    listPrepend(&freeList, newFreeHeader);
    listSwapHeadSort(freeList);
    void *res = (char *)newAllocatedHeader + (sizeof(BlockHeader));

    assertListValid(freeList);
    assertListValid(usedList);
    assert(debugListSize(freeList) == initFreeListSize);
    assert(debugListSize(usedList) == initUsedListSize + 1);
    assertFreeListSorted(freeList);
    return res;
}

void poolfree(void *ptr) {
    int initFreeListSize = debugListSize(freeList);
    int initUsedListSize = debugListSize(usedList);

    char *poolPtr = (char *)ptr;
    BlockHeader *freedHeader = (BlockHeader *)(poolPtr - sizeof(BlockHeader));
    BlockHeader *freedUsedListPtr = listLinearFind(usedList, freedHeader);
    listRemove(freedUsedListPtr);
    listPrepend(&freeList, freedHeader);

    assertListValid(freeList);
    assertListValid(usedList);
    assert(debugListSize(freeList) == initFreeListSize + 1);
    assert(debugListSize(usedList) == initUsedListSize - 1);
    assertFreeListSorted(freeList);
}
