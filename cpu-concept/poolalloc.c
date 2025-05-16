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

static void assertNoCycle(BlockHeader *head) {
    if (head == NULL) {
        return;
    }
    BlockHeader *slow = head;
    BlockHeader *fast = head;
    while (fast != NULL && fast->next != NULL) {
        slow = slow->next;
        fast = fast->next->next;
        assert(slow != fast);
    }
}
static void assertListValid(BlockHeader *head) {
    assertNoCycle(head);
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
        assert(head->size >= head->next->size);
        assert(head->next->prev == head);
        head = head->next;
    }
}

static void listRemove(BlockHeader **head, BlockHeader **target) {
    BlockHeader *front = (*target)->next;
    BlockHeader *back = (*target)->prev;

    if (front != NULL) {
        front->prev = back;
    }
    if (back != NULL) {
        back->next = front;
    } else {
        *head = front;
    }
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

static void listSwapHeadSort(BlockHeader **head) {
    if ((*head)->next == NULL) {
        return;
    }
    BlockHeader *front = (*head)->next->next;
    BlockHeader *back = (*head)->next;
    unsigned long targetSize = (*head)->size;

    while (front != NULL) {
        unsigned long frontSize = front->size;
        unsigned long backSize = back->size;
        if (frontSize <= targetSize && targetSize <= backSize) {
            front->prev = *head;
            back->next = *head;
            BlockHeader *tmp = (*head)->next;
            (*head)->prev = back;
            (*head)->next = front;
            (*head) = tmp;
            (*head)->prev = NULL;
            return;
        }
        front = front->next;
        back = back->next;
    }
    // not found, insert at end
    if (back->size > (*head)->size) {
        BlockHeader *tmp = (*head)->next;
        back->next = *head;
        (*head)->prev = back;
        (*head)->next = NULL;
        *head = tmp;
        (*head)->prev = NULL;
    }
}

static unsigned long max(unsigned long a, unsigned long b) {
    return a > b ? a : b;
}

void poolinit() {
    BlockHeader *header = (BlockHeader *)memPool;
    // create a block that fills entire pool
    header->size = MEM_POOL_SIZE - sizeof(BlockHeader);
    header->free = true;
    listPrepend(&freeList, header);
}

void *poolmalloc(unsigned long size) {
    // all pointer arithmetic is done on char* for clarity
    // TODO: handle edge case where last alloc takes space of last header
    if (freeList == NULL || freeList->size < size) {
        return NULL;
    }

    int initFreeListSize = debugListSize(freeList);
    int initUsedListSize = debugListSize(usedList);

    unsigned long oldBlockSize = freeList->size;
    BlockHeader *newAllocatedHeader = freeList;
    unsigned long sizeToAllocate = max(size, 8);
    newAllocatedHeader->size = sizeToAllocate;
    listRemove(&freeList, &freeList);
    listPrepend(&usedList, newAllocatedHeader);
    newAllocatedHeader->free = false;
    BlockHeader *newFreeHeader =
        (BlockHeader *)((char *)newAllocatedHeader +
                        (sizeof(BlockHeader) + sizeToAllocate));
    newFreeHeader->size = oldBlockSize - (sizeToAllocate + sizeof(BlockHeader));
    newFreeHeader->free = true;
    listPrepend(&freeList, newFreeHeader);
    listSwapHeadSort(&freeList);
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

    listRemove(&usedList, &freedHeader);
    listPrepend(&freeList, freedHeader);
    freedHeader->free = true;

    listSwapHeadSort(&freeList);

    assertListValid(freeList);
    assertListValid(usedList);
    assert(debugListSize(freeList) == initFreeListSize + 1);
    assert(debugListSize(usedList) == initUsedListSize - 1);
    assertFreeListSorted(freeList);
}
