#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "poolalloc.h"

#define MEM_POOL_SIZE 128
// NOTE: MAX_BLOCKS is for printlayout function, as a buffer is created statically, could change to dynamic
#define MAX_BLOCKS 100

// Alignment
// For now, headers are 16 aligned
// H = Header
// P = Padding
// D = Data
// 0   16  32  48  64
// |P|H|D|D|P|H| | | |
// | | | | | | | | | |

// blockheader type to ensure first one is aligned
// TODO: when switching to cuda, remove malloc() in poolinit function
// using malloc because malloc aligns to 16 on my system
// static BlockHeader memPool[(MEM_POOL_SIZE/sizeof(BlockHeader))+1];
char *memPool = NULL;

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

static void assertNoSizeOverflow(BlockHeader *head) {
    while (head != NULL) {
        assert(head->size < MEM_POOL_SIZE);
        head = head->next;
    }
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
    // not found, insert at end, if statement is for special case where list is length 2
    if (back->size > (*head)->size) {
        BlockHeader *tmp = (*head)->next;
        back->next = *head;
        (*head)->prev = back;
        (*head)->next = NULL;
        *head = tmp;
        (*head)->prev = NULL;
    }
}

// static inline unsigned long max(unsigned long a, unsigned long b) {
//     return a > b ? a : b;
// }

void poolinit() {
    memPool = malloc(MEM_POOL_SIZE);
    BlockHeader *header = (BlockHeader *)memPool;
    // create a block that fills entire pool
    header->size = MEM_POOL_SIZE - sizeof(BlockHeader);
    header->free = true;
    listPrepend(&freeList, header);
}

int dataBytes(BlockHeader *head) {
    int tally = 0;
    while (head != NULL) {
        tally += head->size;
        head = head->next;
    }
    return tally;
}

int headerBytes(BlockHeader *head) {
    int tally = 0;
    while (head != NULL) {
        tally += sizeof(BlockHeader);
        head = head->next;
    }
    return tally;
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
    unsigned long sizeToAllocate = size + (16 - size % 16);
    newAllocatedHeader->size = sizeToAllocate;
    listRemove(&freeList, &freeList);
    listPrepend(&usedList, newAllocatedHeader);
    newAllocatedHeader->free = false;
    // TODO: the following line needs to be modified for memory alignment purposes (not sure how it works atm)
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
    assertNoSizeOverflow(freeList);
    assertNoSizeOverflow(usedList);
    assert(debugListSize(freeList) == initFreeListSize);
    assert(debugListSize(usedList) == initUsedListSize + 1);
    assertFreeListSorted(freeList);
    return res;
}

void poolfree(void *ptr) {
    int initFreeListSize = debugListSize(freeList);
    int initUsedListSize = debugListSize(usedList);
    
    if (usedList->next == NULL) {
        freeList = NULL;
        usedList = NULL;
        poolinit();
        return;
    }

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

int BlockHeaderPtrLess(const void *a, const void *b) {
    BlockHeader **aptr = (BlockHeader**) a;
    BlockHeader **bptr = (BlockHeader**) b;
    return *aptr - *bptr;
}

// array based
/*
free: | 1028 |       | 8 \
used: |      | 8 | 8 |
*/
void printlayout() {
    BlockHeader const *headers[MAX_BLOCKS];
    for (int i = 0; i < MAX_BLOCKS; i++) {
        headers[i] = NULL;
    }
    
    int numHeaders = 0;
    BlockHeader *freeListTraverse = freeList;
    BlockHeader *usedListTraverse = usedList;
    while (freeListTraverse != NULL) {
        headers[numHeaders] = freeListTraverse;
        numHeaders++;
        freeListTraverse = freeListTraverse->next;
    }
    while (usedListTraverse != NULL) {
        headers[numHeaders] = usedListTraverse;
        numHeaders++;
        usedListTraverse = usedListTraverse->next;
    }
    
    // sorts headers based on address as they are usually sorted by size or recency
    qsort(headers, numHeaders, sizeof(BlockHeader*), BlockHeaderPtrLess);
    
    printf("Memory Layout (total size %d), size not incl headers:\n", MEM_POOL_SIZE);
    printf("free: |");
    for (int i = 0; i < numHeaders; i++) {
        if (headers[i]->free) {
            printf(" %lu |", headers[i]->size);
        } else {
            int numberLen = (int) (log((double) headers[i]->size) / log((double) 10)) + 1;
            printf(" ");
            for (int i = 0; i < numberLen; i++) {
                printf(" ");
            }
            printf(" |");
        }
    }
    printf("\n");
    printf("used: |");
    for (int i = 0; i < numHeaders; i++) {
        if (!headers[i]->free) {
            printf(" %lu |", headers[i]->size);
        } else {
            int numberLen = (int) (log((double) headers[i]->size) / log((double) 10)) + 1;
            printf(" ");
            for (int i = 0; i < numberLen; i++) {
                printf(" ");
            }
            printf(" |");
        }
    }
    printf("\n");
}