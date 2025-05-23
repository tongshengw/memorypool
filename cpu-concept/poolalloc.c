#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "poolalloc.h"

#define MEM_POOL_SIZE 8000
// NOTE: MAX_BLOCKS is for printlayout function, as a buffer is created
// statically, could change to dynamic
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
    while (current) {
        size++;
        current = current->next;
    }
    return size;
}

static void assertNoSizeOverflow(BlockHeader *head) {
    while (head) {
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
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        assert(slow != fast);
    }
}
static void assertListValid(BlockHeader *head) {
    assertNoCycle(head);
    BlockHeader *current = head;
    while (current) {
        if (current->next) {
            assert(current->next->prev == current);
        }
        if (current->prev) {
            assert(current->prev->next == current);
        } else {
            assert(current == head);
        }
        assert(head->size % 16 == 0);
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

static void assertFootersValid(BlockHeader *head) {
    while (head) {
        char *tmp = (char *)head;
        BlockFooter *footer =
            (BlockFooter *)(tmp + sizeof(BlockHeader) + head->size);
        assert(footer->headerPtr == head);
        head = head->next;
    }
}

static void listRemove(BlockHeader **head, BlockHeader *target) {
    BlockHeader *front = target->next;
    BlockHeader *back = target->prev;

    if (front) {
        front->prev = back;
    }
    if (back) {
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

    while (front) {
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
    // not found, insert at end, if statement is for special case where list is
    // length 2
    if (back->size > (*head)->size) {
        BlockHeader *tmp = (*head)->next;
        back->next = *head;
        (*head)->prev = back;
        (*head)->next = NULL;
        *head = tmp;
        (*head)->prev = NULL;
    }
}

static bool listLinearFind(BlockHeader *head, BlockHeader *target) {
    while (head) {
        if (head == target) {
            return true;
        }
        head = head->next;
    }
    return false;
}

// static inline unsigned long max(unsigned long a, unsigned long b) {
//     return a > b ? a : b;
// }
// 

static BlockFooter *getBlockFooter(BlockHeader *header) {
    return (BlockFooter *)((char *)header + sizeof(BlockHeader) +
                           header->size);
}

void poolinit() {
    memPool = malloc(MEM_POOL_SIZE);
    BlockHeader *header = (BlockHeader *)memPool;
    // create a block that fills entire pool
    unsigned long dataSize =
        MEM_POOL_SIZE - sizeof(BlockHeader) -
        (sizeof(BlockFooter) + 16 - sizeof(BlockFooter) % 16);
    unsigned long dataSizeAligned = dataSize - dataSize % 16;
    header->size = dataSizeAligned;
    header->free = true;

    BlockFooter *footer = (BlockFooter *)((char *)memPool + dataSizeAligned);
    footer->headerPtr = header;
    listPrepend(&freeList, header);
}

int dataBytes(BlockHeader *head) {
    int tally = 0;
    while (head) {
        tally += head->size;
        head = head->next;
    }
    return tally;
}

int headerBytes(BlockHeader *head) {
    int tally = 0;
    while (head) {
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
    unsigned long dataSizeToAllocate = size + (16 - size % 16);
    unsigned long totalSizeUnaligned = dataSizeToAllocate + sizeof(BlockFooter);
    unsigned long totalSizeAligned =
        totalSizeUnaligned + (16 - totalSizeUnaligned % 16);
    newAllocatedHeader->size = dataSizeToAllocate;
    BlockFooter *newAllocatedFooter = getBlockFooter(newAllocatedHeader);
    newAllocatedFooter->headerPtr = newAllocatedHeader;
    listRemove(&freeList, freeList);
    listPrepend(&usedList, newAllocatedHeader);
    newAllocatedHeader->free = false;
    BlockHeader *newFreeHeader =
        (BlockHeader *)((char *)newAllocatedHeader +
                        (sizeof(BlockHeader) + totalSizeAligned));
    newFreeHeader->size =
        oldBlockSize - (dataSizeToAllocate + sizeof(BlockHeader));
    newFreeHeader->free = true;
    BlockFooter *newFreeFooter = getBlockFooter(newFreeHeader);
    newFreeFooter->headerPtr = newFreeHeader;
    listPrepend(&freeList, newFreeHeader);
    listSwapHeadSort(&freeList);
    void *res = (char *)newAllocatedHeader + (sizeof(BlockHeader));

    assertListValid(freeList);
    assertListValid(usedList);
    assertFootersValid(usedList);
    assertFootersValid(freeList);
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
    listRemove(&usedList, freedHeader);

    // coalescing: current only coalescing forwards because need to linear
    // search through freelist to find and remove any headers, coalescing
    // forwards doesn't need to remove any headers

    bool canCoalesce = false;
    BlockHeader *prevBlockHeader = NULL;
    BlockFooter *prevBlockFooter = NULL;
    if ((char *)freedHeader == memPool) {
        canCoalesce = false;
    } else {
        prevBlockFooter = (BlockFooter *)((char *)freedHeader - 16);
        prevBlockHeader = prevBlockFooter->headerPtr;
        assert(listLinearFind(freeList, prevBlockHeader) ||
               listLinearFind(usedList, prevBlockHeader));
        canCoalesce = prevBlockHeader->free;
    }

    if (canCoalesce) {
        // coalesce forwards
        BlockFooter *freedFooter =
            (BlockFooter *)((char *)freedHeader + freedHeader->size +
                            sizeof(BlockHeader));
        assert(freedFooter->headerPtr == freedHeader);
        prevBlockHeader->size =
            (char *)freedFooter - (char *)prevBlockHeader - sizeof(BlockHeader);
        freedFooter->headerPtr = prevBlockHeader;

        assert(((BlockFooter *)((char *)prevBlockHeader +
                                (sizeof(BlockHeader) + prevBlockHeader->size)))
                   ->headerPtr == prevBlockHeader);

        // TODO: change this workaround that is likely pretty slow
        listRemove(&freeList, prevBlockHeader);
        listPrepend(&freeList, prevBlockHeader);
        listSwapHeadSort(&freeList);

        assert(debugListSize(freeList) == initFreeListSize);
        assert(debugListSize(usedList) == initUsedListSize - 1);
    } else {
        // no coalesce, add to freelist
        listPrepend(&freeList, freedHeader);
        freedHeader->free = true;
        listSwapHeadSort(&freeList);

        assert(debugListSize(freeList) == initFreeListSize + 1);
        assert(debugListSize(usedList) == initUsedListSize - 1);
    }

    assertListValid(freeList);
    assertListValid(usedList);
    assertFootersValid(freeList);
    assertFootersValid(usedList);
    assertFreeListSorted(freeList);
}

int BlockHeaderPtrLess(const void *a, const void *b) {
    BlockHeader **aptr = (BlockHeader **)a;
    BlockHeader **bptr = (BlockHeader **)b;
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
    while (freeListTraverse) {
        headers[numHeaders] = freeListTraverse;
        numHeaders++;
        freeListTraverse = freeListTraverse->next;
    }
    while (usedListTraverse) {
        headers[numHeaders] = usedListTraverse;
        numHeaders++;
        usedListTraverse = usedListTraverse->next;
    }

    // sorts headers based on address as they are usually sorted by size or
    // recency
    qsort(headers, numHeaders, sizeof(BlockHeader *), BlockHeaderPtrLess);

    printf("Memory Layout (total size %d), size not incl headers:\n",
           MEM_POOL_SIZE);
    printf("free: |");
    for (int i = 0; i < numHeaders; i++) {
        if (headers[i]->free) {
            printf(" %lu |", headers[i]->size);
        } else {
            int numberLen =
                (int)(log((double)headers[i]->size) / log((double)10)) + 1;
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
            int numberLen =
                (int)(log((double)headers[i]->size) / log((double)10)) + 1;
            printf(" ");
            for (int i = 0; i < numberLen; i++) {
                printf(" ");
            }
            printf(" |");
        }
    }
    printf("\n");
}

void printbytes() {
    printf("\nBytes:\n");
    for (size_t i = 0; i < MEM_POOL_SIZE; i++) {
        printf("%02X", memPool[i]);
        if (i < MEM_POOL_SIZE - 1) {
            printf(" ");
        }
    }
    printf("\n");
    fflush(stdout);
}