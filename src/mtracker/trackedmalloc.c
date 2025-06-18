#include <trackedmalloc.h>
#include <assert.h>
#include <stdbool.h>

// have to use custom tracking because malloc_usable_size() not available in
// macos
typedef struct {
    size_t size;
    void *ptr;
} AllocatedPtr;

static size_t curUsage = 0;
static size_t maxUsage = 0;

static AllocatedPtr *allocatedPtrs;
static size_t allcoatedPtrsSize = 0;
static size_t allocatedPtrsCapacity = 0;

static void allocatedPtrsAdd(void *ptr, size_t size) {
    if (allocatedPtrsCapacity == 0) {
        allocatedPtrsCapacity = 16;
        allocatedPtrs = malloc(allocatedPtrsCapacity * sizeof(AllocatedPtr));
        assert(allocatedPtrs != NULL);
    } else if (allcoatedPtrsSize >= allocatedPtrsCapacity) {
        allocatedPtrsCapacity *= 2;
        AllocatedPtr *newAllocatedPtrs = realloc(
            allocatedPtrs, allocatedPtrsCapacity * sizeof(AllocatedPtr));
        return;
    }
    allocatedPtrs[allcoatedPtrsSize].ptr = ptr;
    allocatedPtrs[allcoatedPtrsSize].size = size;
    allcoatedPtrsSize++;
    curUsage += size;
    maxUsage = (curUsage > maxUsage) ? curUsage : maxUsage;
}

static void allocatedPtrsRemove(void *ptr) {
    for (size_t i = 0; i < allcoatedPtrsSize; i++) {
        if (allocatedPtrs[i].ptr == ptr) {
            curUsage -= allocatedPtrs[i].size;
            allocatedPtrs[i] = allocatedPtrs[allcoatedPtrsSize - 1];
            allcoatedPtrsSize--;
            return;
        }
    }
    assert(false);
}

void *trackedmalloc(size_t size) {
    void *ptr = malloc(size);
    allocatedPtrsAdd(ptr, size);
    assert(ptr != NULL);
    return ptr;
}

void trackedfree(void *ptr) {
    allocatedPtrsRemove(ptr);
    free(ptr);
}

size_t get_peak_usage() { return maxUsage; }