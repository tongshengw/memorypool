#include "trackedmalloc.h"
#include <assert.h>

unsigned long totalBytes = 0;

void *trackedmalloc(size_t size) {
    totalBytes += size;
    void *ptr = malloc(size);
    assert(ptr != NULL);
    return ptr;
}

unsigned long get_total_bytes() {
    return totalBytes;
}