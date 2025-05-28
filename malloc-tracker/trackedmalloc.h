#include <stdlib.h>

void *trackedmalloc(size_t size);

void trackedfree(void* ptr);

unsigned long get_peak_usage();