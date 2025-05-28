#include <stdio.h>
#include "trackedmalloc.h"

int main(void) {
    int *ptr1 = trackedmalloc(100);
    int *ptr2 = trackedmalloc(200);
    trackedfree(ptr1);
    int *ptr3 = trackedmalloc(300);
    int *ptr4 = trackedmalloc(400);
    trackedfree(ptr2);
    
    printf("Peak memory usage: %lu\n", get_peak_usage());
}