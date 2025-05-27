#include <stdio.h>
#include "trackedmalloc.h"

int main(void) {
    int *ptr1 = trackedmalloc(100);
    int *ptr2 = trackedmalloc(200);
    int *ptr3 = trackedmalloc(300);
    
    printf("Total bytes allocated: %lu\n", get_total_bytes());
}