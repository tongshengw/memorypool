#include "./poolalloc.h"
#include "stdlib.h"
#include "stdio.h"
#include <assert.h>

int main(void) {
    int LEN = 4;
    int *arr1 = malloc(LEN * sizeof(int));
    if (arr1 == NULL) {
        printf("arr1 malloc error\n");
    }
    arr1[0] = 10;
    arr1[1] = 13;
    arr1[2] = 2;

    int *arr2 = malloc(LEN * sizeof(int));
    if (arr2 == NULL) {
        printf("arr1 malloc error\n");
    }

    int *parr1 = poolmalloc(LEN * sizeof(int));
    parr1[0] = 10;
    parr1[1] = 13;
    parr1[2] = 2;


    for (int i = 0; i < LEN; i++) {
        assert(arr1[i] == parr1[i]);
    }


    free(arr1);
    free(arr2);
    poolfree(parr1);
}
