#pragma once

#include <configure.h>

#ifdef USE_MEMORY_POOL
  #ifdef __CUDACC__
    #include <memorypool/gpu/poolalloc.cuh>
  #else
    #include <memorypool/cpu/poolalloc.h>  
  #endif

  #define swappablecalloc(count, size) poolcalloc(count, size)

  #define swappablemalloc(size) poolmalloc(size)
  #define swappablefree(ptr)    poolfree(ptr)

#else
  #define swappablecalloc(count, size) calloc(count, size)

  #define swappablemalloc(size) malloc(size)
  #define swappablefree(ptr)    free(ptr)
#endif
