#pragma once

#include <configure.h>

#ifdef USE_MEMORY_POOL
  #ifdef __CUDACC__
    #include <memorypool/gpu/poolalloc.cuh>
  #else
    #include <memorypool/cpu/poolalloc.h>  
  #endif
  #define swappablemalloc(size) poolmalloc(size)
  #define swappablefree(ptr)    poolfree(ptr)
#endif
