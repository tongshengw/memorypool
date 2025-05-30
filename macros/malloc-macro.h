#pragma once

#ifdef USE_POOL
  #include <poolalloc.h>
  #define malloc(size) poolmalloc(size)
  #define free(ptr)    poolfree(ptr)
#endif