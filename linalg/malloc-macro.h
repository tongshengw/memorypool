#pragma once

#ifdef USE_POOL
  #define malloc(size) poolmalloc(size)
  #define free(ptr)    poolfree(ptr)
#endif