#ifndef CACHE_HELPER
#define CACHE_HELPER
#include <iostream>
#include <vector>
// Timing definitions
#include <chrono>
#define TIME_NOW std::chrono::steady_clock::now()
#define TIME_DIFF(start, end) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
// Commonly used constants
#define GPU_CL_SIZE 128
#define DWARP_SIZE 32
#define GPU_CL_FLOATS (GPU_CL_SIZE / sizeof(float))
#define D_WORD float
#define D_BYTE char
#define FULL_MASK 0xffffffff
#define ull unsigned long long

#define BITS_TO_BYTES(x) ((x + 7) / 8)

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

#define CUDA_CHECK(a) \
    { a; cudaCheckError(); }

enum DYN_FLAGS : int {
  DYN_ENABLE   = (1<<0),
  DYN_SHADOW   = (1<<1),
  DYN_PROXIM   = (1<<2),
  DYN_COMP     = (1<<3),
  DYN_COMP_CPU = (1<<4),
  DYN_OPT      = (1<<5),
  DYN_UNOPT    = (1<<6),
  DYN_COMP_TEST= (1<<7),
  DYN_CPU_TEST2= (1<<8),
};

typedef int LRU;
typedef int LFU;


// Helper to find min value in warp
__inline__ __device__ int reduceMinSync(unsigned mask, int val)
{
# if __CUDA_ARCH__ < 800
    for (int offset = DWARP_SIZE / 2; offset > 0; offset /= 2) {
        int tmpVal = __shfl_down_sync(mask, val, offset);
        if (tmpVal < val) {
            val = tmpVal;
        }
    }
    val = __shfl_sync(mask, val, 0);
    return val;
#else
    return __reduce_min_sync(mask, val);
#endif
}


#endif