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

enum DYN_FLAGS : int {
  DYN_ENABLE = (1<<0),
  DYN_SHADOW = (1<<1),
  DYN_PROXIM = (1<<2),
};

typedef int LRU;
typedef int LFU;

// Optimized padded, aligned CPU -> GPU data copy function
// Copies 4 bytes at a time per thread
// Tested on floating point transfer
__inline__ __device__ void memcpy_warp_isspace(const void *dest, const void *src, size_t size) 
{
    int threadId = threadIdx.x % DWARP_SIZE;
    const int offset = (GPU_CL_SIZE - (((uint64_t)src)  % GPU_CL_SIZE)) / sizeof(D_WORD);
    const int onset  = (GPU_CL_SIZE - (((uint64_t)size * sizeof(D_WORD)) % GPU_CL_SIZE)) / sizeof(D_WORD);
    for(int k = threadId + offset; k < size + offset + onset; k += DWARP_SIZE) {
        __syncwarp();
        D_WORD *dest_element = ((D_WORD *)dest) + k;
        // Combined read, to improve PCIe util.
        D_WORD src_data = *(((D_WORD *)src) + k);
        // Selective write to GPU memory
        if(k < size)
            *dest_element = src_data;
    }

    int sub_val = (offset + 7) / 8 * 8; // Round up to nearest multiple of 8
    for(int k = threadId + (offset - sub_val); k < offset; k += DWARP_SIZE) {
        D_WORD *dest_element = ((D_WORD *)dest) + k;
        // Combined read, to improve PCIe util.
        D_WORD src_data = *(((D_WORD *)src) + k);
        // Selective write to GPU memory
        if(k >= 0)
            *dest_element = src_data;
    }
    __syncwarp();
}

// Optimized aligned CPU -> GPU data copy function
__inline__ __device__ void memcpy_warp(const void *dest, const void *src, size_t size) 
{
    int threadId = threadIdx.x % DWARP_SIZE;
    __syncwarp();
    int offset = (GPU_CL_SIZE - (((uint64_t)src) % GPU_CL_SIZE)) / sizeof(D_WORD);
    for(int k = threadId + offset; k < size; k += DWARP_SIZE) {
        volatile D_WORD *dest_arr = ((D_WORD *)dest) + k;
        D_WORD *src_arr  = ((D_WORD *)src)  + k;
        *dest_arr = *src_arr;
    }
    for(int k = threadId; k < size; k += DWARP_SIZE) {
        volatile D_WORD *dest_arr = ((D_WORD *)dest) + k;
        D_WORD *src_arr  = ((D_WORD *)src)  + k;
        *dest_arr = *src_arr;
    }
}

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
    return val;
#else
    return __reduce_min_sync(mask, val);
#endif
}

#endif