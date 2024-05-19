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

// Optimized aligned CPU -> GPU data copy function
__inline__ __device__ void byte_memcpy_warp(const void *dest, const void *src, size_t size) 
{
    size *= sizeof(D_WORD) / sizeof(char);
    int threadId = threadIdx.x % DWARP_SIZE;
    __syncwarp();
    int offset = (GPU_CL_SIZE - (((uint64_t)src) % GPU_CL_SIZE)) / sizeof(char);
    for(int k = threadId + offset; k < size; k += DWARP_SIZE) {
        volatile char *dest_arr = ((char *)dest) + k;
        char *src_arr  = ((char *)src)  + k;
        *dest_arr = *src_arr;
    }
    for(int k = threadId; k < size; k += DWARP_SIZE) {
        volatile char *dest_arr = ((char *)dest) + k;
        char *src_arr  = ((char *)src)  + k;
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
    val = __shfl_sync(mask, val, 0);
    return val;
#else
    return __reduce_min_sync(mask, val);
#endif
}

// Helper to find min value in warp
__inline__ __device__ int32_t warpInclusiveScanSync(unsigned mask, int32_t val)
{
    for (int offset = 1; offset < DWARP_SIZE; offset <<= 1) {
        val += __shfl_up_sync(mask, val, offset);
        // Needed because non-offset elements just add themselves
        if(threadIdx.x % DWARP_SIZE < offset)
            val /= 2;
    }
    return val;
}

// Helper to find min value in warp
__inline__ __device__ int32_t warpExclusiveScanSync(unsigned mask, int32_t val)
{
    int32_t initial_val = val;
    for (int offset = 1; offset < DWARP_SIZE; offset <<= 1) {
        val += __shfl_up_sync(mask, val, offset);
        // Needed because non-offset elements just add themselves
        if(threadIdx.x % DWARP_SIZE < offset)
            val /= 2;
    }
    return val - initial_val;
}

#endif