#ifndef DYNCACHE_H
#define DYNCACHE_H
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

class DynamicCache {
public:
  DynamicCache() {
    dev_cache_storage = nullptr;
    dev_cache_key = nullptr;
    dev_cache_lru = nullptr;
    lru_counter = 0;
    // TODO AKKAMATH: Make this configurable
    num_ways = 32;
    num_gpus = 0;
    feature_len = 0;
    cache_capacity = 0;
  }
  ~DynamicCache() {
    // Free device memory
    if(dev_cache_storage != nullptr)
      cudaFree(dev_cache_storage);
    cudaCheckError();
    if(dev_cache_key != nullptr)
      cudaFree(dev_cache_key);
    cudaCheckError();
    if(dev_cache_lru != nullptr)
      cudaFree(dev_cache_lru);
    cudaCheckError();
  }
  // One-time init function for cache parameters and seed cache values
  void init_cache(int64_t nodes_per_gpu, int32_t feature_len, 
                  float *cpu_features, int *index_array, int Kg, 
                  int dev_start, int64_t total_nodes);
  // CPU-side functions for cache access
  void insert_features(int64_t num_nodes, float *input_feats, int32_t *index_array, 
    int64_t total_nodes, int *failed_inserts = nullptr);
  void test_lookup_features(int64_t num_nodes, float *input_features, int32_t *index_array, 
    int *success_lookups = nullptr);

private:
    // Actual cache storage
    void **dev_cache_storage;
    // Metadata to track storage
    int **dev_cache_key;
    int **dev_cache_lru;
    int lru_counter;
    // Other relevant params
    int32_t num_gpus;
    int32_t feature_len;
    int64_t cache_capacity;
    int32_t num_sets;
    // TODO AKKAMATH: Make this configurable
    int32_t num_ways;
};
#endif