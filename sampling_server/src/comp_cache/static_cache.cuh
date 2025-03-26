#ifndef STATICCACHE_H
#define STATICCACHE_H
#include "cache_helper.cuh"
#include "ibp_helpers.cuh"

class StaticCache {
public:
  StaticCache() {
    dev_cache_storage = nullptr;
    dev_cache_key = nullptr;
    dev_cache_offset = nullptr;
    host_cache_storage = nullptr;
    // TODO AKKAMATH: Make this configurable
    num_ways = 32;
    num_gpus = 0;
    feature_len = 0;
    cache_capacity = 0;
  }
  ~StaticCache() {
    // Free device memory
    if(dev_cache_storage != nullptr)
      cudaFree(dev_cache_storage);
    cudaCheckError();
    if(dev_cache_key != nullptr)
      cudaFree(dev_cache_key);
    cudaCheckError();
    if(dev_cache_offset != nullptr)
      cudaFree(dev_cache_offset);
    cudaCheckError();
  }
  // One-time init function for cache parameters and seed cache values
  void init_cache(int64_t nodes_per_gpu, int32_t feature_len, float *cpu_features,
    int *index_array, int Kg, int dev_start, int64_t total_nodes, DYN_FLAGS flags, int ways = 32);
  // CPU-side functions for cache access
  void insert_features(void *cache, int64_t num_nodes, int gpu_id, float *input_feats, int32_t *index_array,
    int64_t total_nodes, int *failed_inserts = nullptr);
  void insert_features_compressed(int64_t &nodes_per_gpu, float *input_feats, int32_t *index_array,
    int64_t total_nodes, int dev_start, int num_gpus, int *failed_inserts = nullptr);
  void test_lookup_features(int64_t num_nodes, float *input_features, int32_t *index_array,
    int *success_lookups = nullptr, int *keys_found = nullptr);
  void retrieve(int32_t *nodeIds, int64_t num_nodes, int64_t *node_index, cudaStream_t stream,
      ull *misses = nullptr, ull *lookups = nullptr, ull *inserts = nullptr);
  void transfer(int32_t *nodeIds, int64_t num_nodes, float *output_buffer, int64_t* cache_index,
      float *input_feats, int total_nodes, cudaStream_t stream, ull *misses = nullptr,
      ull *accesses = nullptr, ull* inserts = nullptr);
private:
    // Actual cache storage
    void **dev_cache_storage;
    void ** host_cache_storage;
    // Metadata to track storage
    int **dev_cache_key;
    ull **dev_cache_offset;
    // Other relevant params
    int32_t num_gpus;
    int32_t dev_start;
    int32_t feature_len;
    int32_t compress_len;
    int32_t chunk_size;
    int64_t cache_capacity;
    int32_t num_sets;
    int32_t num_ways;
    DYN_FLAGS flags;
    int32_t *comp_mask;
    int32_t *comp_bitval;
    int32_t *comp_bitmask;
    int *maxShmem;
};
#endif //STATICCACHE_H