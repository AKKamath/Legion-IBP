#include "dyn_cache.cuh"

// Optimized padded, aligned CPU -> GPU data copy function
// Copies 4 bytes at a time per thread
// Tested on floating point transfer
__inline__ __device__ void memcpy_warp_isspace(void *dest, void *src, size_t size) 
{
    int threadId = threadIdx.x % DWARP_SIZE;
    const int offset = (GPU_CL_SIZE - (((uint64_t)src)  % GPU_CL_SIZE)) / sizeof(D_WORD);
    const int onset  = (GPU_CL_SIZE - (((uint64_t)size * sizeof(D_WORD)) % GPU_CL_SIZE)) / sizeof(D_WORD);
    for(int k = threadId + offset; k < size + offset + onset; k += DWARP_SIZE) {
        __syncwarp();
        D_WORD *dest_arr = ((D_WORD *)dest) + k;
        D_WORD *src_arr  = ((D_WORD *)src)  + k;
        // Combined read, to improve PCIe util.
        D_WORD data = *src_arr;
        // Selective write to GPU memory
        if(k < size)
            *dest_arr = data;
    }
    
    for(int k = threadId + (offset - GPU_CL_FLOATS); k < offset; k += DWARP_SIZE) {
        __syncwarp();
        D_WORD *dest_arr = ((D_WORD *)dest) + k;
        D_WORD *src_arr  = ((D_WORD *)src)  + k;
        // Combined read, to improve PCIe util.
        D_WORD data = *src_arr;
        // Selective write to GPU memory
        if(k >= 0)
            *dest_arr = data;
    }
}

// Optimized aligned CPU -> GPU data copy function
__inline__ __device__ void memcpy_warp(void *dest, void *src, size_t size) 
{
    int threadId = threadIdx.x % DWARP_SIZE;
    __syncwarp();
    int offset = (GPU_CL_SIZE - (((uint64_t)src) % GPU_CL_SIZE)) / sizeof(D_WORD);
    for(int k = threadId + offset; k < size; k += DWARP_SIZE) {
        D_WORD *dest_arr = ((D_WORD *)dest) + k;
        D_WORD *src_arr  = ((D_WORD *)src)  + k;
        *dest_arr = *src_arr;
    }
    for(int k = threadId; k < size; k += DWARP_SIZE) {
        D_WORD *dest_arr = ((D_WORD *)dest) + k;
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

// Helper functions for hashing keys
// TODO AKKAMATH: Add fancy hash algorithms
__inline__ __device__ int64_t key_to_set_index(int64_t key, int64_t num_sets, int64_t num_gpus) 
{
    return key % (num_sets / num_gpus);
}

__inline__ __device__ int64_t key_to_gpu_index(int64_t key, int64_t num_sets, int64_t num_gpus) 
{
    return (key % num_sets) / (num_sets / num_gpus);
}

__global__ void reset_cache_metadata(int *cache_keys, int *cache_lru, int64_t nodes) 
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = threadId; i < nodes; i += blockDim.x * gridDim.x) {
        cache_keys[i] = -1;
        cache_lru[i] = -1;
    }
}

// Returns index of inserted feature or -1 if failed to insert
__inline__ __device__ int insert_single_feature(
    void **dev_cache_storage, int **dev_cache_keys, int **dev_cache_lru, 
    int64_t key, int64_t feature_len, int lru_counter, void *input_feature, 
    int num_gpus, int num_ways, int32_t num_sets, bool not_fringe)
{
    // Get details of set to insert into
    int devId = key_to_gpu_index(key, num_sets, num_gpus);
    int setId = key_to_set_index(key, num_sets, num_gpus);
    float *cache = (float*)dev_cache_storage[devId];
    int *cache_keys = dev_cache_keys[devId];
    int *cache_lru = dev_cache_lru[devId];

    int laneId = threadIdx.x % DWARP_SIZE;
    int index = laneId + setId * num_ways;
    int lru = cache_lru[index];
    // Attempt to find slot(s) with minimum LRU
    int min_lru = reduceMinSync(FULL_MASK, lru);
    int candidates = __ballot_sync(FULL_MASK, lru == min_lru && lru < lru_counter);
    int empty_slot = __ffs(candidates) - 1;
    while(empty_slot >= 0) {
        // Empty slot found
        // Attempt to acquire exclusivity via atomic
        int acquired = 0;
        if(laneId == empty_slot) {
            acquired = (atomicCAS(&cache_lru[index], lru, lru_counter) == lru);
        }
        int success = __ballot_sync(FULL_MASK, acquired);

        if(success) {
            index = empty_slot + setId * num_ways;
            if(laneId == empty_slot) {
                atomicExch(&cache_keys[index], (int)key);
            }
            __syncwarp();
            // Copy the features
            if(not_fringe) {
                // If we have extra space on both sides, we can use optimized padded copy
                memcpy_warp_isspace((void *)((D_WORD *)cache + index * feature_len), 
                                    input_feature, feature_len);
            } else {
                // Otherwise just use aligned copies
                memcpy_warp((void *)((D_WORD *)cache + index * feature_len), 
                            input_feature, feature_len);
            }
            __syncwarp();
            break;
        }
        // Not success, retry with new location
        // Need a "strong" read that bypasses hardware cache
        lru = atomicAdd(&cache_lru[index], 0);
        // Reattempt to find slot(s) with minimum LRU
        min_lru = reduceMinSync(FULL_MASK, lru);
        candidates = __ballot_sync(FULL_MASK, lru == min_lru && lru < lru_counter);
        empty_slot = __ffs(candidates) - 1;
    }
    return (empty_slot >= 0 ? index : -1);
}

// Bulk insert into cache, setting appropriate LRU values
__global__ void insert_features_kernel(void **cache, int **cache_key, int **cache_lru, 
    int64_t num_nodes, int64_t feature_len, float *cpu_features, 
    int32_t *index_array, int lru_counter, int num_gpus, int64_t total_nodes, 
    int num_ways, int32_t num_sets, int *failed_inserts = nullptr)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int laneId = threadIdx.x % DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    for(int i = warpId; i < num_nodes; i += numWarps) {
        __syncwarp();
        int64_t nodeId = index_array[i];
        // Attempt an insert. Check if there's enough space for padded read on both sides
        int slot = insert_single_feature(cache, cache_key, cache_lru, nodeId, 
            feature_len, lru_counter, &cpu_features[nodeId * feature_len], num_gpus, 
            num_ways, num_sets, nodeId * feature_len >= GPU_CL_FLOATS && 
            (nodeId + 1) * feature_len + GPU_CL_FLOATS < total_nodes * feature_len);
        
        if(failed_inserts != nullptr && slot < 0 && laneId == 0) {
            atomicAdd(failed_inserts, 1);
        }
    }
}

// [TEST] Check if features properly copied into cache
__global__ void test_lookup_features_kernel(void **dev_cache, int **dev_cache_key, 
    int **dev_cache_lru, int64_t num_nodes, int64_t feature_len, void *cpu_features, 
    int32_t *index_array, int num_gpus, int num_ways, int32_t num_sets, 
    int *success_lookups = nullptr)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int laneId = threadIdx.x % DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    for(int i = warpId; i < num_nodes; i += numWarps) {
        __syncwarp();
        int nodeId = index_array[i];
        // Hash the node id to a set
        int devId = key_to_gpu_index(nodeId, num_sets, num_gpus);
        int setId = key_to_set_index(nodeId, num_sets, num_gpus);
        float *cache = (float*)dev_cache[devId];
        int *cache_keys = dev_cache_key[devId];

        // Lookup present keys in set
        int index = laneId + setId * num_ways;
        int key = cache_keys[index];
        // Attempt to find empty slot
        int mask = __ballot_sync(FULL_MASK, key == nodeId);
        int matched_lane = __ffs(mask) - 1;
        if(matched_lane >= 0) {
            index = matched_lane + setId * num_ways;
            float *dev_features  = ((float *)cache + index * feature_len);
            float *host_features = ((float *)cpu_features + nodeId * feature_len);
            int match = 1;
            for(int j = laneId; j < feature_len; j += DWARP_SIZE) {
                if(dev_features[j] != host_features[j]) {
                    match = 0;
                    break;
                }
            }
            int failed = __ballot_sync(FULL_MASK, match == 0);
            if(!failed && laneId == 0 && success_lookups != nullptr) {
                atomicAdd(success_lookups, 1);
            }
        }
    }
}