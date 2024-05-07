#include "static_cache.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
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

__inline__ __device__ int64_t offset_to_value(int64_t offset, int gpu_id, int64_t num_sets, int64_t num_ways) 
{
    return gpu_id * num_sets * num_ways + offset;
}

__inline__ __device__ int64_t value_to_offset(int64_t value, int64_t num_sets, int64_t num_ways)
{
    return value % (num_sets * num_ways);
}

__inline__ __device__ int64_t value_to_gpu_index(int64_t value, int64_t num_sets, int64_t num_ways)
{
    return value / (num_sets * num_ways);
}

__global__ void static_reset_cache_metadata(int *cache_keys, ull *cache_val, int64_t nodes) 
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = threadId; i < nodes; i += blockDim.x * gridDim.x) {
        cache_keys[i] = -1;
        cache_val[i] = -1;
    }
}

// Returns index of inserted feature or -1 if failed to insert
__inline__ __device__ int static_insert_single_feature(
    int **dev_cache_keys, ull **dev_cache_vals, int key,
    ull val, int num_gpus, int num_ways, int32_t num_sets)
{
    // Get details of set to insert into
    int devId = key_to_gpu_index(key, num_sets, num_gpus);
    int setId = key_to_set_index(key, num_sets, num_gpus);
    int *cache_keys = dev_cache_keys[devId];
    ull *cache_vals = dev_cache_vals[devId];

    int laneId = threadIdx.x % DWARP_SIZE;
    int index, candidates, empty_slot, i;

    int cand_key = -2;
    for(i = laneId; i < max(num_ways, DWARP_SIZE); i += DWARP_SIZE) {
        index = i + setId * num_ways;
        if(i < num_ways)
            cand_key = cache_keys[index];
        candidates = __ballot_sync(FULL_MASK, cand_key == -1);
        empty_slot = __ffs(candidates) - 1;
        if(empty_slot >= 0)
            break;
    }
    while(empty_slot >= 0) {
        // Candidate slot found
        // Attempt to acquire exclusivity via atomic
        int acquired = 0;
        if(laneId == empty_slot) {
            // "Acquire lock" by setting the value of entry
            acquired = (atomicCAS(&cache_keys[index], -1, key) == -1);
        }
        int success = __ballot_sync(FULL_MASK, acquired);

        if(success) {
            // Update key as a "lock-release" style operation so others can now read
            if(laneId == empty_slot) {
                atomicExch(&cache_vals[index], val);
            }
            break;
        }
        // Not success, retry with new location
        for(int i = laneId; i < max(num_ways, DWARP_SIZE); i += DWARP_SIZE) {
            index = i + setId * num_ways;
            // Need a "strong" read that bypasses hardware cache
            if(i < num_ways)
                cand_key = cache_keys[index];
            // Reattempt to find slot(s) with minimum LRU
            candidates = __ballot_sync(FULL_MASK, cand_key == -1);
            empty_slot = __ffs(candidates) - 1;
            if(empty_slot >= 0)
                break;
        }
        
    }
    return (empty_slot >= 0 ? index : -1);
}

// Bulk insert into cache
__global__ void static_insert_features_kernel(void *cache, int **cache_key, 
    ull **cache_val, int gpu_id, int num_gpus, int64_t num_nodes, int64_t feature_len, 
    float *cpu_features, int32_t *index_array, int64_t total_nodes, 
    int num_ways, int32_t num_sets, int dyn_flags, int *failed_inserts = nullptr)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int laneId = threadIdx.x % DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    for(int i = warpId; i < num_nodes; i += numWarps) {
        __syncwarp();
        int64_t nodeId = index_array[i];
        // Otherwise just use aligned copies
        memcpy_warp((void *)((D_WORD *)cache + i * feature_len), 
                    &cpu_features[nodeId * feature_len], feature_len);
        __syncwarp();
        
        ull value = offset_to_value(i, gpu_id, num_sets, num_ways);
        // Attempt an insert. Check if there's enough space for padded read on both sides
        int slot = static_insert_single_feature(cache_key, cache_val, nodeId, value, 
            num_gpus, num_ways, num_sets);
        // Increment fail counter if appropriate
        if(slot < 0 && failed_inserts != nullptr && laneId == 0) {
            atomicAdd(failed_inserts, 1);
        }
    }
}

// [TEST] Check if features properly copied into cache
__global__ void static_test_lookup_features_kernel(void **dev_cache, int **dev_cache_key, 
    ull **dev_cache_val, int64_t num_nodes, int64_t feature_len, void *cpu_features, 
    int32_t *index_array, int num_gpus, int num_ways, int32_t num_sets, 
    int *success_lookups = nullptr, int *keys_found = nullptr)
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
        int *cache_keys = dev_cache_key[devId];
        ull *cache_vals = dev_cache_val[devId];

        int index, key = -1, mask, matched_lane, iter;
        // Not success, retry with new location
        for(iter = laneId; iter < max(num_ways, DWARP_SIZE); iter += DWARP_SIZE) {
            __syncwarp();
            index = iter + setId * num_ways;
            // Need a "strong" read that bypasses hardware cache
            if(iter < num_ways)
                key = cache_keys[index];
            // Attempt to find slot with matching nodeId
            mask = __ballot_sync(FULL_MASK, key == nodeId);
            matched_lane = __ffs(mask) - 1;
            if(matched_lane >= 0)
                break;
        }
        if(matched_lane >= 0) {
            if(keys_found != nullptr && laneId == 0) {
                atomicAdd(keys_found, 1);
            }
            index = matched_lane + iter / DWARP_SIZE * DWARP_SIZE + setId * num_ways;
            ull value = cache_vals[index];
            float *cache = (float*)dev_cache[value_to_gpu_index(value, num_sets, num_ways)];
            ull offset = value_to_offset(value, num_sets, num_ways);
            float *dev_features  = ((float *)cache + offset * feature_len);
            float *host_features = ((float *)cpu_features + nodeId * feature_len);
            int match = 1;
            for(int j = laneId; j < feature_len; j += DWARP_SIZE) {
                if(dev_features[j] != host_features[j]) {
                    match = 0;
                    break;
                }
            }
            int failed = __ballot_sync(FULL_MASK, match == 0);
            if(!failed && success_lookups != nullptr && laneId == 0) {
                atomicAdd(success_lookups, 1);
            }
        }
    }
}

__global__ void static_retrieve_kernel(int **dev_cache_key, ull **dev_cache_vals, 
    int64_t *cache_index, int64_t num_nodes, int32_t *node_arr, int num_gpus,
    int num_ways, int32_t num_sets, int dyn_flags, 
    ull *misses = nullptr, ull *accesses = nullptr, ull *inserts = nullptr) {
    // Split into tiles
    unsigned threads_per_work = min(num_ways, DWARP_SIZE);
    // Compute details of workers
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int workId = threadId / threads_per_work;
    int numWorkers = (blockDim.x * gridDim.x) / threads_per_work;
    int laneId = threadIdx.x % threads_per_work;

    unsigned warpOffset = (threadIdx.x % DWARP_SIZE) / threads_per_work;
    unsigned threadMask = ((1U << threads_per_work) - 1U) << (warpOffset * threads_per_work);

    // Go through node list
    for(int i = workId; i < num_nodes; i += numWorkers) {
        int32_t nodeId = node_arr[i];
        // Get details of set to insert into
        int devId = key_to_gpu_index(nodeId, num_sets, num_gpus);
        int setId = key_to_set_index(nodeId, num_sets, num_gpus);
        int *cache_keys = dev_cache_key[devId];
        int64_t *cache_vals = (int64_t*)dev_cache_vals[devId];
        cache_index[i] = -1;

        int index, mask, matched_lane, iter;
        // Lookup present keys in set
        for(iter = laneId; iter < num_ways; iter += threads_per_work) {
            index = iter + setId * num_ways;
            int32_t key = cache_keys[index];
            // Attempt to find slot with matching nodeId
            mask =  __ballot_sync(threadMask, key == nodeId);
            mask >>= (warpOffset * threads_per_work);
            matched_lane = __ffs(mask) - 1;
            if(matched_lane >= 0) {
                if(laneId == matched_lane) {
                    int64_t val = cache_vals[index];
                    //if(val == 0)
                    //    printf("%d: Found node %d at index %d with value %ld\n", i, nodeId, index, val);
                    cache_index[i] = val;
                }
                //__syncwarp(threadMask);
                break;
            }
        }
    }
}

__global__ void static_transfer_kernel(void **dev_cache, int **dev_cache_key, 
    ull **dev_cache_vals, int64_t *cache_index, int64_t num_nodes, int64_t feature_len, 
    float *output_features, float *cpu_features, int32_t *node_arr, int lru_counter, 
    int num_gpus, int total_nodes, int num_ways, int32_t num_sets, int dyn_flags, 
    ull *misses = nullptr, ull *accesses = nullptr, ull *inserts = nullptr) {
    
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int laneId = threadIdx.x % DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    // Go through node list
    for(int i = warpId; i < num_nodes; i += numWarps) {
        __syncwarp();
        int64_t nodeId = node_arr[i];
        int64_t index = cache_index[i];
        // Not in cache, manual copy
        if(index < 0) {
            // If we have extra space on both sides, we can use optimized padded copy
            memcpy_warp_isspace(&output_features[i * feature_len], 
                                &cpu_features[nodeId * feature_len], feature_len);
        } else {
            int devId = value_to_gpu_index(index, num_sets, num_ways);
            int offset = value_to_offset(index, num_sets, num_ways);
            memcpy_warp_isspace(&output_features[i * feature_len], &((float*)dev_cache[devId])[offset * feature_len], feature_len);
        }
    }
}