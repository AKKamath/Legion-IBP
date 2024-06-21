#include "dyn_cache.cuh"

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
__inline__ __device__ int insert_shadow_feature_lru(
    void **dev_cache_storage, int **dev_cache_keys, int **dev_cache_lru, 
    int64_t key, int64_t feature_len, int lru_counter, void *input_feature, 
    int num_gpus, int num_ways, int32_t num_sets, bool not_fringe)
{
    // Get details of set to insert into
    int devId = key_to_gpu_index(key, num_sets, num_gpus);
    int setId = key_to_set_index(key, num_sets, num_gpus);
    int *cache_keys = dev_cache_keys[devId];
    int *cache_lru = dev_cache_lru[devId];

    int laneId = threadIdx.x % DWARP_SIZE;
    int index, lru, min_lru, candidates, empty_slot;

    for(int i = laneId; i < num_ways; i += DWARP_SIZE) {
        __syncwarp();
        index = i + setId * num_ways;
        lru = cache_lru[index];
        // Attempt to find slot(s) with minimum LRU
        min_lru = reduceMinSync(FULL_MASK, lru);
        candidates = __ballot_sync(FULL_MASK, lru == min_lru && lru < lru_counter);
        empty_slot = __ffs(candidates) - 1;
        if(empty_slot >= 0)
            break;
    }
    while(empty_slot >= 0) {
        // Candidate slot found
        // Attempt to acquire exclusivity via atomic
        int acquired = 0;
        if(laneId == empty_slot) {
            // "Acquire lock" by setting the LRU value of entry
            acquired = (atomicCAS(&cache_lru[index], lru, lru_counter) == lru);
            // Set key to dummy value so no one reads from this slot anymore
            //if(acquired)
            //    atomicExch(&cache_keys[index], -1);
        }
        int success = __ballot_sync(FULL_MASK, acquired);

        if(success) {
            // Update key as a "lock-release" style operation so others can now read
            if(laneId == empty_slot) {
                atomicExch(&cache_keys[index], key);
            }
            __syncwarp();
            break;
        }
        // Not success, retry with new location
        for(int i = laneId; i < num_ways; i += DWARP_SIZE) {
            __syncwarp();
            index = i + setId * num_ways;
            // Need a "strong" read that bypasses hardware cache
            lru = atomicAdd(&cache_lru[index], 0);
            // Reattempt to find slot(s) with minimum LRU
            min_lru = reduceMinSync(FULL_MASK, lru);
            candidates = __ballot_sync(FULL_MASK, lru == min_lru && lru < lru_counter);
            empty_slot = __ffs(candidates) - 1;
            if(empty_slot >= 0)
                break;
        }
    }
    return (empty_slot >= 0 ? index : -1);
}


// Returns index of inserted feature or -1 if failed to insert
__inline__ __device__ int insert_shadow_feature_lfu(
    void **dev_cache_storage, int **dev_cache_keys, int **dev_cache_lru, 
    int64_t key, int64_t feature_len, int lru_counter, void *input_feature, 
    int num_gpus, int num_ways, int32_t num_sets, bool not_fringe)
{
    // Get details of set to insert into
    int devId = key_to_gpu_index(key, num_sets, num_gpus);
    int setId = key_to_set_index(key, num_sets, num_gpus);
    int *cache_keys = dev_cache_keys[devId];
    int *cache_lru = dev_cache_lru[devId];

    int laneId = threadIdx.x % DWARP_SIZE;
    int index, lru, min_lru, candidates, empty_slot;

    for(int i = laneId; i < num_ways; i += DWARP_SIZE) {
        __syncwarp();
        index = i + setId * num_ways;
        lru = cache_lru[index];
        // Attempt to find slot(s) with minimum LRU
        min_lru = reduceMinSync(FULL_MASK, lru);
        candidates = __ballot_sync(FULL_MASK, lru == min_lru);
        empty_slot = __ffs(candidates) - 1;
        if(empty_slot >= 0)
            break;
    }
    while(empty_slot >= 0) {
        // Candidate slot found
        // Attempt to acquire exclusivity via atomic
        int acquired = 0;
        if(laneId == empty_slot) {
            // "Acquire lock" by setting the LRU value of entry
            acquired = (atomicCAS(&cache_lru[index], lru, 0) == lru);
            // Set key to dummy value so no one reads from this slot anymore
            //if(acquired)
            //    atomicExch(&cache_keys[index], -1);
        }
        int success = __ballot_sync(FULL_MASK, acquired);

        if(success) {
            // Update key as a "lock-release" style operation so others can now read
            if(laneId == empty_slot) {
                atomicExch(&cache_keys[index], key);
            }
            __syncwarp();
            break;
        }
        // Not success, retry with new location
        for(int i = laneId; i < num_ways; i += DWARP_SIZE) {
            __syncwarp();
            index = i + setId * num_ways;
            // Need a "strong" read that bypasses hardware cache
            lru = atomicAdd(&cache_lru[index], 0);
            // Reattempt to find slot(s) with minimum LRU
            min_lru = reduceMinSync(FULL_MASK, lru);
            candidates = __ballot_sync(FULL_MASK, lru == min_lru);
            empty_slot = __ffs(candidates) - 1;
            if(empty_slot >= 0)
                break;
        }
    }
    return (empty_slot >= 0 ? index : -1);
}

// Returns index of inserted feature or -1 if failed to insert
__inline__ __device__ int insert_single_feature_lfu(
    void **dev_cache_storage, int **dev_cache_keys, int **dev_cache_lru, 
    int64_t key, int64_t feature_len, int lfu_val, void *input_feature, 
    int num_gpus, int num_ways, int32_t num_sets, bool not_fringe)
{
    // Get details of set to insert into
    int devId = key_to_gpu_index(key, num_sets, num_gpus);
    int setId = key_to_set_index(key, num_sets, num_gpus);
    float *cache = (float*)dev_cache_storage[devId];
    int *cache_keys = dev_cache_keys[devId];
    int *cache_lru = dev_cache_lru[devId];

    int laneId = threadIdx.x % DWARP_SIZE;
    int index, lfu, min_lfu, candidates, empty_slot, i;

    for(i = laneId; i < num_ways; i += DWARP_SIZE) {
        __syncwarp();
        index = i + setId * num_ways;
        lfu = cache_lru[index];
        // Attempt to find slot(s) with minimum LRU
        min_lfu = reduceMinSync(FULL_MASK, lfu);
        candidates = __ballot_sync(FULL_MASK, lfu == min_lfu && lfu < lfu_val);
        empty_slot = __ffs(candidates) - 1;
        if(empty_slot >= 0)
            break;
    }
    while(empty_slot >= 0) {
        // Candidate slot found
        // Attempt to acquire exclusivity via atomic
        int acquired = 0;
        if(laneId == empty_slot) {
            // "Acquire lock" by setting the LRU value of entry
            acquired = (atomicCAS(&cache_lru[index], lfu, lfu_val) == lfu);
            // Set key to dummy value so no one reads from this slot anymore
            //if(acquired)
            //    atomicExch(&cache_keys[index], -1);
        }
        int success = __ballot_sync(FULL_MASK, acquired);

        if(success) {
            index = empty_slot + i / DWARP_SIZE * DWARP_SIZE + setId * num_ways;
            // Update key as a "lock-release" style operation so others can now read
            if(laneId == empty_slot) {
                atomicExch(&cache_keys[index], key);
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
        for(int i = laneId; i < num_ways; i += DWARP_SIZE) {
            __syncwarp();
            index = i + setId * num_ways;
            // Need a "strong" read that bypasses hardware cache
            lfu = atomicAdd(&cache_lru[index], 0);
            // Attempt to find slot(s) with minimum LRU
            min_lfu = reduceMinSync(FULL_MASK, lfu);
            // Reattempt to find slot(s) with minimum LRU
            candidates = __ballot_sync(FULL_MASK, lfu == min_lfu && lfu < lfu_val);
            empty_slot = __ffs(candidates) - 1;
            if(empty_slot >= 0)
                break;
        }
        
    }
    return (empty_slot >= 0 ? index : -1);
}

// Returns index of inserted feature or -1 if failed to insert
__inline__ __device__ int insert_single_feature_lru(
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
    int index, lru, min_lru, candidates, empty_slot, i;

    for(i = laneId; i < num_ways; i += DWARP_SIZE) {
        __syncwarp();
        index = i + setId * num_ways;
        lru = cache_lru[index];
        // Attempt to find slot(s) with minimum LRU
        min_lru = reduceMinSync(FULL_MASK, lru);
        candidates = __ballot_sync(FULL_MASK, lru == min_lru && lru < lru_counter);
        empty_slot = __ffs(candidates) - 1;
        if(empty_slot >= 0)
            break;
    }
    while(empty_slot >= 0) {
        // Candidate slot found
        // Attempt to acquire exclusivity via atomic
        int acquired = 0;
        if(laneId == empty_slot) {
            // "Acquire lock" by setting the LRU value of entry
            acquired = (atomicCAS(&cache_lru[index], lru, lru_counter) == lru);
            // Set key to dummy value so no one reads from this slot anymore
            //if(acquired)
            //    atomicExch(&cache_keys[index], -1);
        }
        int success = __ballot_sync(FULL_MASK, acquired);

        if(success) {
            index = empty_slot + i / DWARP_SIZE * DWARP_SIZE + setId * num_ways;
            // Update key as a "lock-release" style operation so others can now read
            if(laneId == empty_slot) {
                atomicExch(&cache_keys[index], key);
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
        for(int i = laneId; i < num_ways; i += DWARP_SIZE) {
            __syncwarp();
            index = i + setId * num_ways;
            // Need a "strong" read that bypasses hardware cache
            lru = atomicAdd(&cache_lru[index], 0);
            // Attempt to find slot(s) with minimum LRU
            min_lru = reduceMinSync(FULL_MASK, lru);
            // Reattempt to find slot(s) with minimum LRU
            candidates = __ballot_sync(FULL_MASK, lru == min_lru && lru < lru_counter);
            empty_slot = __ffs(candidates) - 1;
            if(empty_slot >= 0)
                break;
        }
        
    }
    return (empty_slot >= 0 ? index : -1);
}

// Returns index of found feature or -1 if failed to insert
__inline__ __device__ int read_and_touch_shadow_feature_lru(
    void **dev_cache_storage, int **dev_cache_keys, int **dev_cache_lru, 
    int64_t input_key, int64_t feature_len,
    int lru_counter, int num_gpus, int num_ways, int32_t num_sets)
{
    int laneId = threadIdx.x % DWARP_SIZE;
    // Get details of set to insert into
    int devId = key_to_gpu_index(input_key, num_sets, num_gpus);
    int setId = key_to_set_index(input_key, num_sets, num_gpus);
    int *cache_keys = dev_cache_keys[devId];
    int *cache_lru = dev_cache_lru[devId];

    // Lookup present keys in set
    int index, key, mask, matched_lane;
    for(int i = laneId; i < num_ways; i += DWARP_SIZE) {
        __syncwarp();
        index = i + setId * num_ways;
        key = cache_keys[index];
        // Attempt to find slot with matching nodeId
        mask = __ballot_sync(FULL_MASK, key == input_key);
        matched_lane = __ffs(mask) - 1;
        if(matched_lane >= 0)
            break;
    }
    while(matched_lane >= 0) {
        // Match found
        int acquired = 1;
        if(laneId == matched_lane) {
            // Update LRU counter atomically to lock the entry and 
            // prevent concurrent inserts from updating the value
            atomicExch(&cache_lru[index], lru_counter);
            // Ensure that key wasn't updated during this process
            acquired = (atomicAdd(&cache_keys[index], 0) == key);
        }
        int success = __ballot_sync(FULL_MASK, acquired);

        if(success) {
            break;
        }
        // Not success, retry with new location
        for(int i = laneId; i < num_ways; i += DWARP_SIZE) {
            __syncwarp();
            index = i + setId * num_ways;
            // Need a "strong" read that bypasses hardware cache
            key = atomicAdd(&cache_keys[index], 0);
            // Attempt to find slot with matching nodeId
            mask = __ballot_sync(FULL_MASK, key == input_key);
            matched_lane = __ffs(mask) - 1;
            if(matched_lane >= 0)
                break;
        }
    }
    return (matched_lane >= 0 ? index : -1);
}


// Returns index of found feature or -1 if failed to insert
__inline__ __device__ int read_and_touch_shadow_feature_lfu(
    void **dev_cache_storage, int **dev_cache_keys, int **dev_cache_lru, 
    int64_t input_key, int64_t feature_len,
    int lru_counter, int num_gpus, int num_ways, int32_t num_sets)
{
    int laneId = threadIdx.x % DWARP_SIZE;
    // Get details of set to insert into
    int devId = key_to_gpu_index(input_key, num_sets, num_gpus);
    int setId = key_to_set_index(input_key, num_sets, num_gpus);
    int *cache_keys = dev_cache_keys[devId];
    int *cache_lru = dev_cache_lru[devId];

    int lfu_val = 0;
    // Lookup present keys in set
    int index, key, mask, matched_lane;
    for(int i = laneId; i < num_ways; i += DWARP_SIZE) {
        __syncwarp();
        index = i + setId * num_ways;
        key = cache_keys[index];
        // Attempt to find slot with matching nodeId
        mask = __ballot_sync(FULL_MASK, key == input_key);
        matched_lane = __ffs(mask) - 1;
        if(matched_lane >= 0)
            break;
    }
    while(matched_lane >= 0) {
        // Match found
        int acquired = 1;
        if(laneId == matched_lane) {
            // Update LRU counter atomically to lock the entry and 
            // prevent concurrent inserts from updating the value
            lfu_val = atomicAdd(&cache_lru[index], 1);
            // Ensure that key wasn't updated during this process
            acquired = (atomicAdd(&cache_keys[index], 0) == key);
        }
        int success = __ballot_sync(FULL_MASK, acquired);

        if(success) {
            lfu_val = __shfl_sync(FULL_MASK, matched_lane, lfu_val);
            break;
        }
        lfu_val = 0;
        // Not success, retry with new location
        for(int i = laneId; i < num_ways; i += DWARP_SIZE) {
            __syncwarp();
            index = i + setId * num_ways;
            // Need a "strong" read that bypasses hardware cache
            key = atomicAdd(&cache_keys[index], 0);
            // Attempt to find slot with matching nodeId
            mask = __ballot_sync(FULL_MASK, key == input_key);
            matched_lane = __ffs(mask) - 1;
            if(matched_lane >= 0)
                break;
        }
    }
    return (lfu_val > 3 ? lfu_val : -1);
}

// Returns index of found feature or -1 if failed to insert
__inline__ __device__ int read_and_touch_single_feature_lru(
    void **dev_cache_storage, int **dev_cache_keys, int **dev_cache_lru, 
    int64_t input_key, int64_t feature_len, float *output_feature, 
    int lru_counter, int num_gpus, int num_ways, int32_t num_sets)
{
    int laneId = threadIdx.x % DWARP_SIZE;
    // Get details of set to insert into
    int devId = key_to_gpu_index(input_key, num_sets, num_gpus);
    int setId = key_to_set_index(input_key, num_sets, num_gpus);
    float *cache = (float*)dev_cache_storage[devId];
    int *cache_keys = dev_cache_keys[devId];
    int *cache_lru = dev_cache_lru[devId];

    int index, key, mask, matched_lane, i;
    // Lookup present keys in set
    for(i = laneId; i < num_ways; i += DWARP_SIZE) {
        __syncwarp();
        index = i + setId * num_ways;
        key = cache_keys[index];
        // Attempt to find slot with matching nodeId
        mask = __ballot_sync(FULL_MASK, key == input_key);
        matched_lane = __ffs(mask) - 1;
        if(matched_lane >= 0)
            break;
    }
    while(matched_lane >= 0) {
        // Match found
        int acquired = 1;
        if(laneId == matched_lane) {
            // Update LRU counter atomically to lock the entry and 
            // prevent concurrent inserts from updating the value
            atomicExch(&cache_lru[index], lru_counter);
            // Ensure that key wasn't updated during this process
            acquired = (atomicAdd(&cache_keys[index], 0) == key);
        }
        int success = __ballot_sync(FULL_MASK, acquired);

        if(success) {
            index = matched_lane + i / DWARP_SIZE * DWARP_SIZE + setId * num_ways;
            // Copy the feature to output
            // Just use aligned copies, GPU copy doesn't need too much optimization
            memcpy_warp(output_feature,
                (void *)((D_WORD *)cache + index * feature_len), feature_len);
            __syncwarp();
            break;
        }
        // Not success, retry with new location
        for(int i = laneId; i < num_ways; i += DWARP_SIZE) {
            __syncwarp();
            index = i + setId * num_ways;
            // Need a "strong" read that bypasses hardware cache
            key = atomicAdd(&cache_keys[index], 0);
            // Attempt to find slot with matching nodeId
            mask = __ballot_sync(FULL_MASK, key == input_key);
            matched_lane = __ffs(mask) - 1;
            if(matched_lane >= 0)
                break;
        }
    }
    return (matched_lane >= 0 ? index : -1);
}

// Returns index of found feature or -1 if failed to insert
__inline__ __device__ int read_and_touch_single_feature_lfu(
    void **dev_cache_storage, int **dev_cache_keys, int **dev_cache_lru, 
    int64_t input_key, int64_t feature_len, float *output_feature, 
    int lfu_val, int num_gpus, int num_ways, int32_t num_sets)
{
    int laneId = threadIdx.x % DWARP_SIZE;
    // Get details of set to insert into
    int devId = key_to_gpu_index(input_key, num_sets, num_gpus);
    int setId = key_to_set_index(input_key, num_sets, num_gpus);
    float *cache = (float*)dev_cache_storage[devId];
    int *cache_keys = dev_cache_keys[devId];

    int index, key, mask, matched_lane, i;
    // Lookup present keys in set
    for(i = laneId; i < num_ways; i += DWARP_SIZE) {
        __syncwarp();
        index = i + setId * num_ways;
        key = cache_keys[index];
        // Attempt to find slot with matching nodeId
        mask = __ballot_sync(FULL_MASK, key == input_key);
        matched_lane = __ffs(mask) - 1;
        if(matched_lane >= 0)
            break;
    }
    while(matched_lane >= 0) {
        // Match found
        int acquired = 1;
        /*if(laneId == matched_lane) {
            // Update LRU counter atomically to lock the entry and 
            // prevent concurrent inserts from updating the value
            atomicAdd(&cache_lru[index], 1);
            // Ensure that key wasn't updated during this process
            acquired = (atomicAdd(&cache_keys[index], 0) == key);
        }*/
        //int success = __ballot_sync(FULL_MASK, acquired);

        if(acquired) {
            index = matched_lane + i / DWARP_SIZE * DWARP_SIZE + setId * num_ways;
            // Copy the feature to output
            // Just use aligned copies, GPU copy doesn't need too much optimization
            memcpy_warp(output_feature,
                (void *)((D_WORD *)cache + index * feature_len), feature_len);
            __syncwarp();
            break;
        }
        // Not success, retry with new location
        for(int i = laneId; i < num_ways; i += DWARP_SIZE) {
            __syncwarp();
            index = i + setId * num_ways;
            // Need a "strong" read that bypasses hardware cache
            key = atomicAdd(&cache_keys[index], 0);
            // Attempt to find slot with matching nodeId
            mask = __ballot_sync(FULL_MASK, key == input_key);
            matched_lane = __ffs(mask) - 1;
            if(matched_lane >= 0)
                break;
        }
    }
    return (matched_lane >= 0 ? index : -1);
}


// Bulk insert into cache, setting appropriate LRU values
__global__ void insert_features_kernel(void **cache, int **cache_key, int **cache_lru, 
    int64_t num_nodes, int64_t feature_len, float *cpu_features, 
    int32_t *index_array, int lru_counter, int num_gpus, int64_t total_nodes, 
    int num_ways, int32_t num_sets, int dyn_flags, int *failed_inserts = nullptr)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int laneId = threadIdx.x % DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    for(int i = warpId; i < num_nodes; i += numWarps) {
        __syncwarp();
        int64_t nodeId = index_array[i];
        // Attempt an insert. Check if there's enough space for padded read on both sides
        int slot = insert_single_feature_lfu(cache, cache_key, cache_lru, nodeId, 
            feature_len, 0, &cpu_features[nodeId * feature_len], num_gpus, 
            num_ways, num_sets, nodeId * feature_len >= GPU_CL_FLOATS && 
            (nodeId + 1) * feature_len + GPU_CL_FLOATS < total_nodes * feature_len);
        if(slot < 0) {
            if(dyn_flags & DYN_SHADOW)
                read_and_touch_shadow_feature_lfu(cache, &cache_key[num_gpus], &cache_lru[num_gpus], 
                    nodeId, feature_len, 0, num_gpus, num_ways, num_sets);
            if(failed_inserts != nullptr && laneId == 0) {
                atomicAdd(failed_inserts, 1);
            }
        }
        
    }
}

// [TEST] Check if features properly copied into cache
__global__ void test_lookup_features_kernel(void **dev_cache, int **dev_cache_key, 
    int **dev_cache_lru, int64_t num_nodes, int64_t feature_len, void *cpu_features, 
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
        float *cache = (float*)dev_cache[devId];
        int *cache_keys = dev_cache_key[devId];

        int index, key, mask, matched_lane, iter;
        // Not success, retry with new location
        for(iter = laneId; iter < num_ways; iter += DWARP_SIZE) {
            __syncwarp();
            index = iter + setId * num_ways;
            // Need a "strong" read that bypasses hardware cache
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
            if(!failed && success_lookups != nullptr && laneId == 0) {
                atomicAdd(success_lookups, 1);
            }
        }
    }
}

__global__ void retrieve_and_touch_kernel(void **dev_cache, int **dev_cache_key, 
    int **dev_cache_lru, int64_t num_nodes, int64_t feature_len, float *output_features,
    float *cpu_features, int32_t *node_arr, int lru_counter, int num_gpus, int total_nodes,
    int num_ways, int32_t num_sets, int dyn_flags, 
    ull *misses = nullptr, ull *accesses = nullptr, ull *inserts = nullptr) {
    
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    // Go through node list
    for(int i = warpId; i < num_nodes; i += numWarps) {
        __syncwarp();
        int64_t nodeId = node_arr[i];
        // Copy from cache if available
        int lfu = read_and_touch_single_feature_lfu(
            dev_cache, dev_cache_key, dev_cache_lru, 
            nodeId, feature_len, &output_features[i * feature_len], 
            0, num_gpus, num_ways, num_sets);
        // Not in cache, manual copy
        if(lfu < 0) {
            // If we have extra space on both sides, we can use optimized padded copy
            if(nodeId * feature_len >= GPU_CL_FLOATS && 
                (nodeId + 1) * feature_len + GPU_CL_FLOATS < total_nodes * feature_len) {
                // If we have extra space on both sides, we can use optimized padded copy
                memcpy_warp_isspace(&output_features[i * feature_len], 
                                    &cpu_features[nodeId * feature_len], feature_len);
            } else {
                // Otherwise just use aligned copies
                memcpy_warp(&output_features[i * feature_len], 
                            &cpu_features[nodeId * feature_len], feature_len);
            }
            __syncwarp();
            /*if(dyn_flags & DYN_SHADOW) {
                int lfu_val = read_and_touch_shadow_feature_lfu(
                    dev_cache, &dev_cache_key[num_gpus], &dev_cache_lru[num_gpus], 
                    nodeId, feature_len, 0, num_gpus, num_ways, num_sets);
                // Exists in shadow cache, try to insert into main cache
                if(lfu_val >= 0) {
                    // Try to dynamically insert
                    int inserted = insert_single_feature_lfu(dev_cache, dev_cache_key, dev_cache_lru, 
                        nodeId, feature_len, lfu_val, &output_features[i * feature_len], 
                        num_gpus, num_ways, num_sets, i * feature_len >= GPU_CL_FLOATS && 
                        (i + 1) * feature_len + GPU_CL_FLOATS < num_nodes * feature_len);
                    int devId = key_to_gpu_index(nodeId, num_sets, num_gpus);
                    if(inserted >= 0 && laneId == 0) {
                        // Free shadow entry
                        //atomicExch(&dev_cache_lru[devId + num_gpus][shadow_entry], -1);
#ifdef MONITOR_DEEP
                        if(inserts)
                            atomicAdd(inserts, 1);
#endif
                    }
                } else {
                    // Insert into shadow cache
                    insert_shadow_feature_lfu(dev_cache, &dev_cache_key[num_gpus], &dev_cache_lru[num_gpus], 
                        nodeId, feature_len, 0, &output_features[i * feature_len], 
                        num_gpus, num_ways, num_sets, i * feature_len >= GPU_CL_FLOATS && 
                        (i + 1) * feature_len + GPU_CL_FLOATS < num_nodes * feature_len);
                }
            } else {
                insert_single_feature_lfu(dev_cache, dev_cache_key, dev_cache_lru, 
                    nodeId, feature_len, 0, &output_features[i * feature_len], 
                    num_gpus, num_ways, num_sets, i * feature_len >= GPU_CL_FLOATS && 
                    (i + 1) * feature_len + GPU_CL_FLOATS < num_nodes * feature_len);
#ifdef MONITOR_DEEP
                if(inserts)
                    atomicAdd(inserts, 1);
#endif
            }*/
#ifdef MONITOR_DEEP
            if(misses != nullptr && laneId == 0)
                atomicAdd(misses, 1);
#endif
        }
#ifdef MONITOR_DEEP
        if(accesses != nullptr && laneId == 0)
            atomicAdd(accesses, 1);
#endif
    }
}

__global__ void retrieve_kernel(void **dev_cache, int **dev_cache_key, 
    int *cache_index, int64_t num_nodes, int32_t *node_arr, int num_gpus,
    int num_ways, int32_t num_sets, int dyn_flags, 
    ull *misses = nullptr, ull *accesses = nullptr, ull *inserts = nullptr) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int laneId = threadIdx.x % DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    // Go through node list
    for(int i = warpId; i < num_nodes; i += numWarps) {
        __syncwarp();
        int64_t nodeId = node_arr[i];
        int laneId = threadIdx.x % DWARP_SIZE;
        // Get details of set to insert into
        int devId = key_to_gpu_index(nodeId, num_sets, num_gpus);
        int setId = key_to_set_index(nodeId, num_sets, num_gpus);
        int *cache_keys = dev_cache_key[devId];
        cache_index[i] = -1;

        int index, key, mask, matched_lane, iter;
        // Lookup present keys in set
        for(iter = laneId; iter < num_ways; iter += DWARP_SIZE) {
            __syncwarp();
            index = iter + setId * num_ways;
            key = cache_keys[index];
            // Attempt to find slot with matching nodeId
            mask = __ballot_sync(FULL_MASK, key == nodeId);
            matched_lane = __ffs(mask) - 1;
            if(matched_lane >= 0) {
                if(laneId == matched_lane)
                    cache_index[i] = index;
                break;
            }
        }
    }
}

__global__ void transfer_kernel(void **dev_cache, int **dev_cache_key, 
    int **dev_cache_lru, int *cache_index, int64_t num_nodes, int64_t feature_len, float *output_features,
    float *cpu_features, int32_t *node_arr, int lru_counter, int num_gpus, int total_nodes,
    int num_ways, int32_t num_sets, int dyn_flags, 
    ull *misses = nullptr, ull *accesses = nullptr, ull *inserts = nullptr) {
    
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int laneId = threadIdx.x % DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    // Go through node list
    for(int i = warpId; i < num_nodes; i += numWarps) {
        __syncwarp();
        int64_t nodeId = node_arr[i];
        int index = cache_index[i];
        // Not in cache, manual copy
        if(index < 0) {
            // If we have extra space on both sides, we can use optimized padded copy
                // If we have extra space on both sides, we can use optimized padded copy
                memcpy_warp_isspace(&output_features[i * feature_len], 
                                    &cpu_features[nodeId * feature_len], feature_len);
        }
    }
}