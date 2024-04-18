#include "dyn_cache.cuh"
#include "dyn_cache_kernel.cuh"

void DynamicCache::init_cache(int64_t nodes_per_gpu, int32_t feature_len, 
                              float *cpu_features, int *index_array, int Kg, 
                              int dev_start, int64_t total_nodes) {
    std::cout << "Initializing dynamic cache with " << nodes_per_gpu 
              << " (out of " << total_nodes << ") per GPU and " 
              << Kg << " GPUs\n";
    fflush(stdout);

    this->num_gpus = Kg;
    this->feature_len = feature_len;
    this->num_sets = (nodes_per_gpu / num_ways) * num_gpus;
    this->cache_capacity = num_sets * num_ways;

    auto start = TIME_NOW;

    // Allocate size of pointers
    void **host_cache_ptr = (void**)malloc(num_gpus * sizeof(void*));
    int **host_cache_lru_ptr = (int**)malloc(2 * num_gpus * sizeof(int*));
    int **host_cache_key_ptr = (int**)malloc(2 * num_gpus * sizeof(int*));

    // Allocate pointers for GPU
    cudaMalloc(&dev_cache_storage, num_gpus * sizeof(void*));
    cudaCheckError();
    cudaMalloc(&dev_cache_lru, 2 * num_gpus * sizeof(int*));
    cudaCheckError();
    cudaMalloc(&dev_cache_key, 2 * num_gpus * sizeof(int*));
    cudaCheckError();

    for(int i = 0; i < num_gpus; i++) {
        int device_id = i + dev_start;
        cudaSetDevice(device_id);

        // Allocate the actual caches
        cudaMalloc(&host_cache_ptr[i], num_sets / num_gpus * num_ways * feature_len * sizeof(float));
        cudaCheckError();
    }
    for(int i = 0; i < 2 * num_gpus; i++) {
        int device_id = (i % num_gpus) + dev_start;
        cudaSetDevice(device_id);

        cudaMalloc(&host_cache_lru_ptr[i], num_sets / num_gpus * num_ways * sizeof(int));
        cudaCheckError();
        cudaMalloc(&host_cache_key_ptr[i], num_sets / num_gpus * num_ways * sizeof(int));
        cudaCheckError();
        reset_cache_metadata<<<32, 512>>>(host_cache_key_ptr[i], host_cache_lru_ptr[i], num_sets / num_gpus * num_ways);
        cudaCheckError();
    }
    // Transfer pointers to GPU
    cudaMemcpy(dev_cache_storage, host_cache_ptr, num_gpus * sizeof(void*), cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(dev_cache_lru, host_cache_lru_ptr, 2 * num_gpus * sizeof(int*), cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(dev_cache_key, host_cache_key_ptr, 2 * num_gpus * sizeof(int*), cudaMemcpyHostToDevice);
    cudaCheckError();
    auto end = TIME_NOW;
    std::cout << "Time taken to allocate data structures:" << (float)TIME_DIFF(start, end) / 1000.0 << " ms\n";

    start = TIME_NOW;
    // Initialize variable for tracking operations
    int *dev_tracker;
    cudaMalloc(&dev_tracker, 2 * sizeof(int));
    cudaMemset(dev_tracker, 0, 2 * sizeof(int));
    cudaCheckError();

    int num_nodes = nodes_per_gpu * num_gpus;
    // Insert features into the dynamic cache
    insert_features(num_nodes, cpu_features, index_array, total_nodes, dev_tracker);
    ++lru_counter;
    cudaCheckError();
    cudaDeviceSynchronize();
    end = TIME_NOW;
    std::cout << "Time taken to insert: " << (float)TIME_DIFF(start, end) / 1000.0 << " ms\n";

    int failed_inserts;
    cudaMemcpy(&failed_inserts, dev_tracker, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Failed inserts: " << failed_inserts << " out of " << num_nodes << "\n";
    
    // Reset variable for tracking operations
    cudaMemset(dev_tracker, 0, sizeof(int));
    cudaCheckError();

    // Double-check if inserted features exist in cache
    test_lookup_features(num_nodes, cpu_features, index_array, dev_tracker, &dev_tracker[1]);
    cudaCheckError();

    int success_searches[2];
    cudaMemcpy(&success_searches, dev_tracker, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Keys found: " << success_searches[1] << ", successful searches: " 
              << success_searches[0] << " out of " << num_nodes << "\n"; 
    
    cudaFree(dev_tracker);
    free(host_cache_ptr);
    free(host_cache_lru_ptr);
    free(host_cache_key_ptr);
}

// Bulk insert into cache, setting appropriate LRU values
void DynamicCache::insert_features(int64_t num_nodes, float *input_feats, 
    int32_t *index_array, int64_t total_nodes, int *failed_inserts)
{
    insert_features_kernel<<<16, 512>>>(dev_cache_storage, dev_cache_key, 
        dev_cache_lru, num_nodes, feature_len, input_feats, index_array, 
        lru_counter, num_gpus, total_nodes, num_ways, num_sets, failed_inserts);
}

// Find features in cache
void DynamicCache::test_lookup_features(int64_t num_nodes, float *input_feats, 
    int32_t *index_array, int *success_lookups, int *keys_found)
{
    test_lookup_features_kernel<<<32, 512>>>(dev_cache_storage, dev_cache_key, 
        dev_cache_lru, num_nodes, feature_len, input_feats, index_array, 
        num_gpus, num_ways, num_sets, success_lookups, keys_found);
}

void DynamicCache::retrieve_and_touch(int32_t *nodeIds, int64_t num_nodes, 
    float *output_buffer, float *input_feats, int total_nodes, cudaStream_t stream, 
    ull *misses, ull *lookups, ull *inserts)
{
    ++lru_counter;
    retrieve_and_touch_kernel<<<32, 1024, 0, stream>>>(
        dev_cache_storage, dev_cache_key, dev_cache_lru, num_nodes, feature_len, 
        output_buffer, input_feats, nodeIds, lru_counter, num_gpus, total_nodes, 
        num_ways, num_sets, misses, lookups, inserts);
}