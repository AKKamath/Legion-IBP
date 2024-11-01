#define IBP_DEBUG_PRINT
#include "static_cache.cuh"
#include "compress_cache_kernel.cuh"
#include <bit>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <thrust/sequence.h>
#include "misc/compress_test.cuh"
#include "misc/ibp_misc_kernels.cuh"
#include "preproc/ibp_preproc_host.cuh"
#include "compress/ibp_compress_host.cuh"
#include "decompress/ibp_decompress_host.cuh"

bool ibp_print_debug = true;
using namespace nvcomp;

void StaticCache::init_cache(int64_t nodes_per_gpu, int32_t feature_len, 
                              float *cpu_features, int *index_array, int Kg, 
                              int dev_start, int64_t total_nodes, DYN_FLAGS flags, int ways) {
    std::cout << "Initializing dynamic cache with " << nodes_per_gpu 
              << " (out of " << total_nodes << ") per GPU and " 
              << Kg << " GPUs\n";
    fflush(stdout);
    cudaSetDevice(dev_start);

    this->num_gpus = Kg;
    this->feature_len = feature_len;
    this->num_ways = ways;
    this->num_sets = ((nodes_per_gpu * 2 + num_ways - 1) / num_ways) * num_gpus;
    this->cache_capacity = num_sets * num_ways;
    this->flags = flags;
    this->dev_start = dev_start;

    std::cout << "Precompression: Number of sets: " << num_sets << ", cache capacity: " << cache_capacity << "\n";

    //-------------- COMPRESSION CODE ----------
    if(flags & (DYN_COMP | DYN_COMP_CPU | DYN_COMP_TEST)) {
        chunk_size = 4;
        comp_mask = nullptr;
        comp_bitval = nullptr;
        compress_len = ibp::preproc_data((int32_t*)cpu_features, total_nodes, 
            feature_len, (int32_t**)&comp_mask, (int32_t**)&comp_bitval);
        printf("Finished compression preprocessing\n");
        fflush(stdout);

        // Need to decide a reasonable cache size
        // Cache size = feature capacity + hashmap capacity
        // Feature capacity = feature_len * num_nodes
        // Hashmap capacity = (key_size [32 bits] + value size [64 bits]) * num_nodes * 2
        // Cache size = num_nodes * (feature_len + key_size * 2 + value_size * 2)
        // Cache size = num_nodes * (feature_len + 6)
        size_t cache_per_gpu = nodes_per_gpu * (feature_len + 6);
        size_t comp_nodes_per_gpu = cache_per_gpu / (compress_len + 6);
        this->num_sets = ((comp_nodes_per_gpu * 2 + num_ways - 1) / num_ways) * num_gpus;
        this->cache_capacity = num_sets * num_ways;
        // Cache size = feature capacity + hashmap capacity
        // Cache size - hashmap capacity = feature_len * num_nodes
        // new_num_nodes = (cache size - hashmap capacity) / feature_len
        nodes_per_gpu = (cache_per_gpu * num_gpus - this->cache_capacity * 3) / (feature_len * num_gpus);
        std::cout << "Postcompression: Required " << cache_per_gpu << ", num sets: " << num_sets << ", cache capacity: " << cache_capacity;
        std::cout << ", new nodes_per_gpu: " << nodes_per_gpu << "\n";
    }

    // TODO: K-Means based compression
    if(flags & (DYN_COMP | DYN_COMP_CPU | DYN_COMP_TEST)) {
        //ibp::preproc_kmeans((int32_t*)cpu_features, total_nodes, feature_len, 
        //    (int32_t**)&comp_mask, (int32_t**)&comp_bitval, 0.5, chunk_size);
    }
    
    //------------------------------
    auto start = TIME_NOW;
    // Allocate size of pointers
    host_cache_storage = (void**)malloc(num_gpus * sizeof(void*));
    ull **host_cache_offset_ptr = (ull**)malloc(num_gpus * sizeof(ull*));
    int **host_cache_key_ptr = (int**)malloc(num_gpus * sizeof(int*));

    // Allocate pointers for GPU
    cudaMalloc(&dev_cache_storage, num_gpus * sizeof(void*));
    cudaCheckError();
    cudaMalloc(&dev_cache_offset, num_gpus * sizeof(ull*));
    cudaCheckError();
    cudaMalloc(&dev_cache_key, num_gpus * sizeof(int*));
    cudaCheckError();

    maxShmem = (int*)malloc(sizeof(int) * num_gpus);
    for(int i = 0; i < num_gpus; i++) {
        int device_id = i + dev_start;
        cudaSetDevice(device_id);
        cudaDeviceGetAttribute(&maxShmem[i], cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);

        // Allocate the actual cache
        cudaMalloc(&host_cache_storage[i], nodes_per_gpu * feature_len * sizeof(float));
        cudaMemset(host_cache_storage[i], 0, nodes_per_gpu * feature_len * sizeof(float));
        cudaCheckError();
        cudaMalloc(&host_cache_offset_ptr[i], num_sets / num_gpus * num_ways * sizeof(ull));
        cudaCheckError();
        cudaMalloc(&host_cache_key_ptr[i], num_sets / num_gpus * num_ways * sizeof(int));
        cudaCheckError();
        static_reset_cache_metadata<<<32, 512>>>(host_cache_key_ptr[i], host_cache_offset_ptr[i], num_sets / num_gpus * num_ways);
        cudaCheckError();
    }
    // Transfer pointers to GPU
    cudaMemcpy(dev_cache_storage, host_cache_storage, num_gpus * sizeof(void*), cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(dev_cache_offset, host_cache_offset_ptr, num_gpus * sizeof(ull*), cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(dev_cache_key, host_cache_key_ptr, num_gpus * sizeof(int*), cudaMemcpyHostToDevice);
    cudaCheckError();
    auto end = TIME_NOW;
    std::cout << "Time taken to allocate data structures:" << (float)TIME_DIFF(start, end) / 1000.0 << " ms\n";

    if(flags & DYN_COMP_TEST) {
        nodes_per_gpu = min(total_nodes, (int64_t)200000);
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        long bytes_per_feat = feature_len * sizeof(float);
        long in_bytes = bytes_per_feat * nodes_per_gpu;

        // compute chunk sizes
        size_t* host_uncompressed_bytes;
        const size_t chunk_size = feature_len * sizeof(float);
        const size_t batch_size = nodes_per_gpu;

        char* device_input_data, *device_output_data;
        cudaMalloc(&device_input_data, in_bytes);
        cudaMemcpyAsync(device_input_data, cpu_features, in_bytes, cudaMemcpyHostToDevice, stream);
        cudaMalloc(&device_output_data, in_bytes);
        cudaCheckError();

        cudaMallocHost(&host_uncompressed_bytes, sizeof(size_t)*batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            host_uncompressed_bytes[i] = bytes_per_feat;
        }

        // Setup an array of pointers to the start of each chunk
        void ** host_uncompressed_ptrs;
        cudaMallocHost(&host_uncompressed_ptrs, sizeof(size_t)*batch_size);
        for (size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
            host_uncompressed_ptrs[ix_chunk] = device_input_data + bytes_per_feat;
        }

        size_t* device_uncompressed_bytes;
        void ** device_uncompressed_ptrs;
        cudaMalloc(&device_uncompressed_bytes, sizeof(size_t) * batch_size);
        cudaMalloc(&device_uncompressed_ptrs, sizeof(size_t) * batch_size);
        cudaCheckError();
        
        cudaMemcpyAsync(device_uncompressed_bytes, host_uncompressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_uncompressed_ptrs, host_uncompressed_ptrs, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);

        // Overallocate output bytes
        void **host_compressed_ptrs;
        char *compressed_buffer;
        cudaMallocHost(&host_compressed_ptrs, sizeof(size_t) * batch_size);
        // HOST ALLOC
        if(1)
            cudaMallocHost(&compressed_buffer, 
                batch_size * 32 * feature_len * sizeof(float));
        // DEVICE ALLOC
        else
            cudaMalloc(&compressed_buffer, 
                batch_size * 32 * feature_len * sizeof(float));

        for(size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
            host_compressed_ptrs[ix_chunk] = &compressed_buffer[
                ix_chunk * 32 * feature_len * sizeof(float)];
        }
        cudaCheckError();

        size_t *device_compressed_bytes;
        cudaMalloc(&device_compressed_bytes, sizeof(size_t) * batch_size);
        cudaCheckError();
        size_t *host_compressed_bytes;
        cudaMallocHost(&host_compressed_bytes, sizeof(size_t) * batch_size);
        cudaCheckError();

        // COMPRESSION ALGOS:
        // ANS, Bitcomp, Cascaded, LZ4, Snappy, GDeflate, Zstd, Deflate
        RUN_COMPRESSION(ANS);
        cudaCheckError();
        RUN_COMPRESSION(Bitcomp);
        cudaCheckError();
        RUN_COMPRESSION(Cascaded);
        cudaCheckError();
        RUN_COMPRESSION(Gdeflate);
        cudaCheckError();
        RUN_COMPRESSION(LZ4);
        cudaCheckError();
        RUN_COMPRESSION(Snappy);
        cudaCheckError();
        RUN_COMPRESSION(Zstd);
        cudaCheckError();

        float *host_output_data;
        cudaMallocHost(&host_output_data, in_bytes);

        // Init for NDZIP
        ndzip::extent ext(1);
        ext[0] = nodes_per_gpu * feature_len;
        ndzip::compressor_requirements req(ext);

        // NDZip compressor/decompressor
        std::unique_ptr<ndzip::cuda_compressor<float>> ndzip_comp = ndzip::make_cuda_compressor<float>(req, stream);
        std::unique_ptr<ndzip::cuda_decompressor<float>> ndzip_decomp = ndzip::make_cuda_decompressor<float>(1, stream);

        ndzip_comp->compress((float*)device_input_data, ext, (uint32_t*)compressed_buffer,
            (uint32_t*)device_compressed_bytes);
        cudaDeviceSynchronize();
        cudaCheckError();
        cudaMemcpy(host_compressed_bytes, device_compressed_bytes, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        *host_compressed_bytes *= sizeof(uint32_t);
        cudaCheckError();
        printf("%s: Uncompressed bytes: %ld, compressed bytes: %u, ratio: %f\n",
            "ndzip", in_bytes, *(uint32_t*)host_compressed_bytes, (float)in_bytes / *host_compressed_bytes);

        cudaMemset(device_output_data, 0, in_bytes);
        // NDZip decompression
        auto decomp_start = TIME_NOW;
        ndzip_decomp->decompress((uint32_t*)compressed_buffer, (float*)device_output_data, ext);
        cudaDeviceSynchronize();
        cudaCheckError();
        auto decomp_end = TIME_NOW;
        printf("%s: Time taken to decompress: %f ms. Throughput: %f MB/s; True thput: %f MB/s\n", "ndzip",
            (float)TIME_DIFF(decomp_start, decomp_end) / 1000.0,
            (float)in_bytes / TIME_DIFF(decomp_start, decomp_end),
            (float)*(uint32_t*)host_compressed_bytes / TIME_DIFF(decomp_start, decomp_end));
        cudaMemcpy(host_output_data, device_output_data, in_bytes, cudaMemcpyDeviceToHost);
        cudaCheckError();
        for(size_t i = 0; i < in_bytes / sizeof(float); ++i) {
            if(((float*)cpu_features)[i] != ((float*)host_output_data)[i]) {
                printf("Mismatch at %lu: %x vs %x\n", i, ((int32_t*)cpu_features)[i], ((int32_t*)host_output_data)[i]);
                break;
            }
        }

        // Get SM info
        int device, sm_count;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

        cudaMalloc(&comp_bitmask, (nodes_per_gpu + 31) / 32 * sizeof(int32_t));
        cudaMemset(comp_bitmask, 0, (nodes_per_gpu + 31) / 32 * sizeof(int32_t));
        cudaCheckError();
        // Our scheme
        cudaMemset(compressed_buffer, 0, in_bytes);
        ull *d_comp_size, comp_size = 0;
        cudaMalloc(&d_comp_size, sizeof(ull));
        cudaMemset(d_comp_size, 0, sizeof(ull));
        ibp::compress_inplace((int32_t*)compressed_buffer, 
            (int32_t*)cpu_features, nodes_per_gpu, (int64_t)feature_len, 
            comp_mask, comp_bitval, comp_bitmask, (void*)nullptr, (void*)nullptr, 
            d_comp_size, stream);
        cudaMemcpy(&comp_size, d_comp_size, sizeof(ull), cudaMemcpyDeviceToHost);
        printf("%s: Uncompressed bytes: %ld, compressed bytes: %llu, ratio: %f\n",
            "Us", in_bytes, comp_size, (float)in_bytes / comp_size);
        cudaCheckError();
        
        auto kernel = &test_decompressed_features_kernel<true>;
        int shmem_size;
        if(feature_len * sizeof(float) * 2 < maxShmem[0]){
            shmem_size = feature_len * sizeof(float) * 2;
            kernel = &test_decompressed_features_kernel<true>;
        }
        else {
            shmem_size = 0;
            kernel = &test_decompressed_features_kernel<false>;
        }
        cudaMemset(device_output_data, 0, in_bytes);
        // Need opt-in for large shmem allocations
        if (shmem_size >= 48 * 1024) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
            cudaCheckError();
        }
        decomp_start = TIME_NOW;
        kernel<<<sm_count, 256, shmem_size>>>(
            comp_mask, comp_bitval, (int32_t*)compressed_buffer, (int32_t*)device_output_data, 
            nodes_per_gpu, feature_len, comp_bitmask, 4);
        cudaDeviceSynchronize();
        cudaCheckError();
        decomp_end = TIME_NOW;
        cudaMemcpy(host_output_data, device_output_data, in_bytes, cudaMemcpyDeviceToHost);

        printf("%s: Time taken to decompress: %f ms. Throughput: %f MB/s; True thput: %f MB/s\n", "Us",
            (float)TIME_DIFF(decomp_start, decomp_end) / 1000.0,
            (float)in_bytes / TIME_DIFF(decomp_start, decomp_end),
            (float)comp_size / TIME_DIFF(decomp_start, decomp_end));
        cudaCheckError();
        for(int i = 0; i < in_bytes / sizeof(float); ++i) {
            if(((float*)cpu_features)[i] != ((float*)host_output_data)[i]) {
                printf("Mismatch at %d: %x vs %x\n", i, ((int32_t*)cpu_features)[i], ((int32_t*)host_output_data)[i]);
                break;
            }
        }
        
        cudaMemset(device_output_data, 0, in_bytes);
        decomp_start = TIME_NOW;
        ibp::decompress_fetch<int32_t>((int32_t*)device_output_data, (int32_t*)compressed_buffer, 
            nodes_per_gpu, (int64_t)feature_len, comp_mask, comp_bitval, comp_bitmask,
            (int)((float)comp_size / in_bytes) * feature_len, (void*)nullptr, stream, sm_count, 512);
        cudaDeviceSynchronize();
        cudaCheckError();
        decomp_end = TIME_NOW;
        cudaMemcpy(host_output_data, device_output_data, in_bytes, cudaMemcpyDeviceToHost);

        printf("%s: Time taken to decompress: %f ms. Throughput: %f MB/s; True thput: %f MB/s\n", "Us2",
            (float)TIME_DIFF(decomp_start, decomp_end) / 1000.0,
            (float)in_bytes / TIME_DIFF(decomp_start, decomp_end),
            (float)comp_size / TIME_DIFF(decomp_start, decomp_end));
        cudaCheckError();
        for(int i = 0; i < in_bytes / sizeof(float); ++i) {
            if(((float*)cpu_features)[i] != ((float*)host_output_data)[i]) {
                printf("Mismatch at %d: %x vs %x\n", i, ((int32_t*)cpu_features)[i], ((int32_t*)host_output_data)[i]);
                break;
            }
        }

        // Flat data copy
        cudaMemset(comp_bitmask, 0, (nodes_per_gpu + 31) / 32 * sizeof(int32_t));
        decomp_start = TIME_NOW;
        decompressed_cpu_features_kernel<<<sm_count, 512>>>(
            comp_mask, comp_bitval, (int32_t*)compressed_buffer, (int32_t*)device_output_data, 
            nodes_per_gpu, feature_len, comp_bitmask);
        cudaDeviceSynchronize();
        decomp_end = TIME_NOW;
        printf("%s: Time taken to decompress: %f ms. Throughput: %f MB/s\n", "Transfer",
            (float)TIME_DIFF(decomp_start, decomp_end) / 1000.0,
            (float)in_bytes / TIME_DIFF(decomp_start, decomp_end));
        cudaCheckError();

        cudaDeviceSynchronize();
        fflush(stdout);
        abort();
    }

    start = TIME_NOW;
    // Initialize variable for tracking operations
    int *dev_tracker;
    cudaMalloc(&dev_tracker, 2 * sizeof(int));
    cudaMemset(dev_tracker, 0, 2 * sizeof(int));
    cudaCheckError();
    cudaDeviceSynchronize();
    if(!(flags & DYN_COMP)) {
        for(int i = 0; i < num_gpus; i++) {
            int device_id = i + dev_start;
            cudaSetDevice(device_id);
            // Insert features into the cache
            insert_features(host_cache_storage[i], nodes_per_gpu, i, cpu_features, 
                &index_array[nodes_per_gpu * i], total_nodes, dev_tracker);
        }
    } else {
        insert_features_compressed(nodes_per_gpu, cpu_features, index_array, 
            total_nodes, dev_start, num_gpus, dev_tracker);
    }
    cudaDeviceSynchronize();
    cudaCheckError();
    end = TIME_NOW;
    std::cout << "Time taken to insert: " << (float)TIME_DIFF(start, end) / 1000.0 << " ms\n";

    int num_nodes = nodes_per_gpu * num_gpus;
    int failed_inserts;
    cudaMemcpy(&failed_inserts, dev_tracker, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError();
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
    free(host_cache_offset_ptr);
    free(host_cache_key_ptr);
    if(flags & DYN_COMP_CPU) {
        cudaMalloc(&comp_bitmask, (total_nodes + 31) / 32 * sizeof(int32_t));
        cudaCheckError();
        ibp::compress_inplace((int32_t*)cpu_features, 
            (int32_t*)cpu_features, total_nodes, feature_len, 
            comp_mask, comp_bitval, comp_bitmask);
        cudaCheckError();
        /*
        // Test if compressed CPU features match when decompressed.
        int32_t *decomp;
        cudaMallocHost(&decomp, total_nodes * feature_len * sizeof(int32_t));
        decompressed_cpu_features_kernel<<<32, 512>>>(comp_mask, comp_bitval, (int32_t*)cpu_features, 
            decomp, total_nodes, feature_len, working_space, comp_bitmask);
        cudaDeviceSynchronize();
        int match = true;
        int i = 0;
        for(int j = 0; j < total_nodes; ++j) {
            for(i = 0; i < feature_len; ++i) {
                if(((int32_t *)decomp_cpu_vals)[j * feature_len + i] != decomp[j * feature_len + i]) {
                    printf("Mishmatch at %d, %d. Expected %x, got %x\n", j, i, 
                        *(unsigned*)&decomp_cpu_vals[j * feature_len + i], 
                        *(unsigned*)&decomp[j * feature_len + i]);
                    match = false;
                    break;
                }
            }
            if(!match)
                break;
        }
        printf("Match? %d\n", match);
        fflush(stdout);
        abort();
        */
        cudaDeviceSynchronize();
        cudaCheckError();
    }
}

// Bulk insert into cache, setting appropriate values
void StaticCache::insert_features(void *cache, int64_t num_nodes, int gpu, float *input_feats, 
    int32_t *index_array, int64_t total_nodes, int *failed_inserts)
{
    static_insert_features_kernel<<<16, 512>>>(cache, dev_cache_key, 
        dev_cache_offset, gpu, num_gpus, num_nodes, feature_len, input_feats, index_array, 
        total_nodes, num_ways, num_sets, flags, failed_inserts);
}

// Bulk compress and insert into cache, setting appropriate values
void StaticCache::insert_features_compressed(int64_t &nodes_per_gpu, float *input_feats, 
    int32_t *index_array, int64_t total_nodes, int dev_start, int num_gpus, int *failed_inserts)
{
    int64_t size_per_gpu = nodes_per_gpu * sizeof(float) * feature_len;
    int64_t *inserted_per_gpu = (int64_t*)malloc(sizeof(int64_t) * num_gpus);
    for(int gpu = 0; gpu < num_gpus; ++gpu) {
        inserted_per_gpu[gpu] = 0;
    }
    int inserted_feats = 0;
    int64_t *inserted_size;
    cudaMallocHost(&inserted_size, sizeof(int64_t));
    int64_t *comp_size;
    cudaMalloc(&comp_size, nodes_per_gpu * sizeof(int64_t));
    printf("Initialized data structures for compressed insert\n");
    fflush(stdout);
    // First insert features specific to each GPU
    for(int i = 0; i < num_gpus; i++) {
        int device_id = i + dev_start;
        cudaSetDevice(device_id);
        cudaMemset(comp_size, 0, nodes_per_gpu * sizeof(int64_t));
        // Check space taken by compressed data
        ibp::check_compress_size_kernel<<<32, 512>>>((int32_t *)input_feats, 
            nodes_per_gpu, feature_len, (int32_t *)comp_mask, (int32_t *)comp_bitval, 
            comp_size, &index_array[inserted_feats]);
        cudaCheckError();
        thrust::inclusive_scan(thrust::device, comp_size, comp_size + nodes_per_gpu, comp_size);
        fprintf(stderr, "Cache %p, size %ld (End: %p)\n", host_cache_storage[i], size_per_gpu, 
            (void*)((char*)host_cache_storage[i] + size_per_gpu));
        // Compress and insert data
        compressed_insert_features_kernel<<<16, 256>>>(host_cache_storage[i], 
            dev_cache_key, dev_cache_offset, comp_mask, comp_bitval, comp_size, inserted_per_gpu[i],
            i, num_gpus, nodes_per_gpu, feature_len, input_feats, 
            &index_array[inserted_feats], total_nodes, num_ways, num_sets, 
            flags, chunk_size, failed_inserts);
        cudaDeviceSynchronize();
        cudaCheckError();
        // Get details on how much data inserted
        cudaMemcpy(inserted_size, &comp_size[nodes_per_gpu - 1], sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaCheckError();
        inserted_feats += nodes_per_gpu;
        inserted_per_gpu[i] += *inserted_size;
    }
    printf("Inserted %d compressed feats so far (expected at least %lu)\n", 
        inserted_feats, nodes_per_gpu * num_gpus);
    fflush(stdout);
    // Keep inserting features while we have space
    for(int i = 0; i < num_gpus; i++) {
        int device_id = i + dev_start;
        cudaSetDevice(device_id);
        int64_t feats_to_insert = 0;
        while(inserted_per_gpu[i] + feature_len * sizeof(float) <= size_per_gpu && 
            inserted_feats < total_nodes) {
            feats_to_insert = (size_per_gpu - inserted_per_gpu[i]) / (feature_len * sizeof(float));
            feats_to_insert = min(feats_to_insert, total_nodes - inserted_feats);
            cudaMemset(comp_size, 0, feats_to_insert * sizeof(int64_t));
            // Check space taken by compressed data
            ibp::check_compress_size_kernel<<<32, 512>>>((int32_t *)input_feats, 
                feats_to_insert, feature_len, (int32_t *)comp_mask, 
                (int32_t *)comp_bitval, comp_size, &index_array[inserted_feats]);
            cudaCheckError();
            thrust::inclusive_scan(thrust::device, comp_size, comp_size + feats_to_insert, comp_size);
            // Compress and insert the data
            compressed_insert_features_kernel<<<16, 256>>>(host_cache_storage[i], 
                dev_cache_key, dev_cache_offset, comp_mask, comp_bitval, comp_size, inserted_per_gpu[i],
                i, num_gpus, feats_to_insert, feature_len, input_feats, &index_array[inserted_feats], total_nodes, 
                num_ways, num_sets, flags, chunk_size, failed_inserts);
            cudaMemcpy(inserted_size, &comp_size[feats_to_insert - 1], 
                sizeof(int64_t), cudaMemcpyDeviceToHost);
            cudaCheckError();
            printf("Inserted %ld compressed feat on gpu %d\n", feats_to_insert, i);
            fflush(stdout);
            inserted_feats += feats_to_insert;
            inserted_per_gpu[i] += *inserted_size;
        }
    }
    cudaFree(comp_size);
    cudaFreeHost(inserted_size);
    printf("Inserted %d compressed feats (expected at least %lu)\n", 
        inserted_feats, nodes_per_gpu * num_gpus);
    nodes_per_gpu = inserted_feats / num_gpus;
    /*
    cudaDeviceSynchronize();
    // Try looking up inserted features
    float *output_feats, *h_output_feats;
    int64_t *node_index;
    int *h_index_arr;
    cudaMalloc(&output_feats, 2*inserted_feats * feature_len * sizeof(float));
    cudaMallocHost(&h_output_feats, 2*inserted_feats * feature_len * sizeof(float));
    cudaMalloc(&node_index, 2*inserted_feats * sizeof(int64_t));
    cudaCheckError();
    retrieve(index_array, 2*inserted_feats, node_index, 0);
    transfer(index_array, 2*inserted_feats, output_feats, node_index, input_feats, total_nodes, 0);
    cudaCheckError();
    cudaMemcpy(h_output_feats, output_feats, 2*inserted_feats * feature_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMallocHost(&h_index_arr, 2*inserted_feats * sizeof(int));
    cudaMemcpy(h_index_arr, index_array, 2*inserted_feats * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError();
    int match = true;
    int i = 0;
    for(int j = 0; j < 2*inserted_feats; ++j)
        for(i = 0; i < feature_len; ++i) {
            if(input_feats[h_index_arr[j] * feature_len + i] != h_output_feats[j * feature_len + i]) {
                printf("Mishmatch at %d (%d). Expected %x, got %x\n", i, 
                    h_index_arr[j], *(unsigned*)&input_feats[h_index_arr[j] * feature_len + i], *(unsigned*)&h_output_feats[j * feature_len + i]);
                match = false;
                break;
            }
        }
    printf("Match? %d\n", match);
    cudaFree(output_feats);
    cudaFreeHost(h_output_feats);
    cudaFreeHost(h_index_arr);*/
    cudaCheckError();
}

// Find features in cache
void StaticCache::test_lookup_features(int64_t num_nodes, float *input_feats, 
    int32_t *index_array, int *success_lookups, int *keys_found)
{
    if(!(flags & DYN_COMP))
        static_test_lookup_features_kernel<<<32, 512>>>(dev_cache_storage, dev_cache_key, 
            dev_cache_offset, num_nodes, feature_len, input_feats, index_array, 
            num_gpus, num_ways, num_sets, success_lookups, keys_found);
    else {
        auto kernel = &compressed_test_lookup_features_kernel<true>;
        int shmem_size;
        // TODO: Do this by the executing GPU. Relevant for heterogeneous GPU machines
        if(maxShmem[0] >= 2 * feature_len * sizeof(int32_t) + sizeof(int32_t) * 512) {
            shmem_size = 2 * feature_len * sizeof(int32_t) + sizeof(int32_t) * 512;
            kernel = &compressed_test_lookup_features_kernel<true>;
        }
        else {
            shmem_size = sizeof(int32_t) * 512;
            kernel = &compressed_test_lookup_features_kernel<false>;
        }
        // Need opt-in for large shmem allocations
        if (shmem_size >= 48 * 1024) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
            cudaCheckError();
        }
        kernel<<<32, 512, shmem_size>>>
            (dev_cache_storage, dev_cache_key, dev_cache_offset, comp_mask, comp_bitval, 
            num_nodes, feature_len, input_feats, index_array, num_gpus, num_ways, num_sets, 
            success_lookups, keys_found, 4);
    }
}

void StaticCache::retrieve(int32_t *nodeIds, int64_t num_nodes, int64_t *node_index, 
    cudaStream_t stream, ull *misses, ull *lookups, ull *inserts)
{
    static_retrieve_kernel<<<32, 512, 0, stream>>>(
        dev_cache_key, dev_cache_offset, node_index, num_nodes, 
        nodeIds, num_gpus, num_ways, num_sets, flags, misses, lookups, inserts);
}

void StaticCache::transfer(int32_t *nodeIds, int64_t num_nodes, float *output_buffer, 
    int64_t *node_index, float *input_feats, int total_nodes, cudaStream_t stream, 
    ull *misses, ull *lookups, ull *inserts)
{
    // Tested for V100, A100. Adjust as needed for your GPU
    int major_version = 7;
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute (&major_version, cudaDevAttrComputeCapabilityMajor, device);

    const int NTHREADS = 512;
    const int NBLOCKS = major_version == 8 ? 64 : 32;
    int shmem_size;
    if(flags & DYN_CPU_TEST2) {
        constexpr int SHM_META = 128;
        constexpr int SHM_WORK = 256;
        auto kernel = &compress_cpu_transfer_kernel2<true, SHM_META, SHM_WORK>;
        // TODO: Change maxShmem based on executing GPU. Relevant for heterogeneous GPU machines
        if(maxShmem[0] >= 2 * feature_len * sizeof(int32_t) + NTHREADS / DWARP_SIZE * 96 * sizeof(int32_t)) {
            shmem_size = 2 * feature_len * sizeof(int32_t) + NTHREADS / DWARP_SIZE * 96 * sizeof(int32_t);
            kernel = &compress_cpu_transfer_kernel2<true, SHM_META, SHM_WORK>;
        }
        else {
            shmem_size = NTHREADS / DWARP_SIZE * 96 * sizeof(int32_t);
            kernel = &compress_cpu_transfer_kernel2<false, SHM_META, SHM_WORK>;
        }
        // Need opt-in for large shmem allocations
        if (shmem_size >= 48 * 1024) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
            cudaCheckError();
        }
        
        kernel<<<NBLOCKS, NTHREADS, shmem_size, stream>>>(dev_cache_storage, 
            node_index, num_nodes, feature_len, compress_len, output_buffer, comp_bitmask, input_feats, 
            nodeIds, comp_mask, comp_bitval, total_nodes, num_ways, num_sets, shmem_size,
            misses, lookups, inserts);
    }
    else {
        if(flags & DYN_COMP_CPU) {
            auto kernel = &compress_cpu_transfer_kernel<true>;
            // TODO: Change maxShmem based on executing GPU. Relevant for heterogeneous GPU machines
            if(maxShmem[0] >= 2 * feature_len * sizeof(int32_t)) {
                shmem_size = 2 * feature_len * sizeof(int32_t);
                kernel = &compress_cpu_transfer_kernel<true>;
            }
            else {
                shmem_size = 0;
                kernel = &compress_cpu_transfer_kernel<false>;
            }
            
            // Need opt-in for large shmem allocations
            if (shmem_size >= 48 * 1024) {
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
                cudaCheckError();
            }
            kernel<<<NBLOCKS, NTHREADS, shmem_size, stream>>>(dev_cache_storage, 
                node_index, num_nodes, feature_len, output_buffer, comp_bitmask, input_feats, 
                nodeIds, comp_mask, comp_bitval, total_nodes, num_ways, num_sets,
                misses, lookups, inserts);
        }
        else if(flags & DYN_COMP) {
            auto kernel = &compress_transfer_kernel<true>;

            // TODO: Change maxShmem based on executing GPU. Relevant for heterogeneous GPU machines
            if(maxShmem[0] >= 2 * feature_len * sizeof(int32_t)) {
                shmem_size = 2 * feature_len * sizeof(int32_t);
                kernel = &compress_transfer_kernel<true>;
            }
            else {
                shmem_size = 0;
                kernel = &compress_transfer_kernel<false>;
            }
            
            // Need opt-in for large shmem allocations
            if (shmem_size >= 48 * 1024) {
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
                cudaCheckError();
            }
            kernel<<<NBLOCKS, NTHREADS, shmem_size, stream>>>(dev_cache_storage, 
                node_index, num_nodes, feature_len, output_buffer, 
                input_feats, nodeIds, comp_mask, comp_bitval, total_nodes, num_ways, num_sets,
                misses, lookups, inserts);
        }
        else
            static_transfer_kernel<<<NBLOCKS, NTHREADS, 0, stream>>>(
                dev_cache_storage, dev_cache_key, dev_cache_offset, node_index, num_nodes, 
                feature_len, output_buffer, input_feats, nodeIds, 0, num_gpus, total_nodes, 
                num_ways, num_sets, flags, misses, lookups, inserts);
    }
    
}