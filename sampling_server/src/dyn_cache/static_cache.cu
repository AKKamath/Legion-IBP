#include "static_cache.cuh"
#include "compress_cache_kernel.cuh"
#include <bit>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <thrust/sequence.h>
#include "compress_test.cuh"
using namespace nvcomp;

// Potential optimization: tiled count
__global__ void count_bit_kernel(int32_t *feature_arr, int *index_arr, int num_nodes, unsigned feature_len, int *bit_count) {
    for(int i = blockIdx.x; i < num_nodes; i += gridDim.x) {
        int64_t nodeId = index_arr[i];
        for(unsigned j = threadIdx.x; j < feature_len; j += blockDim.x) {
            int32_t val = feature_arr[nodeId * feature_len + j];
            for(int bit = 0; bit < 32; ++bit) {
                if(val & (1 << bit))
                    atomicAdd(&bit_count[j * 32 + bit], 1);
            }
        }
    }
}

__global__ void create_mask(int32_t *bit_count, int *mask, int *vals, int feature_len, float num_nodes, float threshold) {
    for(int i = threadIdx.x; i < feature_len; i += blockDim.x) {
        int32_t val = 0;
        int32_t masker = 0;
        for(int j = 0; j < 32; ++j) {
            if(bit_count[i * 32 + j] > threshold * num_nodes) {
                val |= (1 << j);
                masker |= (1 << j);
            } else if(bit_count[i * 32 + j] < (1.0 - threshold) * num_nodes) {
                masker |= (1 << j);
            }
        }
        vals[i] = val;
        mask[i] = masker;
    }
}

__global__ void check_feats(int32_t *feature_arr, int32_t *index_arr, int num_nodes, int feature_len, int *mask, int *vals, long long unsigned *count) {
    __shared__ long long unsigned ctr;
    ctr = 0;
    __syncthreads();
    for(int i = blockIdx.x; i < num_nodes; i += gridDim.x) {
        int64_t nodeId = index_arr[i];
        for(int j = threadIdx.x; j < feature_len; j += blockDim.x) {
            int32_t val = feature_arr[nodeId * feature_len + j];
            if((val & mask[j]) == vals[j])
                atomicAdd(&ctr, __popc(mask[j]));
        }
        __syncthreads();
        if(ctr > feature_len && threadIdx.x == 0) {
            //printf("Node %d\n", i);
            atomicAdd(count, ctr - feature_len);
        }
        __syncthreads();
        ctr = 0;
        __syncthreads();
    }
}

__global__ void check_feats_strict(int32_t *feature_arr, int32_t *index_arr, int num_nodes, int feature_len, int *mask, int *vals, long long unsigned *count) {
    __shared__ long long unsigned ctr, ctr2;
    ctr = 0;
    for(int i = blockIdx.x; i < num_nodes; i += gridDim.x) {
        int64_t nodeId = index_arr[i];
        ctr2 = 0;
        __syncthreads();
        for(int j = threadIdx.x; j < feature_len; j += blockDim.x) {
            int32_t val = feature_arr[nodeId * feature_len + j];
            if((val & mask[j]) == vals[j]) {
                atomicAdd(&ctr, __popc(mask[j]));
                atomicAdd(&ctr2, 1);
            }
        }
        __syncthreads();
        if(ctr2 == feature_len && threadIdx.x == 0) {
            atomicAdd(count, ctr);
        }
        __syncthreads();
        ctr = 0;
        __syncthreads();
    }
}


__global__ void classify_nodes(int *masks, int *vals, int num_centroids,
    int32_t *typecast_feats, int32_t *index_arr, int num_nodes, int32_t *dev_cluster, int feature_len) {
    __shared__ int min_dist;
    int warpId = threadIdx.x / DWARP_SIZE;
    int numWarps = blockDim.x / DWARP_SIZE;
    const bool warpLeader = (threadIdx.x % DWARP_SIZE == 0);
    for(int i = blockIdx.x; i < num_nodes; i += gridDim.x) {
        min_dist = INT_MAX;
        int64_t nodeId = index_arr[i];
        __syncthreads();
        // Align to warp factor
        for(int j = warpId; j < (num_centroids + numWarps - (num_centroids % numWarps)); j += numWarps) {
            int dist = 0;
            if(j < num_centroids) {
                for(int k = threadIdx.x % DWARP_SIZE; k < feature_len; k += DWARP_SIZE) {
                    //typecast_feats[i * feature_len + k] ^ dev_centroids[j * feature_len + k];
                    int32_t val = (typecast_feats[nodeId * feature_len + k] & masks[j * feature_len + k]) ^ vals[j * feature_len + k];
                    dist += __popc(val);
                }
                __syncwarp();
                for (int offset = 16; offset > 0; offset /= 2)
                    dist += __shfl_down_sync(FULL_MASK, dist, offset);
                __syncwarp();
                if(warpLeader) {
                    atomicMin(&min_dist, dist);
                }
            }
            __syncthreads();
            if(warpLeader) {
                if(min_dist == dist) {
                    atomicExch(&dev_cluster[i], j);
                }
            }
        }
    }
}

__global__ void calc_distances(int32_t *centroids, int num_centroids, int32_t *typecast_feats, 
        int32_t *index_arr, int num_nodes, int32_t *dist_vector, int feature_len) {
    int warpId = threadIdx.x / DWARP_SIZE;
    int numWarps = blockDim.x / DWARP_SIZE;
    const bool warpLeader = (threadIdx.x % DWARP_SIZE == 0);
    for(int i = blockIdx.x; i < num_nodes; i += gridDim.x) {
        int64_t nodeId = index_arr[i];
        dist_vector[i] = INT_MAX;
        __syncthreads();
        // Align to warp factor
        for(int j = warpId; j < (num_centroids + numWarps - (num_centroids % numWarps)); j += numWarps) {
            if(j < num_centroids) {
                int dist = 0;
                for(int k = threadIdx.x % DWARP_SIZE; k < feature_len; k += DWARP_SIZE) {
                    //typecast_feats[i * feature_len + k] ^ dev_centroids[j * feature_len + k];
                    int32_t val = (typecast_feats[nodeId * feature_len + k] ^ centroids[j * feature_len + k]);
                    dist += __popc(val);
                }
                __syncwarp();
                for (int offset = 16; offset > 0; offset /= 2)
                    dist += __shfl_down_sync(FULL_MASK, dist, offset);
                __syncwarp();
                if(warpLeader) {
                    atomicMin(&dist_vector[i], dist);
                }
            }
        }
    }
}

__global__ void pick_max_distance(int32_t *dist_vector, int num_nodes, int *choice) 
{
    if(blockIdx.x > 0)
        return;
    
    __shared__ int max_dist;
    max_dist = 0;
    __syncthreads();
    for(int i = threadIdx.x; i < num_nodes; i += blockDim.x) {
        if(dist_vector[i] > max_dist)
            atomicMax(&max_dist, dist_vector[i]);
        __syncthreads();
        if(max_dist == dist_vector[i]) {
            atomicExch(choice, i);
        }
        __syncthreads();
    }
}

#define SHARED_MEM (32 * 256)
__global__ void compute_new_centroids(int32_t *dev_centroids, int num_centroids, int *centroid_count,
    int32_t *typecast_feats, int32_t *index_arr, int num_nodes, int32_t *dev_cluster, int feature_len) {
    __shared__ int bits_set[SHARED_MEM];
    for(int cur_centroid = blockIdx.x; cur_centroid < num_centroids; cur_centroid += gridDim.x) {
        // Reset bit distances for this centroid
        for(int index = threadIdx.x; index < SHARED_MEM; index += blockDim.x)
            bits_set[index] = 0;
        centroid_count[cur_centroid] = 0;
        __syncthreads();
        for(int i = 0; i < num_nodes; ++i) {
            int64_t nodeId = index_arr[i];
            // Find relevant node
            if(dev_cluster[i] == cur_centroid) {
                if(threadIdx.x == 0)
                    centroid_count[cur_centroid]++;
                __syncthreads();
                // Compute distance
                for(int id = threadIdx.x; id < feature_len; id += blockDim.x) {
                    for(int bit = 0; bit < 32; ++bit) {
                        if(typecast_feats[nodeId * feature_len + id] & (1 << bit))
                            bits_set[id * 32 + bit]++;
                    }
                }
            }
        }
        __syncthreads();
        // Compute new centroid based on obtained bit distances
        for(int id = threadIdx.x; id < feature_len; id += blockDim.x) {
            int32_t bitmask = 0;
            for(int bit = 0; bit < 32; ++bit) {
                if(bits_set[id * 32 + bit] > centroid_count[cur_centroid] / 2)
                    bitmask |= (1 << bit);
            }
            dev_centroids[cur_centroid * feature_len + id] = bitmask;
        }
    }
}

__global__ void create_mask_many(int *masks, int *vals, int num_centroids, int32_t *centroid_count, int32_t *typecast_feats, int32_t *index_arr, 
        int32_t *dev_cluster, int feature_len, int num_nodes, float threshold) {

    __shared__ float bits_set[SHARED_MEM];
    for(int cur_centroid = blockIdx.x; cur_centroid < num_centroids; cur_centroid += gridDim.x) {
        // Reset bit distances for this centroid
        for(int index = threadIdx.x; index < SHARED_MEM; index += blockDim.x)
            bits_set[index] = 0;
        centroid_count[cur_centroid] = 0;
        __syncthreads();
        for(int i = 0; i < num_nodes; ++i) {
            int64_t nodeId = index_arr[i];
            // Find relevant node
            if(dev_cluster[i] == cur_centroid) {
                if(threadIdx.x == 0)
                    centroid_count[cur_centroid]++;
                __syncthreads();
                // Compute distance
                for(int id = threadIdx.x; id < feature_len; id += blockDim.x) {
                    for(int bit = 0; bit < 32; ++bit) {
                        if(typecast_feats[nodeId * feature_len + id] & (1 << bit))
                            bits_set[id * 32 + bit]++;
                    }
                }
            }
        }
        __syncthreads();
        // Compute new masks based on obtained bit distances
        for(int id = threadIdx.x; id < feature_len; id += blockDim.x) {
            int32_t val = 0;
            int32_t masker = 0;
            for(int bit = 0; bit < 32; ++bit) {
                if(bits_set[id * 32 + bit] >= threshold * centroid_count[cur_centroid]) {
                    val |= (1 << bit);
                    masker |= (1 << bit);
                } else if(bits_set[id * 32 + bit] <= (1.0 - threshold) * centroid_count[cur_centroid]) {
                    masker |= (1 << bit);
                }
                /*printf("C%d, B%d: %f %f; %f %f; %d %d\n", cur_centroid, id * 32 + bit, 
                    bits_set[id * 32 + bit],
                    cur_centroid_num_nodes,
                    threshold * cur_centroid_num_nodes,
                    (1.0 - threshold) * cur_centroid_num_nodes, 
                    bits_set[id * 32 + bit] <= (1.0 - threshold) * cur_centroid_num_nodes,
                    bits_set[id * 32 + bit] >= threshold * cur_centroid_num_nodes);*/
            }
            masks[cur_centroid * feature_len + id] = masker;
            vals[cur_centroid * feature_len + id] = val;
        }
        __syncthreads();
    }
}

__global__ void check_feats_many(int32_t *feature_arr, int32_t *index_arr, int num_nodes, int32_t *dev_cluster, 
        int feature_len, int *masks, int *vals, long long unsigned *count, bool strict) {
    __shared__ long long unsigned ctr, ctr2;
    ctr = 0;
    ctr2 = 0;
    __syncthreads();
    for(int i = blockIdx.x; i < num_nodes; i += gridDim.x) {
        int64_t nodeId = index_arr[i];
        int clusterId = dev_cluster[i];
        for(int j = threadIdx.x; j < feature_len; j += blockDim.x) {
            int32_t val = feature_arr[nodeId * feature_len + j];
            if((val & masks[clusterId * feature_len + j]) == vals[clusterId * feature_len + j]) {
                atomicAdd(&ctr, __popc(masks[clusterId * feature_len + j]));
            }
            atomicAdd(&ctr2, __popc(masks[clusterId * feature_len + j]));
        }
        __syncthreads();\
        if(threadIdx.x == 0) {
            // All bits must match!
            if(strict && ctr == ctr2)
                atomicAdd(count, ctr);
            // Relaxed. Enough bits must match to justify the tracking overhead
            else if(!strict && ctr > feature_len)
                atomicAdd(count, ctr - feature_len);
        }
        __syncthreads();
        ctr = 0;
        __syncthreads();
    }
}

void comp_decomp_with_single_manager(uint8_t* device_input_ptrs, const size_t input_buffer_len, size_t num_buffers)
{
    //size_t num_buffers = input_buffer_lengths.size();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int chunk_size = 1 << 16;
    nvcompType_t data_type = NVCOMP_TYPE_CHAR;

    nvcompBatchedLZ4Opts_t format_opts{data_type};
    LZ4Manager nvcomp_manager{chunk_size, format_opts, stream};

    std::vector<CompressionConfig> comp_configs;
    comp_configs.reserve(num_buffers);

    std::vector<uint8_t*> comp_result_buffers(num_buffers);

    for(size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
        uint8_t* input_data = &device_input_ptrs[ix_buffer * input_buffer_len];
        size_t input_length = input_buffer_len;//input_buffer_lengths[ix_buffer];

        comp_configs.push_back(nvcomp_manager.configure_compression(input_length));
        auto& comp_config = comp_configs.back();

        CUDA_CHECK(cudaMalloc(&comp_result_buffers[ix_buffer], comp_config.max_compressed_buffer_size));

        nvcomp_manager.compress(input_data, comp_result_buffers[ix_buffer], comp_config);    
    }

    std::vector<uint8_t*> decomp_result_buffers(num_buffers);
    for(size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
        auto decomp_config = nvcomp_manager.configure_decompression(comp_configs[ix_buffer]);

        CUDA_CHECK(cudaMalloc(&decomp_result_buffers[ix_buffer], decomp_config.decomp_data_size));

        nvcomp_manager.decompress(decomp_result_buffers[ix_buffer], comp_result_buffers[ix_buffer], decomp_config);    
    }

    for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
        CUDA_CHECK(cudaFree(decomp_result_buffers[ix_buffer]));
        CUDA_CHECK(cudaFree(comp_result_buffers[ix_buffer]));
    }
}

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
    this->num_sets = ((nodes_per_gpu * 10 + num_ways - 1) / num_ways) * num_gpus;
    this->cache_capacity = num_sets * num_ways * num_gpus;
    this->flags = flags;
    this->dev_start = dev_start;

    std::cout << "Number of sets: " << num_sets << ", cache capacity: " << cache_capacity << "\n";

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
        cudaDeviceGetAttribute(&maxShmem[i], cudaDevAttrMaxSharedMemoryPerBlock, device_id);

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

    //-------------- COMPRESSION CODE ----------
    if(flags & (DYN_COMP | DYN_COMP_CPU | DYN_COMP_TEST)) {
        chunk_size = 4;
        unsigned num_centroids = 100;//nodes_per_gpu * 0.01;
        // Init data structures for future use
        cudaMalloc(&comp_mask, feature_len * sizeof(int));
        cudaMalloc(&comp_bitval, feature_len * sizeof(int));
        int *dev_num_bits;
        int *mask, *vals;
        int *h_mask;
        long long unsigned *count_stuff;
        long long unsigned *host_count;
        int32_t *dev_cluster;
        int *index_arr;
        int32_t *masks, *multivals;
        int32_t *dist_vector, *host_vector;
        int32_t *host_pick, *dev_pick;
        int32_t *dev_centroids_count;
        int *host_centroid_count;
        // Init all memory needed for compression
        cudaMalloc(&dev_num_bits, feature_len * 32 * sizeof(int));
        cudaMalloc(&mask, feature_len * sizeof(int));
        cudaMalloc(&vals, feature_len * sizeof(int));
        cudaMallocHost(&h_mask, feature_len * sizeof(int));
        cudaMalloc(&count_stuff, sizeof(long long unsigned));
        cudaMallocHost(&host_count, sizeof(long long unsigned));
        cudaMalloc(&dev_cluster, nodes_per_gpu * sizeof(int32_t));
        cudaMallocHost(&index_arr, nodes_per_gpu * sizeof(int));
        cudaMalloc(&masks, num_centroids * feature_len * sizeof(int32_t));
        cudaMalloc(&multivals, num_centroids * feature_len * sizeof(int32_t));
        cudaMalloc(&dist_vector,  nodes_per_gpu * sizeof(int32_t));
        cudaMallocHost(&host_vector,  nodes_per_gpu * sizeof(int32_t));
        cudaMallocHost(&host_pick, sizeof(int32_t));
        cudaMalloc(&dev_pick, sizeof(int32_t));
        cudaMalloc(&dev_centroids_count, num_centroids * sizeof(int32_t));
        cudaMallocHost(&host_centroid_count, num_centroids * sizeof(int));

        cudaCheckError();
        // Start logic for compression
        int32_t *typecast_feats = (int32_t*)cpu_features;
        cudaCheckError();
        cudaMemset(dev_num_bits, 0, feature_len * 32 * sizeof(int));
        cudaCheckError();
        count_bit_kernel<<<32, 512>>>
            (typecast_feats, index_array, nodes_per_gpu, feature_len, dev_num_bits);
        cudaDeviceSynchronize();
        cudaCheckError();
        float max_comp = 0;
        printf("Num nodes: %ld\n", nodes_per_gpu);
        for(float threshold = 0.7; threshold <= 1.0; threshold += 0.05) {
            cudaMemset(mask, 0, feature_len * sizeof(int));
            cudaMemset(vals, 0, feature_len * sizeof(int));
            create_mask<<<1, 512>>> (dev_num_bits, mask, vals, feature_len, nodes_per_gpu, threshold);
            cudaMemcpy(h_mask, mask, feature_len * sizeof(int), cudaMemcpyDeviceToHost);
            cudaCheckError();
            int popc = 0;
            for(int i = 0; i < feature_len; ++i) {
                //printf("%x ", h_mask[i]);
                popc += __builtin_popcount(h_mask[i]);
            }
            //printf("\n");
            printf("Set bits %d of %d (%f%%)\n", popc, feature_len * 32, (double)(popc) * 100.0 / ((double)feature_len * 32.0));
            cudaMemset(count_stuff, 0, sizeof(long long unsigned));
            check_feats<<<32, 512>>> (typecast_feats, index_array, nodes_per_gpu, feature_len, mask, vals, count_stuff);
            cudaMemcpy(host_count, count_stuff, sizeof(long long unsigned), cudaMemcpyDeviceToHost);
            cudaCheckError();
            printf("Threshold %f: Single-pass non-strict counts: %llu (Total %ld, %.3f%%)\n", threshold, *host_count, 
                nodes_per_gpu * feature_len * 32, (double)*host_count * 100 / (nodes_per_gpu * feature_len * 32.0));
            // Store the mask/value for max compressed format
            if(*host_count > max_comp) {
                printf("Selected threshold %f\n", threshold);
                cudaMemcpy(comp_mask, mask, feature_len * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(comp_bitval, vals, feature_len * sizeof(float), cudaMemcpyDeviceToDevice);
                max_comp = *host_count;
            }
        }

        //comp_decomp_with_single_manager((uint8_t *)typecast_feats, feature_len * 4, nodes_per_gpu);
        /*
        cudaMemset(dev_cluster, 0, nodes_per_gpu * sizeof(int32_t));
        cudaMemcpy(index_arr, index_array, nodes_per_gpu * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemset(masks, -1, num_centroids * feature_len * sizeof(int32_t));
        cudaMemcpy(multivals, &typecast_feats[index_arr[0] * feature_len], feature_len * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaCheckError();
        
        // K-means++ initialization algorithm
        for(int i = 1; i < num_centroids; ++i) {
            calc_distances<<<32, 512>>>(multivals, i, typecast_feats, index_array, nodes_per_gpu, dist_vector, feature_len);
            pick_max_distance<<<1, 512>>>(dist_vector, nodes_per_gpu, dev_pick);
            cudaMemcpy(host_pick, dev_pick, sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_vector, dist_vector, nodes_per_gpu * sizeof(int32_t), cudaMemcpyDeviceToHost);
            //printf("Picked %d; dist %d\n", *host_pick, host_vector[*host_pick]);
            int64_t nodeId = index_arr[*host_pick];
            cudaMemcpy(&multivals[i * feature_len], &typecast_feats[nodeId * feature_len], feature_len * sizeof(int32_t), cudaMemcpyHostToDevice);
        }
        cudaCheckError();

        const int ITERS = 00;
        printf("Iter ");
        for(int i = 0; i < ITERS; ++i) {
            printf("%d, ", i);
            if(i % 20 == 0 && i > 0)
                printf("\n");
            cudaMemset(dev_centroids_count, 0, num_centroids * sizeof(int32_t));
            cudaDeviceSynchronize();
            classify_nodes<<<32, 512>>>(masks, multivals, num_centroids, typecast_feats, index_array, nodes_per_gpu, dev_cluster, feature_len);
            create_mask_many<<<50, 512>>> (masks, multivals, num_centroids, dev_centroids_count, typecast_feats, index_array, dev_cluster, feature_len, nodes_per_gpu, 0.9);
            cudaDeviceSynchronize();
            cudaCheckError();
/*          if(i == 0) {
                cudaMemcpy(host_centroid_count, dev_centroids_count, num_centroids * sizeof(int), cudaMemcpyDeviceToHost);
                printf("Before clustering\n");
                for(int i = 0; i < num_centroids; ++i) {
                    cudaMemcpy(h_mask, &masks[i * feature_len], feature_len * sizeof(int), cudaMemcpyDeviceToHost);
                    int popc = 0;
                    for(int maskId = 0; maskId < feature_len; ++maskId) {
                        //printf("%x ", h_mask[maskId]);
                        popc += __builtin_popcount(h_mask[maskId]);
                    }
                    if(host_centroid_count[i])
                        printf("Centroid %d: Set bits %d of %d (%f%%) for %d nodes\n", i, popc, 
                            feature_len * 32, (double)(popc) * 100.0 / ((double)feature_len * 32.0), host_centroid_count[i]);
                }
            }
        }
        printf("Finished clustering\n");

        cudaMemcpy(host_centroid_count, dev_centroids_count, num_centroids * sizeof(int), cudaMemcpyDeviceToHost);
        for(int i = 0; i < num_centroids; ++i) {
            cudaMemcpy(h_mask, &masks[i * feature_len], feature_len * sizeof(int), cudaMemcpyDeviceToHost);
            int popc = 0;
            for(int maskId = 0; maskId < feature_len; ++maskId) {
                //printf("%x ", h_mask[maskId]);
                popc += __builtin_popcount(h_mask[maskId]);
            }
            if(host_centroid_count[i])
                printf("Centroid %d: Set bits %d of %d (%f%%) for %d nodes\n", i, popc, 
                    feature_len * 32, (double)(popc) * 100.0 / ((double)feature_len * 32.0), host_centroid_count[i]);
        }

        printf("Num nodes: %ld, num_centroids %d\n", nodes_per_gpu, num_centroids);
        for(float threshold = 0.7; threshold <= 1.0; threshold += 0.05) {
            cudaMemset(count_stuff, 0, sizeof(long long unsigned));
            create_mask_many<<<50, 512>>> (masks, multivals, num_centroids, dev_centroids_count, typecast_feats, index_array, dev_cluster, feature_len, nodes_per_gpu, threshold);
            check_feats_many<<<50, 512>>>(typecast_feats, index_array, nodes_per_gpu, dev_cluster, feature_len, masks, multivals, count_stuff, false);
            cudaDeviceSynchronize();
            cudaCheckError();
            cudaMemcpy(host_count, count_stuff, sizeof(long long unsigned), cudaMemcpyDeviceToHost);
            printf("KMeans %f: counts %llu (Total %ld, %.3f%%)\n", threshold, *host_count, 
                nodes_per_gpu * feature_len * 32, (double)*host_count * 100 / (nodes_per_gpu * feature_len * 32.0));
        }
        */
        // Free all memory needed for compression
        cudaFree(dev_num_bits);
        cudaFree(mask);
        cudaFree(vals);
        cudaFreeHost(h_mask);
        cudaFree(count_stuff);
        cudaFreeHost(host_count);
        cudaFree(dev_cluster);
        cudaFreeHost(index_arr);
        cudaFree(masks);
        cudaFree(multivals);
        cudaFree(dist_vector);
        cudaFreeHost(host_vector);
        cudaFreeHost(host_pick);
        cudaFree(dev_pick);
        cudaFree(dev_centroids_count);
        cudaFreeHost(host_centroid_count);
        printf("Finished compression preprocessing\n");
        fflush(stdout);
    }
    //------------------------------

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
        
        cudaMemcpyAsync(device_uncompressed_bytes, host_uncompressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_uncompressed_ptrs, host_uncompressed_ptrs, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);

        // Overallocate output bytes
        void ** host_compressed_ptrs;
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

        size_t * device_compressed_bytes;
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
        printf("%s: Time taken to decompress: %f ms. Throughput: %f MB/s\n", "ndzip",
            (float)TIME_DIFF(decomp_start, decomp_end) / 1000.0,
            (float)in_bytes / TIME_DIFF(decomp_start, decomp_end));
        cudaMemcpy(host_output_data, device_output_data, in_bytes, cudaMemcpyDeviceToHost);
        cudaCheckError();
        for(int i = 0; i < in_bytes / sizeof(float); ++i) {
            if(((float*)cpu_features)[i] != ((float*)host_output_data)[i]) {
                printf("Mismatch at %d: %x vs %x\n", i, ((int32_t*)cpu_features)[i], ((int32_t*)host_output_data)[i]);
                break;
            }
        }

        cudaMalloc(&comp_bitmask, (nodes_per_gpu + 31) / 32 * sizeof(int32_t));
        cudaMemset(comp_bitmask, 0, (nodes_per_gpu + 31) / 32 * sizeof(int32_t));
        cudaCheckError();
        ull *comp_size;
        cudaMallocHost(&comp_size, sizeof(ull));
        cudaMemset(comp_size, 0, sizeof(ull));
        // Our scheme
        int32_t *working_space;
        cudaMemset(compressed_buffer, 0, in_bytes);
        cudaMalloc(&working_space, (1 + 80 * 512 / DWARP_SIZE * feature_len) * sizeof(int32_t));
        cudaMemset(working_space, 0, (1 + 80 * 512 / DWARP_SIZE * feature_len) * sizeof(int32_t));
        compressed_cpu_features_kernel<<<64, 512>>>(comp_mask, comp_bitval, (int32_t*)cpu_features, (int32_t*)compressed_buffer, 
            nodes_per_gpu, feature_len, working_space, comp_bitmask, 4, nullptr, comp_size);
        cudaDeviceSynchronize();
        printf("%s: Uncompressed bytes: %ld, compressed bytes: %d, ratio: %f\n",
            "Us", in_bytes, *comp_size, (float)in_bytes / *comp_size);
        cudaCheckError();
        
        int shmem_size;
        if(feature_len * sizeof(float) * 2 < maxShmem[0])
            shmem_size = feature_len * sizeof(float) * 2;
        else
            shmem_size = 0;
        cudaMemset(device_output_data, 0, in_bytes);
        decomp_start = TIME_NOW;
        test_decompressed_features_kernel<<<64, 512, shmem_size>>>(
            comp_mask, comp_bitval, (int32_t*)compressed_buffer, (int32_t*)device_output_data, 
            nodes_per_gpu, feature_len, comp_bitmask, shmem_size);
        cudaDeviceSynchronize();
        cudaCheckError();
        decomp_end = TIME_NOW;
        cudaMemcpy(host_output_data, device_output_data, in_bytes, cudaMemcpyDeviceToHost);

        printf("%s: Time taken to decompress: %f ms. Throughput: %f MB/s\n", "Us",
            (float)TIME_DIFF(decomp_start, decomp_end) / 1000.0,
            (float)in_bytes / TIME_DIFF(decomp_start, decomp_end));
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
        decompressed_cpu_features_kernel<<<64, 512>>>(
            comp_mask, comp_bitval, (int32_t*)compressed_buffer, (int32_t*)device_output_data, 
            nodes_per_gpu, feature_len, working_space, comp_bitmask);
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
        cudaMemset(comp_bitmask, 0, (total_nodes + 31) / 32 * sizeof(int32_t));
        //total_nodes = 5;
        //int32_t *comp_cpu_vals, *decomp_cpu_vals;
        //cudaMallocHost(&comp_cpu_vals, total_nodes * feature_len * sizeof(int32_t));
        //cudaMallocHost(&decomp_cpu_vals, total_nodes * feature_len * sizeof(int32_t));
        cudaCheckError();

        int32_t *working_space;
        cudaMalloc(&working_space, (1 + 32 * 512 / DWARP_SIZE * feature_len) * sizeof(int32_t));
        cudaCheckError();
        compressed_cpu_features_kernel<<<32, 512>>>(comp_mask, comp_bitval, (int32_t*)cpu_features, 
             (int32_t*)cpu_features, total_nodes, feature_len, working_space, 
             comp_bitmask, 4, &working_space[32 * 512 / DWARP_SIZE * feature_len]);
        //decompressed_cpu_features_kernel<<<32, 512>>>(comp_mask, comp_bitval, (int32_t*)comp_cpu_vals, 
        //    decomp_cpu_vals, total_nodes, feature_len, working_space);
        int32_t *host_comp_count;
        cudaMallocHost(&host_comp_count, sizeof(int32_t));
        cudaCheckError();
        cudaMemcpy(host_comp_count, &working_space[32 * 512 / DWARP_SIZE * feature_len], sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaCheckError();
        printf("Compressed %d nodes out of %ld\n", *host_comp_count, total_nodes);
        cudaFreeHost(host_comp_count);
        cudaCheckError();
        cudaFree(working_space);
        cudaDeviceSynchronize();
        cudaCheckError();
        /*
        int match = true;
        int i = 0;
        for(int j = 0; j < total_nodes; ++j) {
            for(i = 0; i < feature_len; ++i) {
                if(((int32_t *)cpu_features)[j * feature_len + i] != decomp_cpu_vals[j * feature_len + i]) {
                    printf("Mishmatch at %d. Expected %x, got %x\n", i, *(unsigned*)&cpu_features[j * feature_len + i], *(unsigned*)&decomp_cpu_vals[j * feature_len + i]);
                    match = false;
                    break;
                }
            }
            if(!match)
                break;
        }
        printf("Match? %d\n", match);
        fflush(stdout);
        abort();*/
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
        check_compress_size_kernel<<<32, 512>>>((int32_t *)input_feats, &index_array[inserted_feats], comp_size,
            comp_mask, comp_bitval, nodes_per_gpu, feature_len);
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
            check_compress_size_kernel<<<32, 512>>>((int32_t *)input_feats, &index_array[inserted_feats], comp_size,
                comp_mask, comp_bitval, feats_to_insert, feature_len);
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
    int shmem_size;
    // TODO: Do this by the executing GPU. Relevant for heterogeneous GPU machines
    if(maxShmem[0] >= 2 * feature_len * sizeof(int32_t) + sizeof(int32_t) * 512)
        shmem_size = 2 * feature_len * sizeof(int32_t) + sizeof(int32_t) * 512;
    else 
        shmem_size = sizeof(int32_t) * 512;
    if(!(flags & DYN_COMP))
        static_test_lookup_features_kernel<<<32, 512>>>(dev_cache_storage, dev_cache_key, 
            dev_cache_offset, num_nodes, feature_len, input_feats, index_array, 
            num_gpus, num_ways, num_sets, success_lookups, keys_found);
    else
        compressed_test_lookup_features_kernel<<<32, 512, shmem_size>>>
            (dev_cache_storage, dev_cache_key, dev_cache_offset, comp_mask, comp_bitval, 
            num_nodes, feature_len, input_feats, index_array, num_gpus, num_ways, num_sets, 
            shmem_size, success_lookups, keys_found);
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
    int shmem_size;
    // TODO: Do this by the executing GPU. Relevant for heterogeneous GPU machines
    if(maxShmem[0] >= 2 * feature_len * sizeof(int32_t))
        shmem_size = 2 * feature_len * sizeof(int32_t);
    else 
        shmem_size = 0;
    if(flags & DYN_COMP_CPU)
        compress_cpu_transfer_kernel<<<32, 512, shmem_size, stream>>>(dev_cache_storage, 
            node_index, num_nodes, feature_len, output_buffer, comp_bitmask, input_feats, 
            nodeIds, comp_mask, comp_bitval, total_nodes, num_ways, num_sets, shmem_size,
            misses, lookups, inserts);
    else if(flags & DYN_COMP)
        compress_transfer_kernel<<<32, 512, shmem_size, stream>>>(dev_cache_storage, 
            node_index, num_nodes, feature_len, output_buffer, 
            input_feats, nodeIds, comp_mask, comp_bitval, total_nodes, num_ways, num_sets, shmem_size,
            misses, lookups, inserts);
    else
        static_transfer_kernel<<<32, 512, 0, stream>>>(
            dev_cache_storage, dev_cache_key, dev_cache_offset, node_index, num_nodes, 
            feature_len, output_buffer, input_feats, nodeIds, 0, num_gpus, total_nodes, 
            num_ways, num_sets, flags, misses, lookups, inserts);
}