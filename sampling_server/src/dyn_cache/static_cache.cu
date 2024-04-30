#include "static_cache.cuh"
#include "static_cache_kernel.cuh"
#include <bit>
#include <bitset>
#include <cstdint>
#include <iostream>

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
    this->cache_capacity = nodes_per_gpu;
    this->flags = flags;

    std::cout << "Number of sets: " << num_sets << ", cache capacity: " << cache_capacity << "\n";

    auto start = TIME_NOW;

    // Allocate size of pointers
    void **host_cache_ptr = (void**)malloc(num_gpus * sizeof(void*));
    int **host_cache_offset_ptr = (int**)malloc(num_gpus * sizeof(int*));
    int **host_cache_key_ptr = (int**)malloc(num_gpus * sizeof(int*));

    // Allocate pointers for GPU
    cudaMalloc(&dev_cache_storage, num_gpus * sizeof(void*));
    cudaCheckError();
    cudaMalloc(&dev_cache_offset, num_gpus * sizeof(int*));
    cudaCheckError();
    cudaMalloc(&dev_cache_key, num_gpus * sizeof(int*));
    cudaCheckError();

    for(int i = 0; i < num_gpus; i++) {
        int device_id = i + dev_start;
        cudaSetDevice(device_id);

        // Allocate the actual cache
        cudaMalloc(&host_cache_ptr[i], nodes_per_gpu * feature_len * sizeof(float));
        cudaCheckError();
        cudaMalloc(&host_cache_offset_ptr[i], num_sets / num_gpus * num_ways * sizeof(int));
        cudaCheckError();
        cudaMalloc(&host_cache_key_ptr[i], num_sets / num_gpus * num_ways * sizeof(int));
        cudaCheckError();
        static_reset_cache_metadata<<<32, 512>>>(host_cache_key_ptr[i], host_cache_offset_ptr[i], num_sets / num_gpus * num_ways);
        cudaCheckError();
    }
    // Transfer pointers to GPU
    cudaMemcpy(dev_cache_storage, host_cache_ptr, num_gpus * sizeof(void*), cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(dev_cache_offset, host_cache_offset_ptr, num_gpus * sizeof(int*), cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(dev_cache_key, host_cache_key_ptr, num_gpus * sizeof(int*), cudaMemcpyHostToDevice);
    cudaCheckError();
    auto end = TIME_NOW;
    std::cout << "Time taken to allocate data structures:" << (float)TIME_DIFF(start, end) / 1000.0 << " ms\n";

    start = TIME_NOW;
    // Initialize variable for tracking operations
    int *dev_tracker;
    cudaMalloc(&dev_tracker, 2 * sizeof(int));
    cudaMemset(dev_tracker, 0, 2 * sizeof(int));
    cudaCheckError();
    cudaDeviceSynchronize();

    //-------------------- DELETE THIS ----------
    //nodes_per_gpu = total_nodes;
    int32_t *typecast_feats = (int32_t*)cpu_features;
    int *index_arr;
    cudaMallocHost(&index_arr, nodes_per_gpu * sizeof(int));
    cudaMemcpy(index_arr, index_array, nodes_per_gpu * sizeof(int), cudaMemcpyDeviceToHost);
    /*
    int bit_count[32] = {0};
    int num_bits[32] = {0};
    int diff_bits[8] = {0};
    
    for(int i = 0; i < nodes_per_gpu; ++i) {
        int index = index_arr[i];
        int32_t val = 0;
        for(int j = 1; j < feature_len; ++j){
            int32_t prev = typecast_feats[index * feature_len + j - 1];
            val |= (typecast_feats[index * feature_len + j] ^ prev);
        }
        for(int bit = 0; bit < 32; ++bit) {
            if(val & (1 << bit)) {
                bit_count[bit]++;
                num_bits[bit] += __builtin_popcount(val);
                //num_bits2[bit] += __builtin_popcount(val & (1 << bit));
            }
        }
    }
    printf("Nodes: %d, features: %d\n", nodes_per_gpu, nodes_per_gpu * feature_len);
    for(int bit = 0; bit < 32; ++bit) 
        printf("Bit %d: %d\n", bit, bit_count[bit]);
    for(int j = 0; j < feature_len; ++j){
        int32_t val = 0;
        for(int i = 1; i < nodes_per_gpu; ++i) {
            int prev_index = index_arr[i - 1];
            int index = index_arr[i];
            int32_t prev = typecast_feats[prev_index * feature_len + j];
            val |= (typecast_feats[index * feature_len + j] ^ prev);
        }
        for(int bit = 0; bit < 32; ++bit) {
            if(val & (1 << bit))
                num_bits[bit]++;
        }
    }
    printf("\n");
    for(int bit = 0; bit < 32; ++bit) 
        printf("Bit %d: %d\n", bit, num_bits[bit]);
    */
    int *dev_num_bits;
    cudaMalloc(&dev_num_bits, feature_len * 32 * sizeof(int));
    cudaCheckError();
    cudaMemset(dev_num_bits, 0, feature_len * 32 * sizeof(int));
    cudaCheckError();
    count_bit_kernel<<<32, 512>>>
        (typecast_feats, index_array, nodes_per_gpu, feature_len, dev_num_bits);
    cudaDeviceSynchronize();
    cudaCheckError();
    printf("Num nodes: %ld\n", nodes_per_gpu);
    /*
    int *host_num_bits;
    cudaMallocHost(&host_num_bits, feature_len * 32 * sizeof(int));
    cudaMemcpy(host_num_bits, dev_num_bits, feature_len * 32 * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < feature_len; ++i) {
        for(int j = 0; j < 32; ++j) {
            printf("%d ", host_num_bits[i * 32 + j]);
        }
        printf("\n");
    }*/
    int *mask, *vals;
    cudaMalloc(&mask, feature_len * sizeof(int));
    cudaMalloc(&vals, feature_len * sizeof(int));
    int *h_mask;
    cudaMallocHost(&h_mask, feature_len * sizeof(int));
    long long unsigned *count_stuff;
    cudaMalloc(&count_stuff, sizeof(long long unsigned));
    long long unsigned *host_count;
    cudaMallocHost(&host_count, sizeof(long long unsigned));
    for(float threshold = 0.55; threshold <= 1.1; threshold += 0.05) {
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
        printf("\n");
        printf("Set bits %d of %d (%f%%)\n", popc, feature_len * 32, (double)(popc) * 100.0 / ((double)feature_len * 32.0));
        cudaMemset(count_stuff, 0, sizeof(long long unsigned));
        check_feats<<<32, 512>>> (typecast_feats, index_array, nodes_per_gpu, feature_len, mask, vals, count_stuff);
        cudaMemcpy(host_count, count_stuff, sizeof(long long unsigned), cudaMemcpyDeviceToHost);
        cudaCheckError();
        printf("Threshold %f: Single-pass non-strict counts: %llu (Total %ld, %.3f%%)\n", threshold, *host_count, 
            nodes_per_gpu * feature_len * 32, (double)*host_count * 100 / (nodes_per_gpu * feature_len * 32.0));
        
        cudaMemset(count_stuff, 0, sizeof(long long unsigned));
        check_feats_strict<<<32, 512>>> (typecast_feats, index_array, nodes_per_gpu, feature_len, mask, vals, count_stuff);
        cudaMemcpy(host_count, count_stuff, sizeof(long long unsigned), cudaMemcpyDeviceToHost);
        cudaCheckError();
        printf("Threshold %f: Single-pass strict counts: %llu (Total %ld, %.3f%%)\n", threshold, *host_count, 
            nodes_per_gpu * feature_len * 32, (double)*host_count * 100 / (nodes_per_gpu * feature_len * 32.0));
    }

    unsigned num_centroids = 100;//nodes_per_gpu * 0.01;
    int32_t *dev_cluster;
    cudaMalloc(&dev_cluster, nodes_per_gpu * sizeof(int32_t));
    cudaMemset(dev_cluster, 0, nodes_per_gpu * sizeof(int32_t));

    int32_t *dev_centroids_count;
    cudaMalloc(&dev_centroids_count, num_centroids * sizeof(int32_t));

    int32_t *masks, *multivals;
    cudaMalloc(&masks, num_centroids * feature_len * sizeof(int32_t));
    cudaMemset(masks, -1, num_centroids * feature_len * sizeof(int32_t));
    cudaMalloc(&multivals, num_centroids * feature_len * sizeof(int32_t));
    cudaMemcpy(multivals, &typecast_feats[index_arr[0] * feature_len], feature_len * sizeof(int32_t), cudaMemcpyHostToDevice);

    int32_t *dist_vector, *host_vector;
    cudaMalloc(&dist_vector,  nodes_per_gpu * sizeof(int32_t));
    cudaMallocHost(&host_vector,  nodes_per_gpu * sizeof(int32_t));

    int32_t *host_pick, *dev_pick;
    cudaMallocHost(&host_pick, sizeof(int32_t));
    cudaMalloc(&dev_pick, sizeof(int32_t));
    cudaCheckError();
    
    // K-means++ initialization algorithm
    for(int i = 1; i < num_centroids; ++i) {
        calc_distances<<<32, 512>>>(multivals, i, typecast_feats, index_array, nodes_per_gpu, dist_vector, feature_len);
        pick_max_distance<<<1, 512>>>(dist_vector, nodes_per_gpu, dev_pick);
        cudaMemcpy(host_pick, dev_pick, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_vector, dist_vector, nodes_per_gpu * sizeof(int32_t), cudaMemcpyDeviceToHost);
        printf("Picked %d; dist %d\n", *host_pick, host_vector[*host_pick]);
        int64_t nodeId = index_arr[*host_pick];
        cudaMemcpy(&multivals[i * feature_len], &typecast_feats[nodeId * feature_len], feature_len * sizeof(int32_t), cudaMemcpyHostToDevice);
    }
    int *host_centroid_count;
    cudaMallocHost(&host_centroid_count, num_centroids * sizeof(int));
    cudaCheckError();

    const int ITERS = 100;
    for(int i = 0; i < ITERS; ++i) {
        printf("Iter %d\n", i);
        cudaMemset(dev_centroids_count, 0, num_centroids * sizeof(int32_t));
        classify_nodes<<<32, 512>>>(masks, multivals, num_centroids, typecast_feats, index_array, nodes_per_gpu, dev_cluster, feature_len);
        create_mask_many<<<50, 512>>> (masks, multivals, num_centroids, dev_centroids_count, typecast_feats, index_array, dev_cluster, feature_len, nodes_per_gpu, 0.9);
        cudaDeviceSynchronize();
        cudaCheckError();
        if(i == 0) {
            cudaMemcpy(host_centroid_count, dev_centroids_count, num_centroids * sizeof(int), cudaMemcpyDeviceToHost);
            printf("Before clustering\n");
            for(int i = 0; i < num_centroids; ++i) {
                cudaMemcpy(h_mask, &masks[i * feature_len], feature_len * sizeof(int), cudaMemcpyDeviceToHost);
                int popc = 0;
                for(int maskId = 0; maskId < feature_len; ++maskId) {
                    //printf("%x ", h_mask[maskId]);
                    popc += __builtin_popcount(h_mask[maskId]);
                }
                printf("\n");
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
        printf("\n");
        printf("Centroid %d: Set bits %d of %d (%f%%) for %d nodes\n", i, popc, 
            feature_len * 32, (double)(popc) * 100.0 / ((double)feature_len * 32.0), host_centroid_count[i]);
    }

    printf("Num nodes: %ld, num_centroids %d\n", nodes_per_gpu, num_centroids);
    for(float threshold = 0.55; threshold <= 1.0; threshold += 0.05) {
        cudaMemset(count_stuff, 0, sizeof(long long unsigned));
        create_mask_many<<<50, 512>>> (masks, multivals, num_centroids, dev_centroids_count, typecast_feats, index_array, dev_cluster, feature_len, nodes_per_gpu, threshold);
        check_feats_many<<<50, 512>>>(typecast_feats, index_array, nodes_per_gpu, dev_cluster, feature_len, masks, multivals, count_stuff, false);
        cudaDeviceSynchronize();
        cudaCheckError();
        cudaMemcpy(host_count, count_stuff, sizeof(long long unsigned), cudaMemcpyDeviceToHost);
        printf("KMeans %f: counts %llu (Total %ld, %.3f%%)\n", threshold, *host_count, 
            nodes_per_gpu * feature_len * 32, (double)*host_count * 100 / (nodes_per_gpu * feature_len * 32.0));
    }
    /*for(int i = 0; i < num_centroids; ++i) {
        cudaMemcpy(h_mask, &masks[i * feature_len], feature_len * sizeof(int), cudaMemcpyDeviceToHost);
        int popc = 0;
        for(int maskId = 0; maskId < feature_len; ++maskId) {
            printf("%x ", h_mask[maskId]);
            popc += __builtin_popcount(h_mask[maskId]);
        }
        printf("\n");
        printf("Centroid %d: Set bits %d of %d (%f%%) for %d nodes\n", i, popc, 
            feature_len * 32, (double)(popc) * 100.0 / ((double)feature_len * 32.0), host_centroid_count[i]);
    }*/

    abort();
    //-------------------- DELETE THIS ----------

    for(int i = 0; i < num_gpus; i++) {
        //int device_id = i + dev_start;
        //cudaSetDevice(device_id);
        // Insert features into the cache
        insert_features(host_cache_ptr[i], nodes_per_gpu, i, cpu_features, 
            &index_array[nodes_per_gpu * i], total_nodes, dev_tracker);
        cudaDeviceSynchronize();
        cudaCheckError();
    }
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
    free(host_cache_ptr);
    free(host_cache_offset_ptr);
    free(host_cache_key_ptr);
}

// Bulk insert into cache, setting appropriate values
void StaticCache::insert_features(void *cache, int64_t num_nodes, int gpu, float *input_feats, 
    int32_t *index_array, int64_t total_nodes, int *failed_inserts)
{
    static_insert_features_kernel<<<16, 512>>>(cache, dev_cache_key, 
        dev_cache_offset, gpu, num_gpus, num_nodes, feature_len, input_feats, index_array, 
        total_nodes, num_ways, num_sets, flags, failed_inserts);
}

// Find features in cache
void StaticCache::test_lookup_features(int64_t num_nodes, float *input_feats, 
    int32_t *index_array, int *success_lookups, int *keys_found)
{
    static_test_lookup_features_kernel<<<32, 512>>>(dev_cache_storage, dev_cache_key, 
        dev_cache_offset, num_nodes, feature_len, input_feats, index_array, 
        num_gpus, num_ways, num_sets, success_lookups, keys_found);
}

void StaticCache::retrieve(int32_t *nodeIds, int64_t num_nodes, 
    int *node_index, cudaStream_t stream, 
    ull *misses, ull *lookups, ull *inserts)
{
    static_retrieve_kernel<<<32, 512, 0, stream>>>(
        dev_cache_key, dev_cache_offset, node_index, num_nodes, 
        nodeIds, num_gpus, 
        num_ways, num_sets, flags, misses, lookups, inserts);
}

void StaticCache::transfer(int32_t *nodeIds, int64_t num_nodes, 
    float *output_buffer, int *node_index, float *input_feats, int total_nodes, cudaStream_t stream, 
    ull *misses, ull *lookups, ull *inserts)
{
    static_transfer_kernel<<<32, 512, 0, stream>>>(
        dev_cache_storage, dev_cache_key, dev_cache_offset, node_index, num_nodes, feature_len, 
        output_buffer, input_feats, nodeIds, 0, num_gpus, total_nodes, 
        num_ways, num_sets, flags, misses, lookups, inserts);
}