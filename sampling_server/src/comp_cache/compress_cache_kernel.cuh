#ifndef COMPCACHE_H
#define COMPCACHE_H
#include "cache_helper.cuh"
#include "static_cache_kernel.cuh"
#define OFFSET_BITS (40L)
#define GPU_BITS (3L)
#define COMP_BITS (2L)
// 40 bits for offset mask, gives 1TB bytes
#define OFFSET_MASK ((1L << OFFSET_BITS) - 1L)
#define GPU_MASK ((1L << GPU_BITS) - 1L)
#define COMP_MASK ((1L << COMP_BITS) - 1L)
__inline__ __device__ int64_t construct_hash_ptr(int64_t offset, int64_t gpu_id, int64_t compressed)
{
    return  (offset & OFFSET_MASK) | 
            ((gpu_id & GPU_MASK) << (OFFSET_BITS)) |
            ((compressed & COMP_MASK) << (GPU_BITS + OFFSET_BITS));
}

__inline__ __device__ int64_t deconstruct_hash_ptr(int64_t ptr, int64_t &gpu_id, int64_t &compressed)
{
    gpu_id = (ptr >> (OFFSET_BITS)) & GPU_MASK;
    compressed = (ptr >> (GPU_BITS + OFFSET_BITS)) & COMP_MASK;
    return ptr & OFFSET_MASK;
}

// Function to compress and write features
__inline__ __device__ void compress_and_write(int32_t *dest, int32_t *src, 
    int32_t *mask, int32_t *bitval, int32_t feature_len, int32_t chunk_size = 4)
{
    int laneId = threadIdx.x % DWARP_SIZE;
    // First determine metadata bits
    int64_t bitmask_offset = BITS_TO_BYTES((feature_len * sizeof(float) + chunk_size - 1) / chunk_size);
    // 4-byte align
    bitmask_offset = (bitmask_offset + 3) / 4 * 4;
    int32_t bitshift = 0;
    for(int j = laneId; j < (feature_len + DWARP_SIZE - 1) / DWARP_SIZE * DWARP_SIZE; j += DWARP_SIZE) {
        int32_t cur_bitshift = 0;
        int32_t val = 0;
        if(j < feature_len) {
            int32_t local_mask = mask[j];
            // Read WORD from feature
            val = src[j];
            // Init with bits in this float
            cur_bitshift = 32;
            // If this chunk is compressable, subtract the bits saved
            if((val & local_mask) == bitval[j]) {
                cur_bitshift -= __popc(local_mask);
            }
        }
        // Perform scan to obtain starting bit for this thread
        bitshift += warpExclusiveScanSync(FULL_MASK, cur_bitshift);
        //printf("BS %p %d: %d %d\n", dest, j, cur_bitshift, bitshift);
        if(j < feature_len) {
            int32_t local_mask = mask[j];
            int32_t insert_size = 32 - __popc(local_mask);
            int32_t compressed_val = 0;
            // Start from bitshift and insert current compressed feature
            if((val & local_mask) == bitval[j]) {
                //uncomp = true;
                // Number of compressed bits being inserted
                int32_t num_bits = insert_size;
                int32_t temp_insert_size = __clz(local_mask);
                int32_t num_inserted = 0;
                while(num_bits > 0) {
                    num_inserted += temp_insert_size;
                    compressed_val |= (val & (int32_t)(((1L << temp_insert_size) - 1L) << (32 - num_inserted)));
                    //printf("IP %p %d: %x %x (%d, %d)\n", dest, j, compressed_val, val, temp_insert_size, num_inserted);
                    num_bits -= temp_insert_size;
                    // Shift by inserted bits
                    local_mask <<= temp_insert_size;
                    // Shift by masked bits
                    int32_t shift = __clz(~local_mask);
                    local_mask <<= shift;
                    val <<= shift;
                    // Get new insert size
                    temp_insert_size = min(__clz(local_mask), num_bits);
                }
                //printf("COMP %p %d: %x %x %x (%d)\n", dest, j, compressed_val, src[j], mask[j], insert_size);
                // Set bitmask bit
                int32_t word_offset = j / 32;
                int32_t bit_offset = j % 32;
                atomicOr(dest + word_offset, 1 << bit_offset);
            } else {
                // Ignore mask and insert entire chunk
                insert_size = 32;
                compressed_val = val;
                // Unset bitmask bit
                int32_t word_offset = j / 32;
                int32_t bit_offset =  j % 32;
                atomicAnd(dest + word_offset, ~(1 << (bit_offset)));
                //printf("VAL %p %d: %x (%p)\n", dest, j, compressed_val, src + j);
            }
            int32_t fin_ins_bits = 0;
            while(insert_size > 0) {
                // Now perform actual insertion
                int32_t word_offset = (bitmask_offset + bitshift / 8) / 4;
                int32_t bit_offset = (bitmask_offset * 8 + bitshift) % 32;
                int32_t insert_bits = min(insert_size, 32 - bit_offset);
                int32_t insert = ((compressed_val << fin_ins_bits) & (((1L << insert_bits) - 1L) << (32 - insert_bits))) >> bit_offset;

                //if(uncomp)
                //    printf("PREV %p %d (%d): %x\n", dest, j, word_offset, atomicAdd(dest + word_offset, 0));
                // Insert into cache
                atomicOr(dest + word_offset, insert);
                //printf("%p %d: %x %x %x (%d, %d) (%d, %d)\n", dest, j, compressed_val, insert, 
                //    atomicAdd(dest + word_offset, 0), bit_offset, word_offset, insert_bits, fin_ins_bits);
                // Shift appropriately
                //compressed_val <<= insert_bits;
                fin_ins_bits += insert_bits;
                insert_size -= insert_bits;
                bitshift += insert_bits;
            }
        }
        // Get starting bitshift from previous iteration
        bitshift = __shfl_sync(FULL_MASK, bitshift, DWARP_SIZE - 1);
    }
}

// Function to decompress and write features
__inline__ __device__ void decompress_and_write(int32_t *dest, int32_t *src, 
    int32_t *shm_mask, int32_t *shm_bitval, int32_t feature_len, int32_t chunk_size = 4) {
    int laneId = threadIdx.x % DWARP_SIZE;
    int64_t bitmask_offset = BITS_TO_BYTES((feature_len * sizeof(float) + chunk_size - 1) / chunk_size);
    // 4-byte align
    bitmask_offset = (bitmask_offset + 3) / 4 * 4;
    // Code to decompress
    int32_t bitshift = 0;
    for(int i = laneId; i < (feature_len + DWARP_SIZE - 1) / DWARP_SIZE * DWARP_SIZE; i += DWARP_SIZE) {
        int32_t cur_bitshift = 0;
        bool compressed_feat = false;
        if(i < feature_len) {
            // Check if this feature is in compressed or uncompressed format
            int32_t word_offset = i / 32;
            int32_t bit_offset =  i % 32;
            compressed_feat = (src[word_offset] & (1 << bit_offset));
            // Default 32 bits per thread, less if compressed
            cur_bitshift = 32;
            if(compressed_feat) {
                cur_bitshift -= __popc(shm_mask[i]);
            }
        }
        // Perform scan to obtain starting bit for this thread
        bitshift += warpExclusiveScanSync(FULL_MASK, cur_bitshift);
        // Now decompress
        if(i < feature_len) {
            int32_t num_bits = cur_bitshift;
            int32_t temp_read_size;
            int32_t local_mask = shm_mask[i];
            int32_t temp_dest = 0;
            // Start from bitshift and insert current compressed feature
            if(compressed_feat) {
                temp_dest = shm_bitval[i];
                temp_read_size = min(num_bits, __clz(local_mask));
            } else {
                temp_dest = 0;
                temp_read_size = num_bits;
            }

            int32_t fin_read_bits = 0;
            // Number of compressed bits being inserted
            while(num_bits > 0) {
                // Perform actual read
                int32_t word_offset = (bitmask_offset + bitshift / 8) / 4;
                int32_t bit_offset = (bitmask_offset * 8 + bitshift) % 32;
                int32_t read_bits = min(temp_read_size, 32 - bit_offset);
                temp_dest |= ((src[word_offset] << bit_offset) & (((1L << read_bits) - 1L) << (32 - read_bits))) >> fin_read_bits;
                fin_read_bits += read_bits;
                num_bits -= read_bits;
                bitshift += read_bits;
                // If the feature was compressed, do some bitshifts before next iteration
                if(compressed_feat) {
                    // Shift by inserted bits
                    local_mask <<= read_bits;
                    // Shift by masked bits
                    int32_t shift = __clz(~local_mask);
                    local_mask <<= shift;
                    fin_read_bits += shift;
                    // Get new insert size
                    temp_read_size = min(__clz(local_mask), num_bits);
                } else
                    temp_read_size = num_bits;
            }
            dest[i] = temp_dest;
        }
        // Get starting bitshift from previous iteration
        bitshift = __shfl_sync(FULL_MASK, bitshift, DWARP_SIZE - 1);
    }
}

__inline__ __device__ int read_one_iter(int32_t *cpu_src, int32_t *shm_meta, int32_t *shm_working, 
    int min_elems, int max_elems, int feature_len, int bitmask_offset, int start_offset = 0) 
{
    int threadId = threadIdx.x % DWARP_SIZE;
    const int offset = ((GPU_CL_SIZE - (((uint64_t)(cpu_src + start_offset)) % GPU_CL_SIZE)) % GPU_CL_SIZE) / sizeof(int32_t);
    const int onset  = ((GPU_CL_SIZE - (((uint64_t)min_elems * sizeof(int32_t)) % GPU_CL_SIZE)) % GPU_CL_SIZE) / sizeof(int32_t);

    int sub_val = (offset + 7) / 8 * 8; // Round up to nearest multiple of 8
    for(int k = threadId + (offset - sub_val); k < offset; k += DWARP_SIZE) {
        int32_t *dest_element;
        if(k + start_offset < bitmask_offset / sizeof(int32_t)) {
            // ASSUMPTION: Shared memory working region size is 128B/32 elements
            dest_element = (int32_t*)shm_meta + (k + start_offset) % 32;
        } else {
            // ASSUMPTION: Shared memory working region size is 256B/64 elements
            dest_element = (int32_t*)shm_working + (k + start_offset - bitmask_offset / sizeof(int32_t)) % 64;
        }
        // Combined read, to improve PCIe util.
        int32_t src_data = *((cpu_src + start_offset) + k);
        // Selective write to GPU memory
        if(k >= 0 && k < max_elems) {
            *dest_element = src_data;
            
            /*unsigned int mask = __activemask();
            if(k % 32 == 0) {
                printf("1. Write %d bytes; num_threads %d; %p; min %d max %d\n", 
                    (int)(__popc(mask) * sizeof(int32_t)), sub_val, 
                    cpu_src + start_offset, min_elems, max_elems);
            }*/
        }
    }
    if(offset >= min_elems)
        return start_offset + offset;
    // Read can be completed with less than 32 threads if min_elems is small
    if(max_elems - offset <= 24) {
        int add_val = (min_elems - offset + 7) / 8 * 8; // Round up to nearest multiple of 8
        for(int k = threadId + offset; k < offset + add_val; k += DWARP_SIZE) {
            int32_t *dest_element;
            if(k + start_offset < bitmask_offset / sizeof(int32_t)) {
                // ASSUMPTION: Shared memory working region size is 128B/32 elements
                dest_element = (int32_t*)shm_meta + (k + start_offset) % 32;
            } else {
                // ASSUMPTION: Shared memory working region size is 256B/64 elements
                dest_element = (int32_t*)shm_working + (k + start_offset - bitmask_offset / sizeof(int32_t)) % 64;
            }
            // Combined read, to improve PCIe util.
            int32_t src_data = *((cpu_src + start_offset) + k);
            // Selective write to GPU memory
            if(k >= 0 && k < max_elems) {
                *dest_element = src_data;
                
                /*unsigned int mask = __activemask();
                if(k % 32 == 0) {
                    printf("1b. Write %d bytes; num_threads %d; %p; min %d max %d\n", 
                        (int)(__popc(mask) * sizeof(int32_t)), add_val, 
                        cpu_src + start_offset, min_elems, max_elems);
                }*/
            }
        }
        __syncwarp();
        return start_offset + min(max_elems, offset + add_val);
    }
    int k;
    for(k = threadId + offset; k < min_elems + offset + onset; k += DWARP_SIZE) {
        __syncwarp();
        int32_t *dest_element;
        if(k + start_offset < bitmask_offset / sizeof(int32_t)) {
            // ASSUMPTION: Shared memory working region size is 128B/32 elements
            dest_element = (int32_t*)shm_meta + (k + start_offset) % 32;
        } else {
            // ASSUMPTION: Shared memory working region size is 256B/64 elements
            dest_element = (int32_t*)shm_working + (k + start_offset - bitmask_offset / sizeof(int32_t)) % 64;
        }
        // Combined read, to improve PCIe util.
        int32_t src_data = *((cpu_src + start_offset) + k);
        // Selective write to GPU memory
        if(k < max_elems) {
            *dest_element = src_data;
            
            /*unsigned int mask = __activemask();
            if(k % 32 == 0) {
                printf("2. Write %d bytes; offset %d; min+onset %d; %p; min %d max %d\n", 
                (int)(__popc(mask) * sizeof(int32_t)), offset, min_elems + onset, 
                cpu_src + start_offset, min_elems, max_elems);
            }*/
        }
    }
    __syncwarp();

    return start_offset + min(max_elems, min_elems + offset + onset);
}

// Function to decompress and write features
__inline__ __device__ void decompress_and_write_cpu(int32_t *dest, int32_t *src, 
    int32_t *shm_mask, int32_t *shm_bitval, int32_t feature_len, int32_t *workspace,
    int32_t chunk_size = 4) {
    int laneId = threadIdx.x % DWARP_SIZE;
    int64_t bitmask_offset = BITS_TO_BYTES((feature_len * sizeof(float) + chunk_size - 1) / chunk_size);
    // 4-byte align
    bitmask_offset = (bitmask_offset + 3) / 4 * 4;

    // Shared memory buffers
    int32_t *metadata = workspace;
    int32_t *working_data = workspace + 32;
    int metadata_offset = 0, working_offset = 0;
    // Read up to 64elems/256B from src buffer
    int offset = read_one_iter(src, metadata, working_data, 1, 
                               min(32, feature_len), feature_len, bitmask_offset);

    if(offset > bitmask_offset / sizeof(int32_t)) {
        metadata_offset = bitmask_offset / sizeof(int32_t);
        working_offset = offset;
        /*if(laneId == 0)
        printf("Offset: %d, Metadata offset = %d, working offset = %d\n", 
            offset, metadata_offset, working_offset);*/
    } else {
        metadata_offset = offset;
        // Only read metadata so far, so read working data now
        working_offset = bitmask_offset / sizeof(int32_t);
        working_offset = read_one_iter(src, metadata, working_data, min(32, feature_len), 
                                min(64, feature_len), feature_len, bitmask_offset, working_offset);
        /*if(laneId == 0)
        printf("Offset: %d, Metadata offset = %d, working offset = %d\n", 
            offset, metadata_offset, working_offset);*/
    }

    // Code to decompress
    int32_t bitshift = 0;
    for(int i = laneId; i < (feature_len + DWARP_SIZE - 1) / DWARP_SIZE * DWARP_SIZE; i += DWARP_SIZE) {
        int read_metadata = false;
        // This thread needs next set of metadata
        if(i / 32 >= metadata_offset && i < feature_len) {
            read_metadata = true;
        }
        read_metadata = __ballot_sync(FULL_MASK, read_metadata);
        if(read_metadata) {
            // Read next 128B of metadata
            metadata_offset = read_one_iter(src, metadata, working_data, 
                min(32UL, bitmask_offset / sizeof(int32_t) - metadata_offset), 
                min(32UL, bitmask_offset / sizeof(int32_t) - metadata_offset), feature_len, bitmask_offset, metadata_offset);
            /*if(laneId == 0)
                printf("2. [Added] Metadata offset = %d, working offset = %d\n", 
                    metadata_offset, working_offset);*/
        }
        int32_t cur_bitshift = 0;
        bool compressed_feat = false;
        if(i < feature_len) {
            // Check if this feature is in compressed or uncompressed format
            // i / 32 converts bits to words, % 32 because shmem buffer holds 32 words
            int32_t word_offset = (i / 32) % 32;
            int32_t bit_offset =  i % 32;
            compressed_feat = (metadata[word_offset] & (1 << bit_offset));
            // Default 32 bits per thread, less if compressed
            cur_bitshift = 32;
            if(compressed_feat) {
                cur_bitshift -= __popc(shm_mask[i]);
            }
        }
        // Perform scan to obtain starting bit for this thread
        bitshift += warpExclusiveScanSync(FULL_MASK, cur_bitshift);
        int read_data = false;
        // This thread needs next set of working data
        if((bitshift + cur_bitshift) / 32 >= working_offset - bitmask_offset / sizeof(int32_t) && i < feature_len) {
            read_data = true;
        }
        read_data = __ballot_sync(FULL_MASK, read_data);
        if(read_data) {
            // Read next 128B of metadata
            working_offset = read_one_iter(src, metadata, working_data, 
                min(32, feature_len - working_offset), 
                min(64, feature_len - working_offset), feature_len, bitmask_offset, working_offset);
            /*if(laneId == 0)
                printf("3. Metadata offset = %d, [Added] working offset = %d\n", 
                    metadata_offset, working_offset);*/
        }
        
        // Now decompress
        if(i < feature_len) {
            int32_t num_bits = cur_bitshift;
            int32_t temp_read_size;
            int32_t local_mask = shm_mask[i];
            int32_t temp_dest = 0;
            // Start from bitshift and insert current compressed feature
            if(compressed_feat) {
                temp_dest = shm_bitval[i];
                temp_read_size = min(num_bits, __clz(local_mask));
            } else {
                temp_dest = 0;
                temp_read_size = num_bits;
            }

            int32_t fin_read_bits = 0;
            // Number of compressed bits being inserted
            while(num_bits > 0) {
                // Perform actual read
                // bitshift / 32 converts bits to words, % 64 because shmem buffer holds 64 words
                int32_t word_offset = (bitshift / 32) % 64;
                int32_t bit_offset = bitshift % 32;
                int32_t read_bits = min(temp_read_size, 32 - bit_offset);
                temp_dest |= ((working_data[word_offset] << bit_offset) & (((1L << read_bits) - 1L) << (32L - read_bits))) >> fin_read_bits;
                fin_read_bits += read_bits;
                num_bits -= read_bits;
                bitshift += read_bits;
                // If the feature was compressed, do some bitshifts before next iteration
                if(compressed_feat) {
                    // Shift by inserted bits
                    local_mask <<= read_bits;
                    // Shift by masked bits
                    int32_t shift = __clz(~local_mask);
                    local_mask <<= shift;
                    fin_read_bits += shift;
                    // Get new insert size
                    temp_read_size = min(__clz(local_mask), num_bits);
                } else
                    temp_read_size = num_bits;
            }
            dest[i] = temp_dest;
        }
        // Get starting bitshift from previous iteration
        bitshift = __shfl_sync(FULL_MASK, bitshift, DWARP_SIZE - 1);
    }
}

__global__ void check_compress_size_kernel(int32_t *input_feats, int32_t *index_array, int64_t *compressed_size,
    int32_t *mask, int32_t *values, int64_t num_feats, int64_t feature_len, int64_t chunk_size = 4)
{
    int64_t feat_bytes = feature_len * sizeof(int32_t);
    __shared__ long long unsigned ctr;
    for(int i = blockIdx.x; i < num_feats; i += gridDim.x) {
        ctr = 0;
        __syncthreads();
        int64_t nodeId = index_array[i];
        for(int j = threadIdx.x; j < feature_len; j += blockDim.x) {
            int32_t val = input_feats[nodeId * feature_len + j];
            if((val & mask[j]) == values[j])
                atomicAdd(&ctr, __popc(mask[j]));
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            // Calc bytes for compressed data
            int64_t metadata_size = BITS_TO_BYTES((feat_bytes + chunk_size - 1) / chunk_size);
            int64_t feat_size = feat_bytes - ctr / 8;
            // 4-byte align
            metadata_size = (metadata_size + 3) / 4 * 4;
            feat_size = (feat_size + 3) / 4 * 4;
            int64_t comp_size = metadata_size + feat_size;
            compressed_size[i] = min(comp_size, feat_bytes);
        }
        __syncthreads();
    }
}

__global__ void compressed_insert_features_kernel(
    void *cache, int **cache_key, ull **cache_val, 
    int *mask, int *bitval, int64_t *comp_size, int32_t prev_offset,
    int gpu_id, int num_gpus, int64_t num_nodes, int64_t feature_len, 
    float *cpu_features, int32_t *index_array, int64_t total_nodes, 
    int num_ways, int32_t num_sets, int dyn_flags, int chunk_size = 4,
    int *failed_inserts = nullptr)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int laneId = threadIdx.x % DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    for(int i = warpId; i < num_nodes; i += numWarps) {
        int64_t insert_size;
        int64_t start_offset = prev_offset;
        if(i == 0) {
            insert_size = comp_size[i];
        }
        else {
            start_offset += comp_size[i - 1];
            insert_size = comp_size[i] - comp_size[i - 1];
        }
        __syncwarp();
        ull value = 0;
        int64_t nodeId = index_array[i];
        // Insert either uncompressed or compressed
        if(insert_size == feature_len * sizeof(float)) {
            // Uncompressed insert
            memcpy_warp((void *)((char *)cache + start_offset), 
                        &cpu_features[nodeId * feature_len], feature_len);
            value = construct_hash_ptr(start_offset, gpu_id, 0);
        } else {
            // Compressed insert
            compress_and_write((int32_t *)((char *)cache + start_offset), 
                (int32_t *)&cpu_features[nodeId * feature_len], 
                mask, bitval, feature_len, chunk_size);
            value = construct_hash_ptr(start_offset, gpu_id, 1);
            //if(laneId == 0)
            //    printf("%p: Ptr %lx, val %lx\n", ((char *)cache + start_offset), 
            //        start_offset, value);
        }
        __syncwarp();
        
        // Attempt an insert. Check if there's enough space for padded read on both sides
        int slot = static_insert_single_feature(cache_key, cache_val, nodeId, value, 
            num_gpus, num_ways, num_sets);
        // Increment fail counter if appropriate
        if(slot < 0 && failed_inserts != nullptr && laneId == 0) {
            atomicAdd(failed_inserts, 1);
        }
    }
}

__global__ void compressed_cpu_features_kernel(
    int32_t *mask, int32_t *bitval, int32_t *input_features, int32_t *output_features, 
    int64_t total_nodes, int64_t feature_len, int32_t *workspace, int32_t *bitmask, 
    int chunk_size = 4, int *comp_ctr = nullptr, ull *compress_size = nullptr)
{
    int64_t feat_bytes = feature_len * sizeof(int32_t);
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int laneId = threadIdx.x % DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    int32_t *myworkspace = &workspace[warpId * feature_len];
    for(int i = warpId; i < total_nodes; i += numWarps) {
        int32_t ctr = 0;
        for(int j = laneId; j < feature_len; j += DWARP_SIZE) {
            int32_t val = input_features[i * feature_len + j];
            if((val & mask[j]) == bitval[j])
                ctr += __popc(mask[j]);
        }
        // Get sum of ctr value from warp
        ctr = warpInclusiveScanSync(FULL_MASK, ctr);
        long compressed = 0;
        if(laneId == DWARP_SIZE - 1) {
            // Calc bytes for compressed data
            int64_t metadata_size = BITS_TO_BYTES((feat_bytes + chunk_size - 1) / chunk_size);
            int64_t feat_size = feat_bytes - ctr / 8;
            // 4-byte align
            metadata_size = (metadata_size + 3) / 4 * 4;
            feat_size = (feat_size + 3) / 4 * 4;
            int64_t comp_size = metadata_size + feat_size;
            if(comp_size < feat_bytes)
                compressed = comp_size;
        }
        compressed = __shfl_sync(FULL_MASK, compressed, DWARP_SIZE - 1);
        if(compressed) {
            // Compressed insert
            compress_and_write(myworkspace, (int32_t *)&input_features[i * feature_len], 
                mask, bitval, feature_len, chunk_size);
            __syncwarp();
            for(int j = laneId; j < feature_len; j += DWARP_SIZE) {
                //if(i == 0)
                //    printf("%d: Orig: %x, Comp: %x\n", j, ((int32_t*)input_features)[i * feature_len + j], myworkspace[j]);
                output_features[i * feature_len + j] = myworkspace[j];
                myworkspace[j] = 0;
            }
            if(laneId == 0) {
                atomicOr(&bitmask[i / 32], 1 << (i % 32));
                if(comp_ctr != nullptr)
                    atomicAdd(comp_ctr, 1);
                if(compress_size != nullptr) {
                    atomicAdd(compress_size, (ull)compressed);
                }
            }
        } else {
            for(int j = laneId; j < feature_len; j += DWARP_SIZE) {
                output_features[i * feature_len + j] = input_features[i * feature_len + j];
            }
            if(laneId == 0 && compress_size != nullptr) {
                atomicAdd(compress_size, feature_len * sizeof(float));
            }
        }
    }
}

__global__ void decompressed_cpu_features_kernel(
    int32_t *mask, int32_t *bitval, int32_t *input_features, int32_t *output_features, 
    int64_t total_nodes, int64_t feature_len, int32_t *workspace, int32_t *bitmask, 
    int chunk_size = 4)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int laneId = threadIdx.x % DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    int32_t *myworkspace = &workspace[warpId * feature_len];
    for(int i = warpId; i < total_nodes; i += numWarps) {
        // Compressed insert
        if(bitmask[i / 32] & (1 << (i % 32))){
            decompress_and_write(myworkspace, (int32_t *)&input_features[i * feature_len], 
                mask, bitval, feature_len, chunk_size);
            for(int j = laneId; j < feature_len; j += DWARP_SIZE) {
                output_features[i * feature_len + j] = myworkspace[j];
                myworkspace[j] = 0;
            }
        } else {
            memcpy_warp((void *)&output_features[i * feature_len], 
                        (void *)&input_features[i * feature_len], feature_len);
        }
    }
}

__global__ void test_decompressed_features_kernel(
    int32_t *mask, int32_t *bitval, int32_t *input_features, int32_t *output_features, 
    int64_t total_nodes, int64_t feature_len, int32_t *bitmask, size_t shmem_size,
    int chunk_size = 4)
{
    extern __shared__ int32_t shared_mem[];
    int32_t *shm_mask = shared_mem, *shm_bitval = &shared_mem[feature_len];
    if(shmem_size >= feature_len * sizeof(int32_t) * 2) {
        for(int i = threadIdx.x; i < feature_len; i += blockDim.x) {
            shm_mask[i] = mask[i];
            shm_bitval[i] = bitval[i];
        }
    } else {
        shm_mask = mask;
        shm_bitval = bitval;
    }
    __syncthreads();

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int laneId = threadIdx.x % DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    for(int i = warpId; i < total_nodes; i += numWarps) {
        // Compressed insert
        if(bitmask[i / 32] & (1 << (i % 32))){
            decompress_and_write(&output_features[i * feature_len], (int32_t *)&input_features[i * feature_len], 
                shm_mask, shm_bitval, feature_len, chunk_size);
        } else {
            memcpy_warp((void *)&output_features[i * feature_len], 
                        (void *)&input_features[i * feature_len], feature_len);
        }
    }
}

__global__ void test_decompressed_features_kernel2(
    const int32_t *mask, const int32_t *bitval, const int32_t *input_features, int32_t *output_features, 
    int64_t total_nodes, int64_t feature_len, const int32_t *bitmask, size_t shmem_size,
    int chunk_size = 4)
{
    extern __shared__ int32_t shared_mem[];
    int32_t *shm_mask = shared_mem, *shm_bitval = &shared_mem[feature_len];
    int32_t *workspace;
    if(shmem_size >= feature_len * sizeof(int32_t) * 2) {
        for(int i = threadIdx.x; i < feature_len; i += blockDim.x) {
            shm_mask[i] = mask[i];
            shm_bitval[i] = bitval[i];
        }
        // 32 elements for metadata, 64 elements for working data
        // = 96 elements per warp
        workspace = &shared_mem[2 * feature_len + (threadIdx.x / DWARP_SIZE) * 96];
    } else {
        shm_mask = (int32_t *)mask;
        shm_bitval = (int32_t *)bitval;
        // 32 elements for metadata, 64 elements for working data
        // = 96 elements per warp
        workspace = &shared_mem[(threadIdx.x / DWARP_SIZE) * 96];
    }
    __syncthreads();

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int laneId = threadIdx.x % DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    for(int i = warpId; i < total_nodes; i += numWarps) {
        // Compressed insert
        if(bitmask[i / 32] & (1 << (i % 32))){
            decompress_and_write_cpu(&output_features[i * feature_len], 
                (int32_t *)&input_features[i * feature_len], 
                shm_mask, shm_bitval, feature_len, workspace, chunk_size);
        } else {
            memcpy_warp((void *)&output_features[i * feature_len], 
                        (void *)&input_features[i * feature_len], feature_len);
        }
    }
}

// [TEST] Check if features properly copied into cache
__global__ void compressed_test_lookup_features_kernel(
    void **dev_cache, int **dev_cache_key, ull **dev_cache_val, int32_t *dev_mask, 
    int32_t *dev_bitval, int64_t num_nodes, int64_t feature_len, void *cpu_features, 
    int32_t *index_array, int num_gpus, int num_ways, int32_t num_sets, int shmem_size,
    int *success_lookups = nullptr, int *keys_found = nullptr, int32_t chunk_size = 4)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int laneId = threadIdx.x % DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;

    extern __shared__ int32_t shared_mem[];
    int32_t *shm_mask = shared_mem, *shm_bitval = &shared_mem[feature_len];
    int32_t *shm_workspace;
    if(shmem_size >= feature_len * sizeof(int32_t) * 2) {
        shm_workspace = &shared_mem[2 * feature_len + threadIdx.x];
        for(int i = threadIdx.x; i < feature_len; i += blockDim.x) {
            shm_mask[i] = dev_mask[i];
            shm_bitval[i] = dev_bitval[i];
        }
    } else {
        shm_mask = dev_mask;
        shm_bitval = dev_bitval;
        shm_workspace = &shared_mem[threadIdx.x];
    }
    __syncthreads();
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
            ull ptr = cache_vals[index];
            int64_t compressed = 0;
            int64_t gpu_id = 0;
            ull byte_offset = deconstruct_hash_ptr(ptr, gpu_id, compressed);

            char *cache = (char*)dev_cache[gpu_id];
            int32_t *dev_features  = (int32_t*)((char *)cache + byte_offset);
            int32_t *host_features = (int32_t*)((float *)cpu_features + nodeId * feature_len);
            int match = 1;
            if(!compressed) {
                for(int j = laneId; j < feature_len; j += DWARP_SIZE) {
                    if(dev_features[j] != host_features[j]) {
                        match = 0;
                        break;
                    }
                }
            } else {
                //if(laneId == 0)
                //    printf("%p: Ptr %lx, val %lx\n", dev_features, byte_offset, ptr);
                int64_t bitmask_offset = BITS_TO_BYTES((feature_len * sizeof(float) + chunk_size - 1) / chunk_size);
                // 4-byte align
                bitmask_offset = (bitmask_offset + 3) / 4 * 4;
                // Code to decompress
                int32_t bitshift = 0;
                for(int i = laneId; i < (feature_len + DWARP_SIZE - 1) / DWARP_SIZE * DWARP_SIZE; i += DWARP_SIZE) {
                    int32_t cur_bitshift = 0;
                    bool compressed_feat = false;
                    if(i < feature_len) {
                        // Check if this feature is in compressed or uncompressed format
                        int32_t word_offset = i / 32;
                        int32_t bit_offset =  i % 32;
                        compressed_feat = (dev_features[word_offset] & (1 << bit_offset));
                        // Default 32 bits per thread, less if compressed
                        cur_bitshift = 32;
                        if(compressed_feat) {
                            cur_bitshift -= __popc(shm_mask[i]);
                        }
                    }
                    // Perform scan to obtain starting bit for this thread
                    bitshift += warpExclusiveScanSync(FULL_MASK, cur_bitshift);
                    //printf("BS2 %p %d: %d %d\n", dev_features, i, cur_bitshift, bitshift);
                    // Now decompress
                    if(i < feature_len) {
                        int32_t num_bits = cur_bitshift;
                        int32_t temp_read_size;
                        int32_t local_mask = shm_mask[i];
                        // Start from bitshift and insert current compressed feature
                        if(compressed_feat) {
                            *shm_workspace = shm_bitval[i];
                            temp_read_size = min(__clz(local_mask), num_bits);
                        } else {
                            *shm_workspace = 0;
                            temp_read_size = num_bits;
                        }

                        int32_t fin_read_bits = 0;
                        // Number of compressed bits being inserted
                        while(num_bits > 0) {
                            // Perform actual read
                            int32_t word_offset = (bitmask_offset + bitshift / 8) / 4;
                            int32_t bit_offset = (bitmask_offset * 8 + bitshift) % 32;
                            int32_t read_bits = min(temp_read_size, 32 - bit_offset);
                            *shm_workspace |= ((dev_features[word_offset] << bit_offset) & (((1L << read_bits) - 1L) << (32 - read_bits))) >> fin_read_bits;
                            //printf("CURREAD %p %d: %x (%d) (%d, %d)\n", dev_features, i, *shm_workspace, read_bits, bit_offset, word_offset);
                            fin_read_bits += read_bits;
                            num_bits -= read_bits;
                            bitshift += read_bits;
                            // If the feature was compressed, do some bitshifts before next iteration
                            if(compressed_feat) {
                                // Shift by inserted bits
                                local_mask <<= read_bits;
                                // Shift by masked bits
                                int32_t shift = __clz(~local_mask);
                                local_mask <<= shift;
                                fin_read_bits += shift;
                                // Get new insert size
                                temp_read_size = min(__clz(local_mask), num_bits);
                            } else
                                temp_read_size = num_bits;

                        }
                        if(*shm_workspace != host_features[i]) {
                            printf("%p %d: Expected %x (%p), got %x; comp? %d\n", 
                                dev_features, i, host_features[i], host_features + i, *shm_workspace, compressed_feat);
                            match = 0;
                        }
                    }
                    //printf("PREBS2 %p %d: %d %d\n", dev_features, i, cur_bitshift, bitshift);
                    // Get starting bitshift from previous iteration
                    bitshift = __shfl_sync(FULL_MASK, bitshift, DWARP_SIZE - 1);
                    //printf("FINBS2 %p %d: %d %d\n", dev_features, i, cur_bitshift, bitshift);
                }
            }
            int failed = __ballot_sync(FULL_MASK, match == 0);
            if(!failed && success_lookups != nullptr && laneId == 0) {
                atomicAdd(success_lookups, 1);
            }
        }
    }
}

__global__ void compress_cpu_transfer_kernel(void **dev_cache, int64_t *cache_index, 
    int64_t num_nodes, int64_t feature_len, float *output_features, int32_t *bitmask,
    float *cpu_features, int32_t *node_arr, int32_t *dev_mask, int32_t *dev_bitval, 
    int32_t total_nodes, int num_ways, int32_t num_sets, int shmem,
    ull *misses, ull *lookups, ull *inserts) {
    
    extern __shared__ int32_t shared_mem[];
    int32_t *shm_mask = shared_mem, *shm_bitval = &shared_mem[feature_len];
    if(shmem >= feature_len * sizeof(int32_t) * 2) {
        for(int i = threadIdx.x; i < feature_len; i += blockDim.x) {
            shm_mask[i] = dev_mask[i];
            shm_bitval[i] = dev_bitval[i];
        }
    } else {
        shm_mask = dev_mask;
        shm_bitval = dev_bitval;
    }
    __syncthreads();

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int laneId = threadIdx.x % DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    // Go through node list
    for(int i = warpId; i < num_nodes; i += numWarps) {
        __syncwarp();
        int64_t index = cache_index[i];
        int32_t nodeId = node_arr[i];
#ifdef MONITOR_DEEP
        if(laneId == 0)
            atomicAdd(lookups, 1);
#endif
        // Not in cache, manual copy
        if(index < 0) {
#ifdef MONITOR_DEEP
            if(laneId == 0)
                atomicAdd(misses, 1);
#endif
            if(bitmask[i / 32] & (1 << (i % 32))) {
                decompress_and_write((int32_t*)&output_features[i * feature_len], 
                    (int32_t*)&cpu_features[nodeId * feature_len], 
                    shm_mask, shm_bitval, feature_len);
            } else {
                memcpy_warp(&output_features[i * feature_len], 
                    &cpu_features[nodeId * feature_len], feature_len);
            }
        } else {
            int64_t gpu_id, compressed;
            int64_t offset = deconstruct_hash_ptr(index, gpu_id, compressed);
            if(!compressed){
                memcpy_warp(&output_features[i * feature_len], 
                    &((char*)dev_cache[gpu_id])[offset], feature_len);
            } else {
                decompress_and_write((int32_t*)&output_features[i * feature_len], 
                    (int32_t*)&((char*)dev_cache[gpu_id])[offset], 
                    shm_mask, shm_bitval, feature_len);
            }
        }
    }
}

__global__ void compress_cpu_transfer_kernel2(void **dev_cache, int64_t *cache_index, 
    int64_t num_nodes, int64_t feature_len, float *output_features, int32_t *bitmask,
    float *cpu_features, int32_t *node_arr, int32_t *dev_mask, int32_t *dev_bitval, 
    int32_t total_nodes, int num_ways, int32_t num_sets, int shmem,
    ull *misses, ull *lookups, ull *inserts) {
    
    extern __shared__ int32_t shared_mem[];
    int32_t *shm_mask = shared_mem, *shm_bitval = &shared_mem[feature_len];
    int32_t *workspace;
    if(shmem >= feature_len * sizeof(int32_t) * 2) {
        for(int i = threadIdx.x; i < feature_len; i += blockDim.x) {
            shm_mask[i] = dev_mask[i];
            shm_bitval[i] = dev_bitval[i];
        }
        // 32 elements for metadata, 64 elements for working data
        // = 96 elements per warp
        workspace = &shared_mem[2 * feature_len + (threadIdx.x / DWARP_SIZE) * 96];
    } else {
        shm_mask = dev_mask;
        shm_bitval = dev_bitval;
        // 32 elements for metadata, 64 elements for working data
        // = 96 elements per warp
        workspace = &shared_mem[(threadIdx.x / DWARP_SIZE) * 96];
    }
    __syncthreads();

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int laneId = threadIdx.x % DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    // Go through node list
    for(int i = warpId; i < num_nodes; i += numWarps) {
        __syncwarp();
        int64_t index = cache_index[i];
        int32_t nodeId = node_arr[i];
#ifdef MONITOR_DEEP
        if(laneId == 0)
            atomicAdd(lookups, 1);
#endif
        // Not in cache, manual copy
        if(index < 0) {
#ifdef MONITOR_DEEP
            if(laneId == 0)
                atomicAdd(misses, 1);
#endif
            if(bitmask[i / 32] & (1 << (i % 32))) {
                decompress_and_write_cpu((int32_t*)&output_features[i * feature_len], 
                    (int32_t*)&cpu_features[nodeId * feature_len], 
                    shm_mask, shm_bitval, feature_len, workspace);
            } else {
                memcpy_warp(&output_features[i * feature_len], 
                    &cpu_features[nodeId * feature_len], feature_len);
            }
        } else {
            int64_t gpu_id, compressed;
            int64_t offset = deconstruct_hash_ptr(index, gpu_id, compressed);
            if(!compressed){
                memcpy_warp(&output_features[i * feature_len], 
                    &((char*)dev_cache[gpu_id])[offset], feature_len);
            } else {
                decompress_and_write((int32_t*)&output_features[i * feature_len], 
                    (int32_t*)&((char*)dev_cache[gpu_id])[offset], 
                    shm_mask, shm_bitval, feature_len);
            }
        }
    }
}

__global__ void compress_transfer_kernel(void **dev_cache, int64_t *cache_index, 
    int64_t num_nodes, int64_t feature_len, float *output_features, 
    float *cpu_features, int32_t *node_arr, int32_t *dev_mask, int32_t *dev_bitval, 
    int32_t total_nodes, int num_ways, int32_t num_sets, int shmem_size,
    ull *misses, ull *lookups, ull *inserts) {
    
    extern __shared__ int32_t shared_mem[];
    // Move mask to shared memory if we have space. Otherwise use device memory
    int32_t *shm_mask = shared_mem, *shm_bitval = &shared_mem[feature_len];
    if(shmem_size >= feature_len * sizeof(int32_t) * 2) {
        for(int i = threadIdx.x; i < feature_len; i += blockDim.x) {
            shm_mask[i] = dev_mask[i];
            shm_bitval[i] = dev_bitval[i];
        }
    } else {
        shm_mask = dev_mask;
        shm_bitval = dev_bitval;
    }
    __syncthreads();

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int laneId = threadIdx.x % DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    // Go through node list
    for(int i = warpId; i < num_nodes; i += numWarps) {
        __syncwarp();
        int64_t index = cache_index[i];
#ifdef MONITOR_DEEP
        if(laneId == 0)
            atomicAdd(lookups, 1);
#endif
        // Not in cache, manual copy
        if(index < 0) {
#ifdef MONITOR_DEEP
            if(laneId == 0)
                atomicAdd(misses, 1);
#endif
            int32_t nodeId = node_arr[i];
            // If we have extra space on both sides, we can use optimized padded copy
            if(nodeId * feature_len >= GPU_CL_FLOATS && 
                (nodeId + 1) * feature_len + GPU_CL_FLOATS < total_nodes * feature_len)
                memcpy_warp_isspace(&output_features[i * feature_len], 
                                    &cpu_features[nodeId * feature_len], feature_len);
            else
                memcpy_warp(&output_features[i * feature_len], 
                            &cpu_features[nodeId * feature_len], feature_len);
        } else {
            int64_t gpu_id, compressed;
            int64_t offset = deconstruct_hash_ptr(index, gpu_id, compressed);
            if(!compressed){
                //memcpy_warp(&output_features[i * feature_len], 
                //            &cpu_features[nodeId * feature_len], feature_len);
                memcpy_warp(&output_features[i * feature_len], 
                    &((char*)dev_cache[gpu_id])[offset], feature_len);
            } else {

//                memcpy_warp(&output_features[i * feature_len], 
//                            &cpu_features[nodeId * feature_len], feature_len);
                decompress_and_write((int32_t*)&output_features[i * feature_len], 
                    (int32_t*)&((char*)dev_cache[gpu_id])[offset], 
                    shm_mask, shm_bitval, feature_len);
            }
            /*__syncwarp();
            if(i < 100)
            for(int j = laneId; j < feature_len; j += DWARP_SIZE) {
                if(output_features[i * feature_len + j] != cpu_features[nodeId * feature_len + j]) {
                    if(index == 0) {
                        printf("%d: Mismatch node %d ind %lx %x %x\n", i, nodeId, index, 
                            ((int32_t*)output_features)[i * feature_len + j], 
                            ((int32_t*)cpu_features)[nodeId * feature_len + j]);
                    }
                    return;
                }
            }*/
        }
    }
}

#endif //COMPCACHE_H