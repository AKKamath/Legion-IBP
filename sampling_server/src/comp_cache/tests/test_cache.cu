#include "../dyn_cache.cuh"

#include <stdlib.h>
#define TOTAL_NODES 1000
#define HOTSET 130
#define ITERS 100
#define NODES_PER_ITER 128
#define CACHED_NODES 100
#define FEAT_SIZE 32

int main() {
    DynamicCache cache;
    float *cpu_features;
    //float *cpu_features = (float*)malloc(FEAT_SIZE * sizeof(float) * TOTAL_NODES);
    cudaHostAlloc((void**)&cpu_features, FEAT_SIZE * sizeof(float) * TOTAL_NODES, cudaHostAllocDefault);
    for(int i = 0; i < FEAT_SIZE * TOTAL_NODES; ++i) {
        cpu_features[i] = i / 1000.0f;
    }
    //cudaHostRegister(cpu_features, FEAT_SIZE * sizeof(float) * TOTAL_NODES, cudaHostRegisterDefault);
    cudaCheckError();
    int *index_array; // = (int*)malloc(sizeof(int) * TOTAL_NODES);
    cudaHostAlloc((void**)&index_array, sizeof(int) * TOTAL_NODES, cudaHostAllocDefault);
    for(int i = 0; i < TOTAL_NODES; ++i) {
        index_array[i] = i;
    }
    cudaHostRegister(index_array, sizeof(int) * TOTAL_NODES, cudaHostRegisterDefault);
    cudaCheckError();
    cache.init_cache(CACHED_NODES, FEAT_SIZE, cpu_features, index_array, 1, 0, TOTAL_NODES, DYN_FLAGS::DYN_ENABLE);
    cudaCheckError();

    int *hotset = (int*)malloc(sizeof(int) * HOTSET);
    for(int i = 0; i < HOTSET; ++i) {
        hotset[i] = rand() % TOTAL_NODES;
    }

    float *output_buffer;
    cudaMalloc(&output_buffer, sizeof(float) * FEAT_SIZE * NODES_PER_ITER);
    cudaCheckError();
    ull *misses, *accesses, *inserts;
    cudaMalloc(&misses, sizeof(ull));
    cudaMalloc(&accesses, sizeof(ull));
    cudaMalloc(&inserts, sizeof(ull));
    cudaCheckError();

    int *nodeIds = (int*)malloc(sizeof(int) * NODES_PER_ITER);
    cudaHostRegister(nodeIds, sizeof(int) * NODES_PER_ITER, cudaHostRegisterDefault);
    for(int iter = 0; iter < ITERS; ++iter) {
        for(int i = 0; i < NODES_PER_ITER; ++i) {
            if(i % 10 < 9)
                nodeIds[i] = hotset[rand() % HOTSET];
            else
                nodeIds[i] = rand() % TOTAL_NODES;
        }
        cudaMemset(misses, 0, sizeof(ull));
        cudaMemset(accesses, 0, sizeof(ull));
        cudaMemset(inserts, 0, sizeof(ull));
        cache.retrieve_and_touch(nodeIds, NODES_PER_ITER, output_buffer, cpu_features, TOTAL_NODES,
                0, misses, accesses, inserts);

        ull host_misses, host_accesses, host_inserts;
        cudaMemcpy(&host_misses, misses, sizeof(ull), cudaMemcpyDeviceToHost);
        cudaMemcpy(&host_accesses, accesses, sizeof(ull), cudaMemcpyDeviceToHost);
        cudaMemcpy(&host_inserts, inserts, sizeof(ull), cudaMemcpyDeviceToHost);
        cudaCheckError();
        printf("Misses: %llu, Accesses: %llu, Inserts: %llu\n", host_misses, host_accesses, host_inserts);
    }
}