#include "../dyn_cache.cuh"

#include <stdlib.h>
#define TOTAL_NODES 100000
#define ITERS 10
#define NODES_PER_ITER 51
#define CACHED_NODES 100
#define FEAT_SIZE 256



int main() {
    DynamicCache cache;
    float *cpu_features = (float*)malloc(FEAT_SIZE * sizeof(float) * TOTAL_NODES);
    /*for(int i = 0; i < FEAT_SIZE * TOTAL_NODES; ++i) {
        cpu_features[i] = i / 1000.0f;
    }*/
    memset(cpu_features, 0, FEAT_SIZE * sizeof(float) * TOTAL_NODES);
    cudaHostRegister(cpu_features, FEAT_SIZE * sizeof(float) * TOTAL_NODES, cudaHostRegisterDefault);
    cudaCheckError();
    int *index_array = (int*)malloc(sizeof(int) * TOTAL_NODES);
    for(int i = 0; i < TOTAL_NODES; ++i) {
        index_array[i] = i;
    }
    cudaHostRegister(index_array, sizeof(int) * TOTAL_NODES, cudaHostRegisterDefault);
    cudaCheckError();
    cache.init_cache(CACHED_NODES, FEAT_SIZE, cpu_features, index_array, 1, 0, TOTAL_NODES, DYN_FLAGS::DYN_ENABLE);
    cudaCheckError();

    float *output_buffer;
    int *cache_index;
    cudaMalloc(&output_buffer, sizeof(float) * FEAT_SIZE * NODES_PER_ITER);
    cudaMalloc(&cache_index, sizeof(int) * NODES_PER_ITER);
    cudaMemset(cache_index, -1, sizeof(int) * NODES_PER_ITER);
    cudaCheckError();
    int *nodeIds = (int*)malloc(sizeof(int) * NODES_PER_ITER);
    cudaHostRegister(nodeIds, sizeof(int) * NODES_PER_ITER, cudaHostRegisterDefault);
    for(int iter = 0; iter < ITERS; ++iter) {
        for(int i = 0; i < NODES_PER_ITER; ++i) {
            nodeIds[i] = rand() % TOTAL_NODES;
        }
        auto start = TIME_NOW;
        printf("%p %p\n", cpu_features, output_buffer);
        cache.transfer(nodeIds, NODES_PER_ITER, output_buffer, cache_index, cpu_features, TOTAL_NODES, 0);
        cudaCheckError();
        cudaDeviceSynchronize();
        auto end = TIME_NOW;
        std::cout << "Transfer time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " ms" << std::endl;
        std::cout << "Bandwidth usage: " << (float)NODES_PER_ITER * FEAT_SIZE * sizeof(float) / 1024 / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1000 << " MB/s" << std::endl;
    }
}