#ifndef OPERATOR_H
#define OPERATOR_H
#define ull unsigned long long

struct OpParams {
    int device_id;
    cudaStream_t stream;
    cudaEvent_t event;
    void* memorypool;
    void* cache;
    void* graph;
    void* feature;
    void* env;
    int neighbor_count;
    bool is_presc;
    bool in_memory;
    int  hop_num;
#ifdef MONITOR
    ull *dev_ctr;
    ull *dev_hits;
#endif
};

class Operator {
public:
    virtual void run(OpParams* params) = 0;
};

Operator* NewBatchGenerateOP(int op_id);
Operator* NewRandomSampleOP(int op_id);
Operator* NewCacheLookupOP(int op_id);
Operator* NewSSDIOSubmitOP(int op_id);
Operator* NewSSDIOCompleteOP(int op_id);

#endif