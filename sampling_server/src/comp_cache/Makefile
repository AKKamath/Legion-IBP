TESTS = ./tests

all: ${TESTS}/test_transfer

${TESTS}/test_cache: ${TESTS}/test_cache.cu dyn_cache.cu
	nvcc $(NVCXXFLAGS) -o $@ $^ -lrt -DMONITOR_DEEP

${TESTS}/test_transfer: ${TESTS}/test_transfer.cu dyn_cache.cu
	nvcc $(NVCXXFLAGS) -o $@ $^ -lrt -DMONITOR_DEEP