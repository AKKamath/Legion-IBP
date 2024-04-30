RESULTS=./results
DATASET_PATH=dataset

init:
	mkdir -p ${RESULTS}

clean:
	rm -f /dev/shm/sem.sem*
	pkill legion*  || true
	-pkill *sampling_server
	sleep 5

DATASET ?=paper100m
BATCH_SIZE ?=8192
FEAT_NUM ?=128
CLASS_NUM ?=172
EPOCHS ?=10
CACHE_SIZE ?=38000000
DYN_CACHE ?=0
OTHER_OPTS?=
NUM_GPUS?=2
PREFIX?=
graphsage:
	stdbuf -oL python legion_server.py --dataset_path '${DATASET_PATH}' --dataset_name ${DATASET} \
		--train_batch_size ${BATCH_SIZE} --fanout [25,10] --gpu_number ${NUM_GPUS} --epoch ${EPOCHS} \
		--cache_memory ${CACHE_SIZE} --dyn_cache ${DYN_CACHE} ${OTHER_OPTS} > ${RESULTS}/${DATASET}${PREFIX}_sampling.log & \
		bash ./scripts/wait_file.sh ${RESULTS}/${DATASET}${PREFIX}_sampling.log;

	CUDA_VISIBLE_DEVICES=0,1 stdbuf -oL python training_backend/legion_graphsage.py --class_num ${CLASS_NUM} \
		--features_num ${FEAT_NUM} --hidden_dim 256 --hops_num 2 --gpu_number ${NUM_GPUS} \
		--epoch ${EPOCHS} > ${RESULTS}/${DATASET}${PREFIX}_training.log

# Command to run with dynamic cache
%_dyn:
	$(MAKE) $* DYN_CACHE=1 PREFIX=_cached${PREFIX}

# Command to run with dynamic cache
%_shadow:
	$(MAKE) $* DYN_CACHE=3 PREFIX=_shadowcached${PREFIX}

# Command to run with single GPU
%_singlegpu:
	$(MAKE) $* OTHER_OPTS="--usenvlink=0" NUM_GPUS=1 PREFIX=_singlegpu${PREFIX}

# Individual experiment commands
run_papers100m: init clean
	$(MAKE) graphsage DATASET=paper100m FEAT_NUM=128 CLASS_NUM=172

run_products: init clean
	$(MAKE) graphsage DATASET=products FEAT_NUM=100 CLASS_NUM=47

run_ukunion: init clean
	$(MAKE) graphsage DATASET=ukunion FEAT_NUM=116 CLASS_NUM=2

run_cora: init clean
	$(MAKE) graphsage DATASET=cora FEAT_NUM=8710 CLASS_NUM=70 BATCH_SIZE=1024 EPOCHS=100

run_reddit: init clean
	$(MAKE) graphsage DATASET=reddit FEAT_NUM=602 BATCH_SIZE=1024 CLASS_NUM=50

run_pubmed: init clean
	$(MAKE) graphsage DATASET=pubmed FEAT_NUM=500 BATCH_SIZE=1024 CLASS_NUM=3

