RESULTS=./results
DATASET_PATH=dataset

init:
	mkdir -p ${RESULTS}
	mkdir -p ${RESULTS}/${DATASET}

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
POSTFIX?=
graphsage: init clean
	stdbuf -oL python legion_server.py --dataset_path '${DATASET_PATH}' --dataset_name ${DATASET} \
		--train_batch_size ${BATCH_SIZE} --fanout [25,10] --gpu_number ${NUM_GPUS} --epoch ${EPOCHS} \
		--cache_memory ${CACHE_SIZE} --dyn_cache ${DYN_CACHE} ${OTHER_OPTS} > ${RESULTS}/${DATASET}/sampling${POSTFIX}.log & \
		bash ./scripts/wait_file.sh ${RESULTS}/${DATASET}/sampling${POSTFIX}.log;

	CUDA_VISIBLE_DEVICES=0,1 stdbuf -oL python training_backend/legion_graphsage.py --class_num ${CLASS_NUM} \
		--features_num ${FEAT_NUM} --hidden_dim 256 --hops_num 2 --gpu_number ${NUM_GPUS} \
		--epoch ${EPOCHS} > ${RESULTS}/${DATASET}/training${POSTFIX}.log

extract_%:
	python scripts/extract_results.py ${RESULTS}/$*

%_all:
	$(MAKE) run_$*_comp
	$(MAKE) run_$*
	$(MAKE) run_$*_mod
	$(MAKE) extract_$*

# Command to run with dynamic cache
%_dyn:
	$(MAKE) $* DYN_CACHE=1 POSTFIX=_cached${POSTFIX}

# Command to run with dynamic cache
%_mod:
	$(MAKE) $* DYN_CACHE=3 POSTFIX=_mod${POSTFIX}

# Command to run with single GPU
%_singlegpu:
	$(MAKE) $* OTHER_OPTS="--usenvlink=0" NUM_GPUS=1 POSTFIX=_singlegpu${POSTFIX}

%_comp:
	$(MAKE) $* DYN_CACHE=8 POSTFIX=_comp${POSTFIX}

# Individual experiment commands
run_paper100m:
	$(MAKE) graphsage DATASET=paper100m FEAT_NUM=128 CLASS_NUM=172

run_products:
	$(MAKE) graphsage DATASET=products FEAT_NUM=100 CLASS_NUM=47

run_ukunion:
	$(MAKE) graphsage DATASET=ukunion FEAT_NUM=116 CLASS_NUM=2

run_cora:
	$(MAKE) graphsage DATASET=cora FEAT_NUM=8710 CLASS_NUM=70 BATCH_SIZE=1024 EPOCHS=100 CACHE_SIZE=4000000

run_reddit:
	$(MAKE) graphsage DATASET=reddit FEAT_NUM=602 BATCH_SIZE=1024 CLASS_NUM=50

run_pubmed:
	$(MAKE) graphsage DATASET=pubmed FEAT_NUM=500 BATCH_SIZE=1024 CLASS_NUM=3 CACHE_SIZE=200000 EPOCHS=500

run_mag:
	$(MAKE) graphsage DATASET=mag FEAT_NUM=368 CLASS_NUM=153

run_citeseer:
	$(MAKE) graphsage DATASET=citeseer FEAT_NUM=3703 CLASS_NUM=6 BATCH_SIZE=512 EPOCHS=100 CACHE_SIZE=3800000
