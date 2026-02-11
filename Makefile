RESULTS ?= ./results
DATASET_PATH=dataset

init:
	mkdir -p ${RESULTS}
	mkdir -p ${RESULTS}/${DATASET}

clean:
	rm -f /dev/shm/sem.sem*
	pkill legion*  || true
	-pkill *sampling_server
	sleep 5

DATASET ?=reddit
BATCH_SIZE ?=8192
FEAT_NUM ?=602
CLASS_NUM ?=50
EPOCHS ?=10
CACHE_SIZE ?=38000000
DYN_CACHE ?=0
OTHER_OPTS?=
NUM_GPUS?=2
POSTFIX?=
FP16 ?= False
COMPRESS ?= 0

dgl_graphsage: init clean
	CUDA_VISIBLE_DEVICES=0,1 stdbuf -oL python training_backend/dgl_graphsage.py \
		--dataset_path '${DATASET_PATH}' --dataset_name ${DATASET} --class_num ${CLASS_NUM} \
		--train_batch_size ${BATCH_SIZE} --cache_memory ${CACHE_SIZE} \
		--features_num ${FEAT_NUM} --hidden_dim 256 --hops_num 2 --gpu_number ${NUM_GPUS} --float16 ${FP16} \
		--epoch ${EPOCHS} --compress ${COMPRESS} > ${RESULTS}/${DATASET}/training_dgl${POSTFIX}.log

graphsage:
	$(MAKE) sampling & \
	bash ./scripts/wait_file.sh ${RESULTS}/${DATASET}/sampling${POSTFIX}.log;

	stdbuf -oL python training_backend/legion_graphsage.py --class_num ${CLASS_NUM} \
		--features_num ${FEAT_NUM} --hidden_dim 256 --hops_num 2 --gpu_number ${NUM_GPUS} --float16 ${FP16} \
		--epoch ${EPOCHS} > ${RESULTS}/${DATASET}/training${POSTFIX}.log

sampling: init clean
	stdbuf -oL python legion_server.py --dataset_path '${DATASET_PATH}' \
		--dataset_name ${DATASET} --train_batch_size ${BATCH_SIZE} \
		--gpu_number ${NUM_GPUS} --epoch ${EPOCHS} --cache_memory ${CACHE_SIZE} \
		--dyn_cache ${DYN_CACHE} ${OTHER_OPTS} > ${RESULTS}/${DATASET}/sampling${POSTFIX}.log

extract_%:
	python scripts/extract_results.py ${RESULTS} $* "dgl dglcomp base comp cpuonly cpuasync" > ${RESULTS}/$*.log

extract_expts:
	python scripts/extract_results.py ${RESULTS} "cora_ls pubmed_ls citeseer_ls reddit products mag" "dgl dglcomp base comp cpuonly cpuasync"

extract_expts_single:
	python scripts/extract_results.py ${RESULTS} "cora_ls pubmed_ls citeseer_ls reddit products mag" \
		"dgl_singlegpu dglcomp_singlegpu base_singlegpu comp_singlegpu cpuonly_singlegpu cpuasync_singlegpu" \
		"DGL DGL+IBP(M) Legion Legion+IBP(C) Legion+IBP(M) Legion+IBP(C/M)"

extract_comptest:
	python scripts/extract_compression.py ${RESULTS} "pubmed citeseer cora reddit products mag paper100m"

run_expts:
	$(MAKE) run_reddit_all
	$(MAKE) run_products_all
	$(MAKE) run_pubmed_ls_all
	$(MAKE) run_citeseer_ls_all
	$(MAKE) run_cora_ls_all
	$(MAKE) run_mag_all

run_comptests:
	$(MAKE) run_reddit_comptest
	$(MAKE) run_products_comptest
	$(MAKE) run_cora_comptest
	$(MAKE) run_pubmed_comptest
	$(MAKE) run_citeseer_comptest
	$(MAKE) run_paper100m_comptest
	$(MAKE) run_mag_comptest

run_%_all:
	$(MAKE) run_$*_dgl
	$(MAKE) run_$*_dglcomp
	$(MAKE) run_$*_base
	$(MAKE) run_$*_comp
	#$(MAKE) run_$*_cputest
	$(MAKE) run_$*_cpuonly
	$(MAKE) run_$*_cpuasync
	#$(MAKE) run_$*_mod
	#$(MAKE) run_$*_opt
	#$(MAKE) run_$*_unopt
	$(MAKE) extract_$*

run_cpuasync_singlegpu:
	$(MAKE) run_reddit_cpuasync_singlegpu
	$(MAKE) run_products_cpuasync_singlegpu
	$(MAKE) run_cora_ls_cpuasync_singlegpu
	$(MAKE) run_pubmed_ls_cpuasync_singlegpu
	$(MAKE) run_citeseer_ls_cpuasync_singlegpu
	$(MAKE) run_paper100m_cpuasync_singlegpu
	$(MAKE) run_mag_cpuasync_singlegpu

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

%_asynccomp:
	$(MAKE) $* DYN_CACHE=520 POSTFIX=_asynccomp${POSTFIX}

#%_compcpu:
#	$(MAKE) $* DYN_CACHE=24 POSTFIX=_compcpu${POSTFIX}

%_base:
	$(MAKE) $* DYN_CACHE=0 POSTFIX=_base${POSTFIX}

%_opt:
	$(MAKE) $* DYN_CACHE=32 POSTFIX=_opt${POSTFIX}

%_unopt:
	$(MAKE) $* DYN_CACHE=64 POSTFIX=_unopt${POSTFIX}

%_comptest:
	$(MAKE) $* EXPT=sampling DYN_CACHE=128 POSTFIX=_comptest${POSTFIX} CACHE_SIZE=38000000 OTHER_OPTS="--usenvlink=0" NUM_GPUS=1

%_cputest:
	$(MAKE) $* DYN_CACHE=280 POSTFIX=_cputest${POSTFIX}

%_cpuasync:
	$(MAKE) $* DYN_CACHE=792 POSTFIX=_cpuasync${POSTFIX}

%_cpuonly:
	$(MAKE) $* DYN_CACHE=272 POSTFIX=_cpuonly${POSTFIX}

%_cpuonlyasync:
	$(MAKE) $* DYN_CACHE=784 POSTFIX=_cpuonlyasync${POSTFIX}

%_dgl:
	$(MAKE) $* EXPT=dgl_graphsage

%_dglcomp:
	$(MAKE) $* EXPT=dgl_graphsage COMPRESS=1 POSTFIX=comp${POSTFIX}

EXPT=graphsage
# Individual experiment commands
run_paper100m:
	$(MAKE) ${EXPT} DATASET=paper100m FEAT_NUM=128 CLASS_NUM=172 CACHE_SIZE=568626975

run_products:
	$(MAKE) ${EXPT} DATASET=products FEAT_NUM=100 CLASS_NUM=47 CACHE_SIZE=9796116

run_ukunion:
	$(MAKE) ${EXPT} DATASET=ukunion FEAT_NUM=116 CLASS_NUM=2

run_cora:
	$(MAKE) ${EXPT} DATASET=cora FEAT_NUM=8710 CLASS_NUM=70 BATCH_SIZE=512 EPOCHS=100 CACHE_SIZE=4000000

run_reddit:
	$(MAKE) ${EXPT} DATASET=reddit FEAT_NUM=602 BATCH_SIZE=1024 CLASS_NUM=50 CACHE_SIZE=5609797

run_pubmed:
	$(MAKE) ${EXPT} DATASET=pubmed FEAT_NUM=500 BATCH_SIZE=2048 CLASS_NUM=3 CACHE_SIZE=200000 EPOCHS=500

run_mag:
	$(MAKE) ${EXPT} DATASET=mag FEAT_NUM=384 CLASS_NUM=153 FP16=True CACHE_SIZE=1870105590

run_citeseer:
	$(MAKE) ${EXPT} DATASET=citeseer FEAT_NUM=3703 CLASS_NUM=6 BATCH_SIZE=512 EPOCHS=100 CACHE_SIZE=3800000

run_pubmed_ls:
	$(MAKE) ${EXPT} DATASET=pubmed_ls FEAT_NUM=500 CLASS_NUM=3 CACHE_SIZE=48980580

run_citeseer_ls:
	$(MAKE) ${EXPT} DATASET=citeseer_ls FEAT_NUM=3703 CLASS_NUM=6 BATCH_SIZE=1024 CACHE_SIZE=362750175

run_cora_ls:
	$(MAKE) ${EXPT} DATASET=cora_ls FEAT_NUM=8710 CLASS_NUM=70 BATCH_SIZE=512 CACHE_SIZE=853241704