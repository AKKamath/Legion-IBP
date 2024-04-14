RESULTS=./results
DATASET_PATH=dataset
init:
	mkdir -p ${RESULTS}

clean:
	rm -f /dev/shm/sem.sem*

DATASET ?= paper100m
BATCH_SIZE ?= 8000
FEAT_NUM ?= 128
expt:
	stdbuf -oL python legion_server.py --dataset_path '${DATASET_PATH}' --dataset_name ${DATASET} --train_batch_size ${BATCH_SIZE} --fanout [25,10] --gpu_number 2 --epoch 1 --cache_memory 3800000000 > ${RESULTS}/${DATASET}_server.log & \
	bash wait_file.sh ${RESULTS}/${DATASET}_server.log;
	stdbuf -oL python training_backend/legion_graphsage.py --class_num 172  --features_num ${FEAT_NUM} --hidden_dim 256 --hops_num 2 --gpu_number 2 --epoch 1 > ${RESULTS}/${DATASET}_training.log
	pkill -f legion_server.py

expt_dyn_cache:
	stdbuf -oL python legion_server.py --dataset_path '${DATASET_PATH}' --dataset_name ${DATASET} --train_batch_size ${BATCH_SIZE} --fanout [25,10] --gpu_number 2 --epoch 1 --cache_memory 3800000000 --dyn_cache=1 > ${RESULTS}/${DATASET}_cached_server.log
#	bash wait_file.sh ${RESULTS}/${DATASET}_cached_server.log;
#	stdbuf -oL python training_backend/legion_graphsage.py --class_num 172  --features_num ${FEAT_NUM} --hidden_dim 256 --hops_num 2 --gpu_number 2 --epoch 1 > ${RESULTS}/${DATASET}_cached_training.log
#	pkill -f legion_server.py

run_papers100m: init clean
	$(MAKE) expt DATASET=paper100m BATCH_SIZE=8000 FEAT_NUM=128

run_papers100m_dyn: init clean
	$(MAKE) expt_dyn_cache DATASET=paper100m BATCH_SIZE=8000 FEAT_NUM=128

run_products: init clean
	$(MAKE) expt DATASET=products BATCH_SIZE=8000 FEAT_NUM=100

run_products_dyn: init clean
	$(MAKE) expt_dyn_cache DATASET=products BATCH_SIZE=8000 FEAT_NUM=100
