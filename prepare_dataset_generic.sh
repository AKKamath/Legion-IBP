DATASET=$1
cd dataset/
mkdir -p ${DATASET}
yes y | python prepare_${DATASET}.py 
cd .. 
