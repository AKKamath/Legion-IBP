#!/bin/bash --login
#
#SBATCH --job-name=gnn_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300GB
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=mcnode22

#set -e # stop bash script on first error

#conda activate dgl-dev-gpu-117

export CUDA_MPS_PIPE_DIRECTORY=~/Legion/mps-a100 # Select a location that's accessible to the given $UID
export CUDA_MPS_LOG_DIRECTORY=~/Legion/mps-a100-log # Select a location that's accessible to the given $UID
nvidia-cuda-mps-control -d # Start the daemon.

bash build.sh

pushd sampling_server/src/IBP/
rm -rf build/
pip install -v -e .
popd

# Optionally add "$@" to pass additional arguments to the script
make run_mag_all_singlegpu RESULTS=./results_new3_a100
#make run_comptests RESULTS=./results_new3_a100