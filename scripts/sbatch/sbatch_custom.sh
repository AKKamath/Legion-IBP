#!/bin/bash --login
#
#SBATCH --job-name=gnn_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300GB
#SBATCH --gres=gpu:a100:1

#set -e # stop bash script on first error

conda activate dgl-dev-gpu-117

export CUDA_MPS_PIPE_DIRECTORY=~/Legion/mps-a100 # Select a location that's accessible to the given $UID
export CUDA_MPS_LOG_DIRECTORY=~/Legion/mps-a100-log # Select a location that's accessible to the given $UID
nvidia-cuda-mps-control -d # Start the daemon.

# Optionally add "$@" to pass additional arguments to the script
make run_mag_dgl_singlegpu RESULTS=./results_a100
make run_mag_dglcomp_singlegpu RESULTS=./results_a100