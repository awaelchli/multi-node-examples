#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --time=0-02:00:00

# (optional) debugging flags
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# (optional) set the network interface name manually
# export NCCL_SOCKET_IFNAME=lo

# load your Python environment, for example with conda
conda activate your_env_name

# (optional) run script with fake data for debugging
# srun python train.py --trainer.num_nodes 2 --trainer.gpus 8 --data.fake_data true

# run full training
srun python train.py --trainer.num_nodes 2 --trainer.gpus 8