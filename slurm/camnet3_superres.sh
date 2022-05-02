#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

#SBATCH --array=0-139%7
#SBATCH --time=0-11:59:59       # Time: D-H:M:S
#SBATCH --account=rrg-keli      # Account: def-keli/rrg-keli
#SBATCH --mem=32G               # Memory in total
#SBATCH --nodes=1               # Number of nodes requested.
#SBATCH --cpus-per-task=10      # Number of cores per task.
#SBATCH --gres=gpu:a100:2       # 40G A100

# Uncomment to control the output files. By default stdout and stderr go to
# the same place, but if you use both commands below they'll be split up.
# %N is the hostname (if used, will create output(s) per node).
# %j is jobid.

#SBATCH --output=job_results/camnet3_superresolution_%A_%a.txt

# Below sets the email notification, swap to your email to receive notifications
#SBATCH --mail-user=tristanengst@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Print some info for context.
pwd
hostname
date

module load httpproxy

echo "Starting job number $SLURM_ARRAY_TASK_ID"

source ~/.bashrc
conda activate py39ISICLE

# Python will buffer output of your script unless you set this.
# If you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed.
export PYTHONUNBUFFERED=1

# Do all the research.
if [[ $((SLURM_ARRAY_TASK_ID % 3)) == 0 ]]
then
    python TrainGeneratorWandB.py --lr 5e-4 --epochs 20 --data_path ~/scratch/ISICLE/data --bs 16 --res 16 32 64 128 256 --ns 128 128 128 128 --sp 128 128 64 16 --data camnet3 --normalize 1 --gpus 0 1 --wandb offline --chunk_epochs 1 --resume $((SLURM_ARRAY_TASK_ID / 7)) --seed $SLURM_ARRAY_TASK_ID --warmup 1024
elif [[ $((SLURM_ARRAY_TASK_ID % 3)) == 1 ]]
then
    python TrainGeneratorWandB.py --lr 5e-4 --epochs 20 --data_path ~/scratch/ISICLE/data --bs 16 --res 16 32 64 128 256 --ns 128 128 128 128 --sp 128 128 64 16 --data camnet3 --normalize 1  --gpus 0 1 --wandb offline --chunk_epochs 1 --resume $((SLURM_ARRAY_TASK_ID / 7)) --seed $SLURM_ARRAY_TASK_ID --warmup 1024 --grayscale .5
elif [[ $((SLURM_ARRAY_TASK_ID % 3)) == 2 ]]
then
    python TrainGeneratorWandB.py --lr 5e-4 --epochs 20 --data_path ~/scratch/ISICLE/data --bs 16 --res 16 32 64 128 256 --ns 128 128 128 128 --sp 128 128 64 16 --data camnet3 --gpus 0 1 --wandb offline --chunk_epochs 1 --resume $((SLURM_ARRAY_TASK_ID / 7)) --seed $SLURM_ARRAY_TASK_ID --warmup 1024
elif [[ $((SLURM_ARRAY_TASK_ID % 3)) == 3 ]]
then
    python TrainGeneratorWandB.py --lr 5e-4 --epochs 20 --data_path ~/scratch/ISICLE/data --bs 16 --res 16 32 64 128 256 --ns 128 128 128 128 --sp 128 128 64 16 --data camnet3 --gpus 0 1 --wandb offline --chunk_epochs 1 --resume $((SLURM_ARRAY_TASK_ID / 7)) --seed $SLURM_ARRAY_TASK_ID --warmup 1024 --grayscale .5
elif [[ $((SLURM_ARRAY_TASK_ID % 3)) == 4 ]]
then
    python TrainGeneratorWandB.py --lr 1e-4 --epochs 20 --data_path ~/scratch/ISICLE/data --bs 16 --res 16 32 64 128 256 --ns 128 128 128 128 --sp 128 128 64 16 --data camnet3 --gpus 0 1 --wandb offline --chunk_epochs 1 --resume $((SLURM_ARRAY_TASK_ID / 7)) --seed $SLURM_ARRAY_TASK_ID --warmup 1024
elif [[ $((SLURM_ARRAY_TASK_ID % 3)) == 5 ]]
then
    python TrainGeneratorWandB.py --lr 5e-5 --epochs 20 --data_path ~/scratch/ISICLE/data --bs 16 --res 16 32 64 128 256 --ns 128 128 128 128 --sp 128 128 64 16 --data camnet3 --gpus 0 1 --wandb offline --chunk_epochs 1 --resume $((SLURM_ARRAY_TASK_ID / 7)) --seed $SLURM_ARRAY_TASK_ID --warmup 1024
elif [[ $((SLURM_ARRAY_TASK_ID % 3)) == 6 ]]
then
    python TrainGeneratorWandB.py --lr 1e-5 --epochs 20 --data_path ~/scratch/ISICLE/data --bs 16 --res 16 32 64 128 256 --ns 128 128 128 128 --sp 128 128 64 16 --data camnet3 --gpus 0 1 --wandb offline --chunk_epochs 1 --resume $((SLURM_ARRAY_TASK_ID / 7)) --seed $SLURM_ARRAY_TASK_ID --warmup 1024
else
    echo "No case here"
fi

# Print completion time.
date
