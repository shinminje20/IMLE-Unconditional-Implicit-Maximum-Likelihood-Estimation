#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

#SBATCH --array=0-59%3
#SBATCH --time=0-11:59:59       # Time: D-H:M:S
#SBATCH --account=rrg-keli      # Account: def-keli/rrg-keli
#SBATCH --mem=32G               # Memory in total
#SBATCH --nodes=1               # Number of nodes requested.
#SBATCH --cpus-per-task=10      # Number of cores per task.
#SBATCH --gres=gpu:a100:2       # 40G A100

# Set the job name, and a 
#SBATCH --job-name=miniImageent_generation_2x
#SBATCH --output=job_results/%x_%A_%a.txt

# Below sets the email notification, swap to your email to receive notifications
#SBATCH --mail-user=tristanengst@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Start building the job's output file. First, make sure the directory to put it
# in exists, and then print some context info to the output file.
mkdir job_results
pwd
hostname
date
echo "Starting job number $SLURM_ARRAY_TASK_ID"

# Start up the right environment to run the code. This entails starting the
# useful bash environment and activating a conda environment.
source ~/.bashrc
conda activate py39ISICLE

# Python will buffer output of your script unless you set this.
# If you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed.
export PYTHONUNBUFFERED=1

# Do all the research.

# Keep jobs from starting at the same time. This wouldn't be needed if we could just have normal WandB. asdfghjk.
sleep $((SLURM_ARRAY_TASK_ID % 3 * 60 + 1))

if [[ $((SLURM_ARRAY_TASK_ID % 3)) == 0 ]]
then
    python TrainGeneratorWandB.py --lr 1e-3 --epochs 20 --data_path ~/scratch/ISICLE/data --bs 8 --res 32 64 128 256 --ns 128 128 128 --sp 128 64 16 --data miniImagenet_deci --gpus 0 1 --grayscale .5 --suffix large_model --resid_nc 256 128 128 128 --dense_nc 512 384 256 128 --wandb offline --chunk_epochs 1 --resume $((SLURM_ARRAY_TASK_ID / 3)) --seed $SLURM_ARRAY_TASK_ID
elif [[ $((SLURM_ARRAY_TASK_ID % 3)) == 1 ]]
then
    python TrainGeneratorWandB.py --lr 5e-4 --epochs 20 --data_path ~/scratch/ISICLE/data --bs 8 --res 32 64 128 256 --ns 128 128 128 --sp 128 64 16 --data miniImagenet_deci --gpus 0 1 --grayscale .5 --suffix large_model --resid_nc 256 128 128 128 --dense_nc 512 384 256 128 --wandb offline --chunk_epochs 1 --resume $((SLURM_ARRAY_TASK_ID / 3)) --seed $SLURM_ARRAY_TASK_ID
elif [[ $((SLURM_ARRAY_TASK_ID % 3)) == 2 ]]
then
    python TrainGeneratorWandB.py --lr 1e-4 --epochs 20 --data_path ~/scratch/ISICLE/data --bs 8 --res 32 64 128 256 --ns 128 128 128 --sp 128 64 16 --data miniImagenet_deci --gpus 0 1 --grayscale .5 --suffix large_model --resid_nc 256 128 128 128 --dense_nc 512 384 256 128 --wandb offline --chunk_epochs 1 --resume $((SLURM_ARRAY_TASK_ID / 3)) --seed $SLURM_ARRAY_TASK_ID
else
    echo "No case here"
fi

# Print completion time.
date
