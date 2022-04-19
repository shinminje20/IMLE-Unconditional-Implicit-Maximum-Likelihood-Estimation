#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

#SBATCH --array=1-3
#SBATCH --time=6-23:59:59       # Time: D-H:M:S
#SBATCH --account=rrg-keli      # Account: def-keli/rrg-keli
#SBATCH --mem=32G               # Memory in total
#SBATCH --nodes=1               # Number of nodes requested.
#SBATCH --cpus-per-task=10      # Number of cores per task.
#SBATCH --gres=gpu:a100:2       # 40G A100

# Uncomment to control the output files. By default stdout and stderr go to
# the same place, but if you use both commands below they'll be split up.
# %N is the hostname (if used, will create output(s) per node).
# %j is jobid.

#SBATCH --output=job_results/miniImagenet_generation_%A_%a.txt

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

echo "Starting job number $SLURM_ARRAY_TASK_ID"

source ~/.bashrc
conda activate py39ISICLE

# Python will buffer output of your script unless you set this.
# If you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed.
export PYTHONUNBUFFERED=1

# Do all the research.
if [ "$SLURM_ARRAY_TASK_ID" = 1 ]
then
    python TrainDPGenerator.py --epochs 20 --data_path ~/scratch/ISICLE/data --bs 4 --res 64 64 64 64 128 --ns 128 128 128 128 --sp 128 128 128 64 --data miniImagenet_deci --gpus 0 1 --grayscale 1 --mask_frac 0 --suffix large_model --resid_nc 256 128 128 128 --dense_nc 512 384 256 128  --wandb 2
elif [ "$SLURM_ARRAY_TASK_ID" = 2 ]
then
    python TrainDPGenerator.py --epochs 20 --data_path ~/scratch/ISICLE/data --bs 4 --res 64 64 64 64 128 --ns 128 128 128 128 --sp 128 128 128 64 --data miniImagenet_deci --gpus 0 1 --grayscale 1 --mask_frac .5 --suffix large_model --resid_nc 256 128 128 128 --dense_nc 512 384 256 128  --wandb 2
elif [ "$SLURM_ARRAY_TASK_ID" = 3 ]
then
    python TrainDPGenerator.py --epochs 20 --data_path ~/scratch/ISICLE/data --bs 4 --res 64 64 64 64 128 --ns 128 128 128 128 --sp 128 128 128 64 --data miniImagenet_deci --gpus 0 1 --grayscale 1 --mask_frac .25 --mask_res 6 --suffix large_model --resid_nc 256 128 128 128 --dense_nc 512 384 256 128  --wandb 2
else
    echo "No case here"
fi

# Print completion time.
date
