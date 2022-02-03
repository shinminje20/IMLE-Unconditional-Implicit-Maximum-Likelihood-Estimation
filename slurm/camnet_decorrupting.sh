#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

#SBATCH --array=1-4
#SBATCH --time=3-0:0:0          # Time: D-H:M:S
#SBATCH --account=def-keli      # Account: def-keli/rrg-keli
#SBATCH --mem=50G               # Memory in total
#SBATCH --nodes=1               # Number of nodes requested.
#SBATCH --cpus-per-task=10      # Number of cores per task.
#SBATCH --gres=gpu:v100l:1      # 32G V100

# Uncomment this to have Slurm cd to a directory before running the script.
# You can also just run the script from the directory you want to be in.
# Change the folder below to your code directory
#SBATCH -D /projects/def-keli/tme3/ISICLE

# Uncomment to control the output files. By default stdout and stderr go to
# the same place, but if you use both commands below they'll be split up.
# %N is the hostname (if used, will create output(s) per node).
# %j is jobid.

#SBATCH --output=job_results/%j.txt

# Below sets the email notification, swap to your email to receive notifications
#SBATCH --mail-user=tme28@cornell.edu
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

wandb login 90d0248dd4a9fde052b760cdb131373d49b4dad8




# Do all the research.
if [ "$SLURM_ARRAY_TASK_ID" = 1 ]
then
    python TrainGenerator.py --res 32 64 --data camnet3_deci --bs 64 --code_bs 64 --mini_bs 8 --ipcpe 4 --suffix normal --color_space lab
elif [ "$SLURM_ARRAY_TASK_ID" = 2 ]
then
    python TrainGenerator.py --res 32 64 --data camnet3_deci --bs 64 --code_bs 64 --mini_bs 8 --ipcpe 4 --in_nc 3 --suffix in_nc_3 --color_space lab
elif [ "$SLURM_ARRAY_TASK_ID" = 3 ]
then
    python TrainGenerator.py --res 32 64 --data camnet3_deci --bs 64 --code_bs 64 --mini_bs 8 --ipcpe 4 --epochs 40 --suffix more_eps --color_space lab
elif [ "$SLURM_ARRAY_TASK_ID" = 4 ]
then
    python TrainGenerator.py --res 32 64 --data camnet3_deci --bs 64 --code_bs 64 --mini_bs 8 --ipcpe 4 --epochs 40 --in_nc 3 --suffix more_eps_more_nc --color_space lab
else
    echo "No case here"
fi

# Print completion time.
date
