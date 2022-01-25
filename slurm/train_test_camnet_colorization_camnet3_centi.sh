#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

#SBATCH --time=3-0:0:0  # Time: D-H:M:S
#SBATCH --account={account-name} # Account: def-keli/rrg-keli
#SBATCH --mem=50G           # Memory in total
#SBATCH --nodes=1          # Number of nodes requested.
#SBATCH --cpus-per-task=10  # Number of cores per task.
#SBATCH --gres=gpu:v100l:1 # 32G V100

# Uncomment this to have Slurm cd to a directory before running the script.
# You can also just run the script from the directory you want to be in.
# Change the folder below to your code directory
#SBATCH -D ~/projects/def-keli/tme3/ISICLE

# Uncomment to control the output files. By default stdout and stderr go to
# the same place, but if you use both commands below they'll be split up.
# %N is the hostname (if used, will create output(s) per node).
# %j is jobid.

#SBATCH --output=models/camnet/Colorization/camnet3_centi-bs12-code_bs120-epochs10-gpus_0-ipe2-mini_bs2/slurm_output.txt

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

echo "Starting job..."

source ~/.bashrc
source ~/

# Python will buffer output of your script unless you set this.
# If you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed.
export PYTHONUNBUFFERED=1

# Do all the research.
source ../py39ISICLE/bin/activate
python TrainCAMNet.py --task Colorization --data camnet3_centi --epochs 100 --code_bs 120 --mini_bs 6 --bs 12
python generators/camnet/test.py -opt models/camnet/Colorization/camnet3_centi-bs12-code_bs120-epochs10-gpus_0-ipe2-mini_bs2/test_config.json

# Print completion time.
date
