"""File to use for submitting training scripts. Each run of this script will
submit a SLURM script corresponding to a singular run of a experiment script,
which handling SLURMification—in particular, chunking the job for ComputeCanada.

Because of the involved complexity and the fact that this project is extremely
compute-heavy, there's no support for hyperparameter tuning here; one must
submit a job for each desired hyperparameter configuration.

USAGE:
python SlurmSubmit.py Script.py --arg1 ...

It should be that deleting SlurmSubmit.py from the command yields exactly the
command desired for SLURM to run.
"""
import os
import sys
from utils.Utils import *

if __name__ == "__main__":
    sys_args = sys.argv
    with open("slurm/slurm_template.txt", "r") as f:
        slurm_template = f.read()

    if sys_args[1] == "TrainGeneratorWandB.py":
        from TrainGeneratorWandB import get_args
        args = get_args(sys_args[2:])
        CHUNKS = str(args.epochs - 1)
        NAME = generator_folder(args).replace(f"{project_dir}/generators/", "")
        NUM_GPUS = str(len(args.gpus))
        slurm_template = slurm_template.replace("CHUNKS", CHUNKS)
        slurm_template = slurm_template.replace("NAME", NAME)
        slurm_template = slurm_template.replace("NUM_GPUS", NUM_GPUS)
    else:
        raise ValueError(f"Unknown script '{sys_args[2]}")

    SCRIPT = f"python {' '.join(sys_args[1:])} --resume $SLURM_ARRAY_TASK_ID --jobid $SLURM_ARRAY_JOB_ID --data_path ~/scratch/ISICLE/data"
    slurm_template = slurm_template.replace("SCRIPT", SCRIPT)
    
    slurm_script = f"slurm/_{NAME}.sh"
    with open(slurm_script, "w+") as f:
        f.write(slurm_template)

    os.system(f"sbatch {slurm_script}")
    


