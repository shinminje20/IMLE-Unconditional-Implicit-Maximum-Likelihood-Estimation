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
import argparse
import os
from utils.Utils import *

def get_time():
    if hours == 3:
        return "0-2:59:59"
    elif hours == 12:
        return "0-11:59:59"
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("script",
        help="script to run")
    P.add_argument("--time", type=int, default=3,
        help="number of hours per task")
    submission_args, unparsed_script_args = P.parse_known_args()

    with open("slurm/slurm_template.txt", "r") as f:
        slurm_template = f.read()

    if submission_args.script == "TrainGeneratorWandB.py":
        from TrainGeneratorWandB import get_args
        args = get_args(unparsed_script_args)
        CHUNKS = str(args.epochs - 1)
        NAME = generator_folder(args).replace(f"{project_dir}/generators/", "")
        NUM_GPUS = str(len(args.gpus))
        TIME = get_time(args.time)
        slurm_template = slurm_template.replace("CHUNKS", CHUNKS)
        slurm_template = slurm_template.replace("TIME", TIME)
        slurm_template = slurm_template.replace("NAME", NAME)
        slurm_template = slurm_template.replace("NUM_GPUS", NUM_GPUS)
    else:
        raise ValueError(f"Unknown script '{submission_args.script}")

    SCRIPT = f"python {' '.join(unparsed_script_args)} --resume $SLURM_ARRAY_TASK_ID --jobid $SLURM_ARRAY_JOB_ID --data_path ~/scratch/ISICLE/data"
    slurm_template = slurm_template.replace("SCRIPT", SCRIPT)
    
    slurm_script = f"slurm/_{NAME}.sh"
    with open(slurm_script, "w+") as f:
        f.write(slurm_template)

    os.system(f"sbatch {slurm_script}")
    


