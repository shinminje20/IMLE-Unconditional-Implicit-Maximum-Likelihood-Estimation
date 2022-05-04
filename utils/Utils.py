"""File containing utilities."""
from copy import deepcopy
from datetime import datetime
import functools
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.transforms import functional as functional_TF

################################################################################
# Set up seeds, CUDA, and number of workers
################################################################################
# Set up CUDA usage
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
num_workers = 24

# Turn off WandB console logging, since we don't need it and it breaks TQDM.
os.environ["WANDB_CONSOLE"] = "off"

# Make non-determinism work out. This function should be called first
def set_seed(seed):
    """Seeds the program to use seed [seed]."""
    if isinstance(seed, int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        tqdm.write(f"Set the NumPy, PyTorch, and Random modules seeds to {seed}")
    elif isinstance(seed, dict):
        random.setstate(seed["random_seed"])
        np.random.set_state(seed["numpy_seed"])
        torch.set_rng_state(seed["pytorch_seed"])
        tqdm.write(f"Reseeded program with old seed")
    else:
        raise ValueError(f"Seed should be int or contain resuming keys")

    return seed

################################################################################
# File I/O Utils
################################################################################
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = f"{project_dir}/data"

def experiment_folder(args, candidate_folder):
    """Returns and creates [candidate_folder] it does not exist, or if it exists
    and was created from the same hyperparameters as those in [args], or raises
    a ValueError.

    Args:
    args                -- Argparse namespace
    candidate_folder    -- folder we are attempting to save to
    """
     # Get the hyperparameter to save folder mapping
    experiment_dir = os.path.dirname(candidate_folder)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if not os.path.exists(f"{experiment_dir}/hparams2folder.json"):
        with open(f"{experiment_dir}/hparams2folder.json", "w+") as f:
            json.dump({}, f)
    with open(f"{experiment_dir}/hparams2folder.json", "r") as f:
        hparam2folder = json.load(f)

    # If we already have a mapping from the hyperparameters, just return it. If
    # [candidate_folder] already exists under another set of hyperparameters,
    # raise an error if the file does in fact still exist. If it doesn't exist
    # add the mapping and return.
    hparam_str = args_to_hparam_str(args)
    if hparam_str in hparam2folder:
        return hparam2folder[hparam_str]
    elif candidate_folder in hparam2folder.values():
        all_folders = os.listdir(experiment_dir)
        all_folders = [f"{experiment_dir}/{f}" for f in all_folders]
        if candidate_folder in all_folders:
            raise ValueError(f"{candidate_folder.replace(project_dir, '').strip('/')}    already exists. Delete it or add a new --suffix to create a unique experiment (or improve the naming scheme!)")
        else:
            folder2hparam = {v: k for k,v in hparam2folder.items()}
            del hparam2folder[folder2hparam[candidate_folder]]
            with open(f"{experiment_dir}/hparams2folder.json", "w+") as f:
                json.dump(hparam2folder, f)
            if not os.path.exists(candidate_folder):
                os.makedirs(candidate_folder)
            return candidate_folder
    else:
        hparam2folder[hparam_str] = candidate_folder
        with open(f"{experiment_dir}/hparams2folder.json", "w+") as f:
            json.dump(hparam2folder, f)
        if not os.path.exists(candidate_folder):
            os.makedirs(candidate_folder)
        return candidate_folder

def suffix_str(args):
    """Returns the suffix string for [args]."""
    return f"-{args.suffix}" if not args.suffix == "" else ""

def simclr_folder(args):
    """Returns the folder to which to save a resnet trained with [args]."""
    folder = f"{project_dir}/models_simclr/{args.data}-{args.backbone}{suffix_str(args)}"
    return experiment_folder(args, folder)

def generator_folder(args):
    """Returns the folder to which to save a Generator saved with [args]."""
    folder = f"{project_dir}/generators/{args.data}-bs{args.bs}-grayscale{args.grayscale}-ipc{args.ipc}-lr{args.lr}-mask_frac{args.mask_frac}-mask_res{args.mask_res}-norm{args.normalize}-ns{'_'.join([str(n) for n in args.ns])}-res{'_'.join([str(r) for r in args.res])}-seed{args.seed}" + suffix_str(args)
    return experiment_folder(args, folder)

def isicle_folder(args):
    raise NotImplementedError()

################################################################################
# File I/O
################################################################################
def args_to_hparams(args):
    """Returns a dictionary of hyperparameters from Namespace [args]."""
    excluded_args =  ["resume", "chunk_epochs", "gpus", "comet", "data_path",
        "wandb", "val_iter", "suffix", "spi", "run_id", "sp"]
    return {k: v for k,v in vars(args).items() if not k in excluded_args}

def args_to_hparam_str(args):
    """Returns the string uniquely given by the hyperparametes in [args]."""
    hparams_str = [f"{k}_{v}" if isinstance(v, str) else f"{k}{v}"
        for k,v in args_to_hparams(args).items()]
    return "-".join(sorted(hparams_str))

def save_checkpoint(dictionary, path):
    """Saves contents of [dictionary] along with random states to [file]."""
    seed_states = {"seed": {
        "pytorch_seed": torch.get_rng_state(),
        "numpy_seed": np.random.get_state(),
        "random_seed": random.getstate()}
    }
    torch.save(dictionary | seed_states, path)
    tqdm.write(f"Saved files to {path.replace(project_dir, '')}")

################################################################################
# Printing I/O Utilities
################################################################################

def dict_to_nice_str(dict, max_line_length=80):
    """Returns a pretty string representation of [dict]."""
    s, last_line_length = "", 0
    for k in sorted(dict.keys()):
        item_len = len(f"{k}: {dict[k]}, ")
        if last_line_length + item_len > max_line_length:
            s += f"\n{k}: {dict[k]}, "
            last_line_length = item_len
        else:
            s += f"{k}: {dict[k]}, "
            last_line_length += item_len
    return s

################################################################################
# Image I/O Utilities
################################################################################
plt.rcParams["savefig.bbox"] = "tight"
plt.tight_layout(pad=0.00)

def make_2d_list_of_tensor(x):
    """Returns [x] as a 2D list where inner element is a Tensor."""
    if isinstance(x, torch.Tensor):
        return [[x]]
    elif isinstance(x, list) and isinstance(x[0], torch.Tensor):
        return [x]
    elif isinstance(x, list) and isinstance(x[0], list):
        return x
    else:
        raise ValueError("Unknown collection of types in 'images'")

def show_image_grid(images):
    """Shows list of images [images], either a Tensor giving one image, a List
    where each element is a Tensors giving one images, or a 2D List where each
    element is a Tensor giving an image.
    """
    images = make_2d_list_of_tensor(images)

    fix, axs = plt.subplots(ncols=max([len(image_row) for image_row in images]),
        nrows=len(images), squeeze=False)
    for i,images_row in enumerate(images):
        for j,image in enumerate(images_row):
            axs[i, j].imshow(np.asarray(functional_TF.to_pil_image(image.detach())), cmap='Greys_r')
            axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def save_image_grid(images, path):
    """Builds a grid of images out of [images] and saves the image containing
    the grid to [path].
    """
    images = make_2d_list_of_tensor(images)

    fix, axs = plt.subplots(ncols=max([len(image_row) for image_row in images]),
        nrows=len(images), squeeze=False)
    for i,images_row in enumerate(images):
        for j,image in enumerate(images_row):
            axs[i, j].imshow(np.asarray(functional_TF.to_pil_image(image.detach())), cmap='Greys_r')
            axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    plt.savefig(path, dpi=512)
    plt.close("all")


################################################################################
# Miscellaneous utilities
################################################################################

def remove_duplicates(x):
    """Removes duplicates from order 1 list [x]."""
    seen_elements = set()
    result = []
    for e in x:
        if e in seen_elements:
            continue
        else:
            result.append(e)
            seen_elements.add(e)

    return result

def make_cpu(input):
    if isinstance(input, list):
        return [make_cpu(x) for x in input]
    else:
        return input.cpu()

def make_device(input):
    if isinstance(input, list):
        return [make_device(x) for x in input]
    else:
        return input.to(device)

def make_3dim(input):
    if isinstance(input, list):
        return [make_3dim(x) for x in input]
    elif isinstance(input, torch.Tensor) and len(input.shape) == 4 and input.shape[0] == 1:
        return input.squeeze(0)
    elif isinstance(input, torch.Tensor) and len(input.shape) == 3:
        return input
    else:
        raise ValueError()

def evenly_divides(x, y):
    """Returns if [x] evenly divides [y]."""
    return int(y / x) == y / x

def round_so_evenly_divides(x, y):
    """Returns [x] adjusted up or down by up to so [y] divides it evenly."""
    return x + (y - (x % y)) if ((x % y) > y / 2) else x - (x % y)

def flatten(xs):
    """Returns collection [xs] after recursively flattening into a list."""
    if isinstance(xs, list) or isinstance(xs, set) or isinstance(xs, tuple):
        result = []
        for x in xs:
            result += flatten(x)
        return result
    else:
        return [xs]

def get_all_files(f):
    """Returns absolute paths to all files under [f]."""
    if os.path.isdir(f):
        return flatten([get_all_files(f"{f}/{x}") for x in os.listdir(f)])
    else:
        return f

def make_list(x, length=1):
    """Returns a list of length [length] where each elment is [x], or, if [x]
    is a list of length [length], returns [x].
    """
    if isinstance(x, list) and len(x) == length:
        return x
    elif isinstance(x, list) and len(x) == 1:
        return x * length
    elif isinstance(x, list) and not len(x) == length and len(x) > 1:
        raise ValueError(f"Can not convert list {x} to length {length}")
    else:
        return [x] * length

def json_to_dict(f):
    """Returns the dictionary given by JSON file [f]."""
    if isinstance(f, str) and json_file.endswith(".json") and os.path.exists(f):
        with open(f, "r") as json_file:
            return json.load(json_file)
    else:
        return ValueError(f"Can not read dictionary from {f}")

def dict_to_json(dictionary, f):
    """Saves dict [dictionary] to file [f]."""
    with open(f, "w+") as f:
        json.dump(dictionary, f)