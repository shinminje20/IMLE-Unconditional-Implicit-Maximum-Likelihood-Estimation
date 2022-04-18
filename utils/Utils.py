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
import wandb

import torch
import torch.nn as nn
from torchvision.transforms import functional as functional_TF
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

################################################################################
# Set up seeds, CUDA, and number of workers
################################################################################
# Set up CUDA usage
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

num_workers = 6

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
# Miscellaneous utilities
################################################################################

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

def flatten(xs):
    """Returns collection [xs] after recursively flattening into a list."""
    if isinstance(xs, list) or isinstance(xs, set) or isinstance(xs, tuple):
        result = []
        for x in xs:
            result += flatten(x)
        return result
    else:
        return [xs]

def round_so_evenly_divides(x, y):
    """Returns [x] adjusted up or down by up to so [y] divides it evenly."""
    return x + (y - (x % y)) if ((x % y) > y / 2) else x - (x % y)

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

################################################################################
# File I/O Utils
################################################################################
def check_paths_exist(paths):
    """Raises a ValueError if every path in [paths] exists, otherwise does
    nothing.
    """
    for path in flatten(paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path}' but this path couldn't be found")

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def strip_slash(s):
    """Returns string [s] without a trailing slash.

    This project uses rather basic path-handling, which makes for slightly
    clunky but easy-to-debug code. Generally, paths CAN NOT end in slashes or
    f-strings using them will break!
    """
    return s if not s[-1] == "/" else s[:-1]

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

def opts_str(args):
    """Returns the options string for [args]."""
    return f"-{'-'.join(args.options)}" if len(args.options) > 0 else ""

def suffix_str(args):
    """Returns the suffix string for [args]."""
    return f"-{args.suffix}" if not args.suffix == "" else ""

def simclr_folder(args):
    """Returns the folder to which to save a resnet trained with [args]."""
    folder = f"{project_dir}/models_simclr/{args.data}_{args.backbone}_{args.run_id}{suffix_str(args)}"
    if not os.path.exists(folder): os.makedirs(folder)
    return folder

def generator_folder(args):
    """Returns the folder to which to save a Generator saved with [args].

    Because there are so many arguments to a generator, it's impossible to get
    them all into a file name, so we need to use times to keep track of things.
    This is also the reason we use WandB!
    """
    return f"{project_dir}/generators/camnet_{args.data}_bs{args.bs}-grayscale{args.grayscale}-ipcpe{args.ipcpe}-lr{args.lr}-mask_frac{args.mask_frac}-mask_res{args.mask_res}-ns{'_'.join([str(n) for n in args.ns])}-res{'_'.join([str(r) for r in args.res])}" + suffix_str(args)
    if not os.path.exists(folder): os.makedirs(folder)
    return folder

def isicle_folder(args):
    """
    """
    folder = f"{project_dir}/models_isicle/{args.data}_{args.backbone}_{args.run_id}{suffix_str(args)}"
    if not os.path.exists(folder): os.makedirs(folder)
    return folder

def wandb_load(checkoint):
    """Returns a (wandb run ID, data) tuple.

    Args:
    checkpoint  -- either a local checkpoint dictionary to resume from, or
                    something of the form RUN_PATH/CHECKPOINT, eg.
                    'tristanengst/isicle-generator/abc123/40.pt'
    """
    saved_data = torch.load(checkoint)
    return saved_data["run_id"], saved_data

def wandb_save(dictionary, path):
    """Saves [file] to WandB folder [path] and uploads the result to WandB."""
    resume_dictionary = {"seed": {
        "pytorch_seed": torch.get_rng_state(),
        "numpy_seed": np.random.get_state(),
        "random_seed": random.getstate()}
    }
    torch.save(dictionary | resume_dictionary, path)
    tqdm.write(f"Saved files to {path}")

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
    tqdm.write(f"Saved image grid to {path}")


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
