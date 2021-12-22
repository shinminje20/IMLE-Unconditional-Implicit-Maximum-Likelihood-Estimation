"""File containing utilities."""
import math
import os
import random
from tqdm import tqdm

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# Set up CUDA usage
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if "cuda" in device:
    torch.backends.cudnn.benchmark = True

# Make non-determinism work out. This function should be called first
def set_seed(seed):
    """Seeds the program to use seed [seed]."""
    random.seed(seed)
    torch.manual_seed(seed)
    tqdm.write(f"Set the PyTorch and Random modules seeds to {seed}")

################################################################################
# I/O Utils
################################################################################
state_sep_str = "=" * 40

project_dir = os.path.dirname(os.path.abspath(__file__))

def strip_slash(s):
    """Returns string [s] without a trailing slash.

    This project uses rather basic path-handling, which makes for slightly
    clunky but easier-to-debug code. Generally, paths CAN NOT end in slashes or
    f-strings using them will break!
    """
    return s if not s[-1] == "/" else s[:-1]

def opts_str(args):
    """Returns the options string for [args]."""
    return f"-{'-'.join(args.options)}" if len(args.options) > 0 else "-"

def suffix_str(args):
    """Returns the suffix string for [args]."""
    return f"-{args.suffix}" if not args.suffix == "" else ""

def load_(file):
    """Returns a (model, optimizer, last_epoch, args, tensorboard results) tuple
    from [file].
    """
    data = torch.load(file)
    model = data["model"].to(device)
    optimizer = data["optimizer"]
    last_epoch = data["last_epoch"]
    args = data["args"]
    tb_results = data["tb_results"]
    return model, optimizer, last_epoch, args, tb_results

def save_(model, optimizer, last_epoch, args, tb_results, folder):
    """Saves input experiment objects to the [last_epoch].pt file [folder]."""
    tb_results.flush()
    tb_results.close()
    torch.save({"model": model.cpu(), "optimizer": optimizer,
        "last_epoch": last_epoch, "args": args, "tb_results": tb_results},
        f"{folder}/{last_epoch}.pt")
    model.to(device)

def generator_folder(args):
    """Returns the folder to save a generator trained with [args] to."""
    folder = f"{project_dir}/models/generators-{args.data}-{opts_str(args)}{suffix_str(args)}"
    if not os.path.exists(folder): os.makedirs(folder)
    return folder

def resnet_folder(args):
    """Returns the folder to to which to save a resnet trained with [args]."""
    folder = f"{project_dir}/models/resnets-{args.backbone}-{args.data}{opts_str(args)}{suffix_str(args)}"
    if not os.path.exists(folder): os.makedirs(folder)
    return folder

################################################################################
# Miscellaneous
################################################################################

def flatten(xs):
    """Returns collection [xs] after recursively flattening into a list."""
    result = []
    for x in xs:
        if isinstance(x, list) or isinstance(x, set) or isinstance(x, tuple):
            result += flatten(x)
        else:
            result.append(x)

    return result
