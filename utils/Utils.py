"""File containing utilities."""
from copy import deepcopy
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.transforms import functional as functional_TF
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# Set up CUDA usage
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if "cuda" in device:
    torch.backends.cudnn.benchmark = True

num_workers = 6

# Make non-determinism work out. This function should be called first
def set_seed(seed):
    """Seeds the program to use seed [seed]."""
    random.seed(seed)
    torch.manual_seed(seed)
    tqdm.write(f"Set the PyTorch and Random modules seeds to {seed}")

################################################################################
# Interactive I/O Utils
################################################################################


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


state_sep_str = "=" * 40

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def strip_slash(s):
    """Returns string [s] without a trailing slash.

    This project uses rather basic path-handling, which makes for slightly
    clunky but easier-to-debug code. Generally, paths CAN NOT end in slashes or
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

def load_camnet(file):
    pass

def camnet_folder(args):
    """Returns the folder for saving a CAMNet model trained with [args].

    Due to CAMNet's existing workings, models will actually be saved to
    project_directory/models/camnet/camnet_folder(args)
    """
    folder = f"{args.task}/{args.data}{opts_str(args)}{suffix_str(args)}"
    actual_folder = f"{project_dir}/models/camnet/{folder}"
    if not os.path.exists(actual_folder): os.makedirs(actual_folder)
    return actual_folder

def load_simclr(file):
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

def save_simclr(model, optimizer, last_epoch, args, tb_results, folder):
    """Saves input experiment objects to the [last_epoch].pt file [folder]."""
    tb_results.flush()
    tb_results.close()
    torch.save({"model": model.cpu(), "optimizer": optimizer,
        "last_epoch": last_epoch, "args": args, "tb_results": tb_results},
        f"{folder}/{last_epoch}.pt")
    model.to(device)

def simclr_folder(args):
    """Returns the folder to which to save a resnet trained with [args]."""
    folder = f"{project_dir}/models/simclr/{args.data}/{args.backbone}/{opts_str(args)}{suffix_str(args)}"
    if not os.path.exists(folder): os.makedirs(folder)
    return folder

################################################################################
# Image I/O Utilities
################################################################################
plt.rcParams["savefig.bbox"] = "tight"

def show_image_grid(images):
    """Shows list of images [images], either a Tensor giving one image, a List
    where each element is a Tensors giving one images, or a 2D List where each
    element is a Tensor giving an image.
    """
    if isinstance(images, torch.Tensor):
        images = [[images]]
    elif isinstance(images, list) and isinstance(images[0], torch.Tensor):
        images = [images]
    elif isinstance(images, list) and isinstance(images[0], list):
        images = images
    else:
        raise ValueError("Unknown collection of types in 'images'")

    fix, axs = plt.subplots(ncols=max([len(image_row) for image_row in images]),
                            nrows=len(images),
                            squeeze=False)

    for i,images_row in enumerate(images):
        for j,image in enumerate(images_row):
            axs[i, j].imshow(np.asarray(functional_TF.to_pil_image(image.detach())), cmap='Greys_r')
            axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()

################################################################################
# Miscellaneous
################################################################################


class CosineAnnealingLinearRampLR(_LRScheduler):
    """Cosine Annealing scheduler with a linear ramp."""

    def __init__(self, optimizer, T_0, n_ramp, T_mult=1, eta_min=0,
        last_epoch=-1, verbose=False):
        """
        Args:
        optimizer   -- the wrapped optimizer
        T_0         -- base COSINE period
        n_ramp      -- number of linear ramp epochs
        T_mult      -- multiplicative period change
        eta_min     -- minumum learning rate
        last_epoch  -- index of the last epoch run
        verbose     -- whether to have verbose output or not
        """
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if n_ramp < 0 or not isinstance(n_ramp, int):
            raise ValueError(f"Expected integer n_ramp >= 0, but got {n_ramp}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        self.T_0 = T_0
        self.n_ramp = n_ramp
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.ramped = (last_epoch >= self.n_ramp)
        self.T_cur = last_epoch

        super(CosineAnnealingLinearRampLR, self).__init__(optimizer, last_epoch,
            verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if not self.ramped:
            return [b * ((self.T_cur + 1) / self.n_ramp) for b in self.base_lrs]
        else:
            cos = (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            return [self.eta_min + (b - self.eta_min) * cos for b in self.base_lrs]

    def step(self):
        if not self.ramped and self.last_epoch + 1 < self.n_ramp:
            self.T_cur += 1
            self.last_epoch += 1
        elif not self.ramped and self.last_epoch + 1 >= self.n_ramp:
            self.last_epoch += 1
            self.T_cur = 0
            self.ramped = True
        elif self.ramped and self.T_cur >= self.T_i:
            self.last_epoch += 1
            self.T_cur = self.T_cur - self.T_i
            self.T_i = self.T_i * self.T_mult
        elif self.ramped and self.T_cur < self.T_i:
            self.last_epoch += 1
            self.T_cur += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group["lr"] = lr
                self.print_lr(self.verbose, i, lr, max(0, self.last_epoch))

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

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
