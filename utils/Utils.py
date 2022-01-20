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

# Make non-determinism work out. This function should be called first
def set_seed(seed):
    """Seeds the program to use seed [seed]."""
    random.seed(seed)
    torch.manual_seed(seed)
    tqdm.write(f"Set the PyTorch and Random modules seeds to {seed}")

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

class NestedNamespace():
    """Class representing a NestedNamespace. Any value that can be interpreted
    as a dictionary (dictionary, JSON file, anything with a __dict__) attribute
    is pulled into the hierarchy as a NestedNamespace itself. There are three
    cases for adding a new item:

    1. If a key is None or ends with config and maps to a dictionary-like value,
        then the key-value pairs in the value are added to the hierarchy where
        the key would have been, and the key isn't added.
    2. The value is dictionary-like, but case (1) doesn't hold. Then the
        key-value pairs in the value are added below the key in the hierarchy.
    3. Otherwise, the key value pair is simple and added to the hierarchy.
    """
    def __init__(self, *inputs):
        for input in inputs:
            if isinstance(input, dict):
                for k,v in input.items():
                    self.add_item(k, v)
            elif hasattr(input, "__dict__"):
                for k,v in input.__dict__.items():
                    self.add_item(k, v)
            elif (isinstance(input, str) and input.endswith(".json")
                and os.path.exists(input)):
                with open(input, "r") as f:
                    for k,v in json.load(f).items():
                        self.add_item(k, v)
            else:
                raise ValueError(f"Not able to build {input} into Namespace")

    def __str__(self): return f"NestedNamespace({', '.join([f'{k}={str(v)}' for k,v in self.__dict__.items()])})"
    def __repr__(self): return self.__str__()
    def __eq__(self, other): return self.__repr__() == other.__repr__()


    def set_attribute(self, k, v, override=False):
        """Creates attribute [key] set to value [value].

        Args:
        k           -- the key to set
        v           -- the value to set
        overrride   -- whether or not to allow setting an already existing key
        """
        if hasattr(self, k) and not override:
            raise ValueError(f"key {k} used twice")
        else:
            setattr(self, deepcopy(k), deepcopy(v))

    def add_item(self, k, v):
        """Adds key [k] and value [v] to the NestedNamespace."""
        if k.endswith("config") and NestedNamespace.is_dict_like(v):
            for k_,v_ in NestedNamespace(v).__dict__.items():
                self.set_attribute(k_, v_)
        elif NestedNamespace.is_dict_like(v):
            self.set_attribute(k, NestedNamespace(v))
        else:
            self.set_attribute(k, v)

    @staticmethod
    def to_dict(x, allow_identity=False):
        """Returns the NestedNamespace [x] as a dictionary."""
        if allow_identity and not isinstance(x, NestedNamespace):
            return deepcopy(x)
        elif isinstance(x, NestedNamespace):
            return {deepcopy(k): NestedNamespace.to_dict(v, allow_identity=True)
                for k,v in x.__dict__.items()}
        else:
            raise TypeError(f"'{x}' is of type {type(x)} not NestedNamespace")

    @staticmethod
    def leaf_union(x, y):
        """Returns a NestedNamespace that is the union of NestedNamespaces [x]
        and [y].

        When the values of an attribute in both [x] and [y] are
        NestedNamespaces, then the returned NestedNamespace has this attribute
        set to leaf_union(x.attribute, y.attribute).

        Otherwise, precedence is given to [y], so when [x] and [y] have
        different values for an attribute, the returned NestedNamespace has that
        attribute set to the value of that attribute in [y].
        """
        x, y = NestedNamespace(x), NestedNamespace(y)
        result = NestedNamespace(x)

        for k,y_v in y.__dict__.items():
            x_v = x.__dict__[k] if k in x.__dict__ else None
            if (NestedNamespace.is_dict_like(x_v)
                and NestedNamespace.is_dict_like(y_v)):
                new_val = NestedNamespace.leaf_union(x_v, y_v)
                result.set_attribute(k, new_val, override=True)
            else:
                result.set_attribute(k, y_v, override=True)

        return result

    @staticmethod
    def is_dict_like(x):
        """Returns if [x] can be interpreted like a dictionary."""
        return ((hasattr(x, "__dict__"))
            or (isinstance(x, dict))
            or (isinstance(x, str) and x[:-5] == ".json" and os.path.exists(x)))

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

def flatten(xs):
    """Returns collection [xs] after recursively flattening into a list."""
    if isinstance(xs, list) or isinstance(xs, set) or isinstance(xs, tuple):
        result = []
        for x in xs:
            result += flatten(x)
        return result
    else:
        return [xs]
