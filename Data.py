"""Code for building datasets.

See data/SetupDataset.py for downloading their underlying data.

The important portions of this file are get_data_splits(), which returns
training and evaluation ImageFolders, and the various Dataset subclasses that
can be used to construct various useful datasets.
"""
from collections import OrderedDict, defaultdict
from copy import deepcopy
import numpy as np
import random
import sys
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split

from torchvision.datasets import ImageFolder, CIFAR10
from torchvision import transforms
from torchvision.transforms.functional import hflip

from Corruptions import *
from utils.Utils import *

################################################################################
# Dataset medatata. For each dataset, we need to know what splits it has, and
# what resolutions it exists at. The initial declaration of [dataset2metadata]
# gives the "base" datasets known to the repository. For instance, the folder
#
#   data_dir/bird_128x128/val
#
# can be turned into an ImageFolder, where [data_dir] is usually but not
# necessarily the data folder (if it isn't, the --data_path argument must be
# specified).
#
# Because datasets may be large, one can specify a suffix for each that will
# indicate how much smaller it is. The exact amount by which such a dataset can
# be smaller is somewhat interpretive so it can be done intelligently. For
# instance, the camnet3 dataset contains 3840 training examples and 60
# validation examples, each divided evenly into three classes. The camnet3_centi
# dataset would likely contain 36≈3840 / 100 training examples—36 is easier to
# work with than 38—and the same 60 validation samples, as 60 is already small.
# Making the size of a dataset smaller would typically not impact the
# resolutions we maintain for it. The training split of the camnet3_centi
# dataset, at, say, 32x32 resolution would then require its own
# ImageFolder-compatiblefolder at
#
#   data_dir/camnet3_centi_32x32/train
#
# Datasets can be resized via the MakeSmallDataset.py script in the data folder.
#
# It is assumed that all suffixes exist for all datasets in this code. This is
# not necessarily the case, and as such [datasets] contains a superset of the
# actually available data. This permits less memory usage, and constructing a
# dataset where the underlying data is missing will throw an error.
################################################################################
data_suffixes = ["", "_deci", "_centi", "_milli"]
dataset2metadata = {
    "bird": {"splits": ["train", "val"],
        "res": [16, 32, 64, 128, 256],
        "same_distribution_splits": True},
    "butterfly": {"splits": ["train", "val"],
        "res": [16, 32, 64, 128, 256],
        "same_distribution_splits": True},
    "camnet3": {"splits": ["train", "val"],
        "res": [16, 32, 64, 128, 256],
        "same_distribution_splits": True},
    "strawberry": {"splits": ["train", "val"],
        "res": [16, 32, 64, 128, 256],
        "same_distribution_splits": True},
    "cifar10": {"splits": ["train", "test"],
        "res": [32]},
    "miniImagenet": {"splits": ["train", "val", "test"],
        "res": [32, 64, 128, 256],
        "same_distribution_splits": False},
}
dataset2metadata = {f"{d}{s}": v for d,v in dataset2metadata.items()
    for s in data_suffixes}
datasets = dataset2metadata.keys()

def data_name_without_suffix(data_name):
    data_name = data_name.replace("_deci", "")
    data_name = data_name.replace("_centi", "")
    data_name = data_name.replace("_milli", "")
    return data_name

def seed_kwargs(seed=0):
    """Returns kwargs to be passed into a DataLoader to give it seed [seed]."""
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    return {"generator": g, "worker_init_fn": seed_worker}

################################################################################
# Functionality for loading datasets
################################################################################

def get_data_splits(data_str, eval_str="cv", res=32, data_path=data_dir):
    """Returns data for training and evaluation. All Datasets returned are
    ImageFolders, meaning that another kind of dataset likely needs to be built
    on top of them.

    Args:
    data_str    -- string specifying the dataset to load. It must exactly exist
                    in the data directory, ie. data/data_str exists.
    eval_str    -- how to get validation/testing data
    res         -- resolutions in which to return data
                    - if None, returned data will correspond to [data_str]
                    - if a single int, returned data will correspond to
                        [data_str] at resolution [res]
                    - if a list of ints, returned data will be a list of
                        datasets with the ith value in [res] the [ith]
                        resolution of the returned data. Evaluation data will
                        have two resolutions, sequentially, the min and max of
                        [res]
    data_path   -- path to dataset; data can be found at data_path/data_str
    """
    def path_to_imagefolder(paths):
        """Returns ImageFolder(s) given [paths], a list of data paths for a
        list of ImageFolders, or a string for a single ImageFolder.
        """
        if isinstance(paths, list):
            return [PreAugmentedImageFolder(p) for p in paths]
        else:
            return PreAugmentedImageFolder(paths)

    data_path = f"{data_path}/{data_str}"
    eval_str = "train" if eval_str == "cv" else eval_str
    if data_str not in dataset2metadata:
        raise ValueError(f"dataset {data_str} not in dataset2metadata dictionary")
    if not eval_str in dataset2metadata[data_str]["splits"]:
        raise ValueError(f"dataset {data_str} has no split {eval_str}")

    ############################################################################
    # CIFAR-10 has its own weird logic
    ############################################################################
    if data_str == "cifar10" and (res == 32 or all([r == 32 for r in res])):
        data_tr = CIFAR10(root=data_path, train=True, download=True)
        data_eval = CIFAR10(root=data_path, train=(eval_str == "train"),
            download=True)
        return data_tr, data_eval

    if res is None:
        data_paths_tr = f"{data_path}/train"
        data_paths_eval = f"{data_path}/{eval_str}"
    elif isinstance(res, int) or (isinstance(res, list) and len(res) == 1):
        res = res if isinstance(res, int) else res[0]
        data_paths_tr = f"{data_path}_{res}x{res}/train"
        data_paths_eval = f"{data_path}_{res}x{res}/{eval_str}"
    elif isinstance(res, list) and len(res) > 1:
        data_paths_tr = [f"{data_path}_{r}x{r}/train" for r in res]
        data_paths_eval = [f"{data_path}_{r}x{r}/{eval_str}" for r in res]
    else:
        raise ValueError(f"Unmatched type for `res`: {res}")

    return path_to_imagefolder(data_paths_tr), path_to_imagefolder(data_paths_eval)

def get_artificial_train_val_splits(data, val_size=None, val_fraction=None):
    """Returns a (training data, validation data) tuple by splitting [data]
    according to [val_size] or [val_fraction], with the validationd data chosen
    evenly spaced within [data].
    """
    if val_fraction is not None and (val_fraction < 0 or val_fraction > 1):
        raise ValueError(f"Got `val_fraction` of {val_fraction}, but it must be in [0, 1].")
    if val_size is not None and val_size > len(data):
        raise ValueError(f"Got `val_size` of {val_size} but only {len(data)} examples exist.")

    if val_size is not None and val_fraction is not None:
        val_length = max(val_size, int(val_frac * len(data)))
        tqdm.write("Both `val_fraction` and `val_size` are specified. Whichever leads to a larger validation split will is used, giving a validation split of length {val_length}.")
    elif val_size is not None:
        val_length = val_size
    elif val_fraction is not None:
        val_length = int(val_frac * len(data))
    else:
        raise ValueError("Impossible case")

    val_idxs = set(range(0, len(data), len(data) // val_length))
    train_idxs = {idx for idx in range(len(data)) if not idx in val_idxs}
    return Subset(data, indices=train_idxs), Subset(data, indices=val_idxs)


################################################################################
# Augmentations
################################################################################
def get_real_augs(res=32):
    """Returns augmentations that ensure images remain on the real manifold."""
    augs_tr = transforms.Compose([
        transforms.RandomResizedCrop(res),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    augs_te = transforms.Compose([
        transforms.RandomResizedCrop(res),
        transforms.ToTensor()])

    return augs_tr, augs_tr, augs_te

def get_contrastive_augs(res=32, gaussian_blur=False, color_s=0):
    """Returns a (SSL transforms, finetuning transforms, testing transforms)
    tuple based on [data_str].

    Args:
    res             -- the resolution of images output by the transforms
    gaussian_blur   -- whether to use Gaussian blur or not
    color_s         -- color distortion strength
    """
    color_jitter = transforms.ColorJitter(0.8 * color_s,
         0.8 * color_s, 0.8 * color_s, 0.2 * color_s)
    color_distortion = transforms.Compose([
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2)])

    augs_tr_list = [transforms.RandomResizedCrop(res)]

    if color_s > 0:
        augs_tr_list.append(color_distortion)
    if gaussian_blur:
        augs_tr_list.append(transforms.GaussianBlur(res // 10, sigma=(.1, 2)))

    augs_tr_list += [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ]

    augs_tr = transforms.Compose(augs_tr_list)

    augs_te = transforms.Compose([
        transforms.RandomResizedCrop(res),
        transforms.ToTensor()])

    return augs_tr, augs_tr, augs_te

def get_gen_augs(args):
    """Returns a list of base transforms for image generation. Each should be
    able to accept multiple input images and be deterministic between any two
    images input at the same time, and return a list of the transformed images.
    """
    class RandomHorizontalFlips(nn.Module):
        """RandomHorizontalFlip but can be applied to multiple images."""
        def __init__(self, p=0.5):
            super(RandomHorizontalFlips, self).__init__()
            self.p = p

        def forward(self, images):
            """Returns [images] but with all elements flipped in the same
            direction, with the direction chosen randomly.

            Args:
            images  -- list of (PIL Image or Tensor): Images to be flipped
            """
            if torch.rand(1) < self.p:
                return [hflip(img) for img in images]
            return images

        def __repr__(self): return f"{self.__class__.__name__}(p={self.p})"

    class ToTensors(nn.Module):
        def __init__(self):
            super(ToTensors, self).__init__()
            self.to_tensor = transforms.ToTensor()

        def forward(self, images): return [self.to_tensor(x) for x in images]

        def __repr__(self): return self.__class__.__name__

    return transforms.Compose([
            RandomHorizontalFlips(),
            ToTensors()
        ])
        

################################################################################
# Datasets
################################################################################
class XYDataset(Dataset):
    """A simple dataset returning examples of the form (transform(x), y)."""

    def __init__(self, data, transform=transforms.ToTensor(), memoize=False):
        """Args:
        data        -- a sequence of (x,y) pairs
        transform   -- the transform to apply to each returned x-value
        """
        super(XYDataset, self).__init__()
        self.transform = transform
        self.data = data

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return self.transform(image), label

class FeatureDataset(Dataset):
    """A dataset of model features.

    Args:
    F       -- a feature extractor, eg. the backbone of a ResNet
    data    -- a dataset of XY pairs
    bs      -- the batch size to use for feature extraction
    """

    def __init__(self, data, F, bs=1000, num_workers=24):
        super(FeatureDataset, self).__init__()
        loader = DataLoader(data, batch_size=bs, drop_last=False, pin_memory=True, num_workers=num_workers)

        data_x, data_y = [], []
        F = F.to(device)
        F.eval()
        with torch.no_grad():
            for x,y in tqdm(loader, desc="Building FeatureDataset", leave=False, file=sys.stdout):
                data_x.append(F(x.to(device)).cpu())
                data_y.append(y)

        data_x = [x for x_batch in data_x for x in x_batch]
        data_y = [y for y_batch in data_y for y in y_batch]
        self.data = list(zip(data_x, data_y))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return self.data[idx]

class ZippedDataset(Dataset):
    """A Dataset that zips together iterables. Its transform should be
    Args:
    *datasets   -- the wrapped iterables. All must be of the same length
    """
    def __init__(self, *datasets):
        super(ZippedDataset, self).__init__()
        self.datasets = datasets

        if not all([len(d) == len(datasets[0]) for d in datasets]):
            raise ValueError(f"All constituent datasets must have same length, but got lengths {[len(d) for d in datasets]}")

    def __len__(self): return len(self.datasets[0])

    def __getitem__(self, idx): return [d[idx] for d in self.datasets]

class CorruptedCodeYDataset(Dataset):
    """
    Args:
    cx      -- BSxCxHxW tensor of corrupted images
    codes   -- list of codes of shape BSxCODE_DIM. Elements in the list should
                be codes for sequentially greater resolutions
    ys      -- list of BSxCxHxW target images. Elements in the list should be
                for sequentially greater resolutions
    """
    def __init__(self, cx, codes, ys, expand_factor=1):
        super(CorruptedCodeYDataset, self).__init__()
        assert len(codes) == len(ys)
        assert all([len(c) == len(y) == cx.shape[0] for c,y in zip(codes, ys)])
        self.cx = cx.cpu()
        self.codes = [c.cpu() for c in codes]
        self.ys = [y.cpu() for y in ys]
        self.expand_factor = expand_factor
    
    def __len__(self): return len(self.cx) * self.expand_factor

    def __getitem__(self, idx):
        idx = idx // self.expand_factor
        cx = self.cx[idx]
        codes = [c[idx] for c in self.codes]
        ys = [y[idx] for y in self.ys]
        return cx, codes, ys


class GeneratorDataset(Dataset):
    """A dataset for returning data for generative modeling. Returns data in as

        model_input, [model_output_1, ... model_output_n]

    where [model_input] is half the resolution of [model_output_1], and
    [model_output_i] is half the resolution of [model_output_i+1]. All returned
    images are CxHxW.

    **Apply corruptions at the minibatch level in the training loop directly.**

    Args:
    datasets    -- list of ImageFolders containing training data at sequentially
                    doubling resolutions
    transform   -- transformation applied deterministically to both input and
                    target images
    """
    def __init__(self, datasets, transform, validate=False):
        self.datasets = datasets
        self.transform = transform

        ########################################################################
        # Validate the sequence of datasets. The H and W dimensions of
        ########################################################################
        if validate:
            tqdm.write("----- Validating GeneratorDataset -----")
            if not all([len(d) == len(self.datasets[0]) for d in self.datasets]):
                raise ValueError(f"All input datasets must have the same shape, but shapes were {[len(d) for d in self.datasets]}")

            shapes = [d[0][0].size for d in self.datasets]
            if len(self.datasets) > 2:
                for s1,s2 in zip(shapes[:-1], shapes[1:]):
                    if not s1[1] == s2[1] / 2 and  s1[2] == s2[2] / 2:
                        raise ValueError(f"Got sequential resolutions of {s1} and {s2}")
            else:
                tqdm.write(f"Shape sequence is {shapes}. Ensure that the generative model is correctly configred to use these.")

            self.shapes = [s[0] for s in shapes]
            tqdm.write(f"Validated source datasets: lengths {[len(d) for d in self.datasets]} | shape sequence {shapes}")

    def __len__(self): return len(self.datasets[0])

    def __getitem__(self, idx):
        images = [d[idx][0] for d in self.datasets]
        images = self.transform(images)
        return images[0], images[1:]

    def __repr__(self): return f"GeneratorDataset\n\tshapes {self.shapes}"

def collate_fn(data):
    """Collate function for batching data."""
    return torch.stack([d[0] for d in data], axis=0), [d[1] for d in data]

class ManyTransformsDataset(Dataset):
    """A dataset that wraps a single source dataset, but returns a tuple of
    transformed items from it, with the ith item coming from the ith transform
    in [transforms].

    Args:
    source_dataset  -- the source dataset
    *transforms     -- the transforms to use
    """
    def __init__(self, source_dataset, *transforms):
        super(ManyTransformsDataset, self).__init__()
        self.source_dataset = source_dataset
        self.transforms = transforms

    def __len__(self): return len(self.source_dataset)

    def __getitem__(self, idx):
        x = self.source_dataset[idx][0]
        return tuple([t(x) for t in self.transforms])

class PreAugmentedImageFolder(Dataset):
    """A drop-in replacement for an ImageFolder for use where some
    augmentations are pre-generated. It will behave differently as described
    below.

    Args:
    source              -- path to an folder of images laid out for an
                            ImageFolder. Files which differ by only an `_augN`
                            string are considered augmentations of each other.
    transform           -- transform to apply
    target_transform    -- target transform
    verbose             -- whether to print info about constructed dataset
    """
    def __init__(self, source, transform=None, target_transform=None,
        verbose=True):

        def remove_aug_info(s):
            """Returns string [s] without information indicating which
            augmentation it is. Concretely, this means that the `_augN` where
            `N` is some (possibly multi-digit) number substring is removed.

            This requires images to be named without breaking this function.
            """
            if "_aug" in s:
                underscore_idx = s.find("_")
                dot_idx = s.find(".")
                return f"{s[:underscore_idx]}{s[dot_idx]}"
            else:
                return s

        # Build a mapping from keys representing unique images to indices to the
        # files under [source] that are augmentations of that image
        image2idxs, counter = defaultdict(lambda: []), 0
        for c in tqdm(os.listdir(source), leave=False, desc="Buidling PreAugmentedImageFolder"):
            for image in os.listdir(f"{source}/{c}"):
                if os.path.splitext(image)[1].lower() in [".jpg", ".jpeg", ".png"]:
                    image2idxs[remove_aug_info(f"{c}/{image}")].append(counter)
                    counter += 1

        super(PreAugmentedImageFolder, self).__init__()
        self.data_idx2aug_idxs = [v for v in image2idxs.values() if len(v) > 0]
        self.data = ImageFolder(source,
            transform=transform,
            target_transform=target_transform)
        self.num_classes = len(os.listdir(source))

        # Print dataset statistics
        if verbose:
            aug_stats = [len(idxs) for idxs in self.data_idx2aug_idxs]
            data_name = f"{os.path.basename(os.path.dirname(source))}/{os.path.basename(source)}"
            s = f"Constructed PreAugmentedImageFolder over {data_name}. Length: {len(self.data_idx2aug_idxs)} | Min augmentations: {min(aug_stats)} | Average: {np.mean(aug_stats):.5f}| Max: {max(aug_stats)}"
            tqdm.write(s)

    def __len__(self): return len(self.data_idx2aug_idxs)

    def __getitem__(self, idx):
        return self.data[random.choice(self.data_idx2aug_idxs[idx])]
