"""Code for building datasets.

See data/SetupDataset.py for downloading their underlying data.

The important portions of this file are get_data_splits(), which returns
training and evaluation ImageFolders, and the various Dataset subclasses that
can be used to construct various useful datasets.
"""
from collections import OrderedDict
import numpy as np
import PIL
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
# Dataset medatata. In general, unless it's properly logged here, you can use a
# new dataset. If it is logged here, using it should be easy!
#
# To make running on a dataset possible, ensure images in it can be accessed via
# `data/dataset_name/split/class/image.png` where `dataset_name` doesn't include
# resolution information, ie. an actual path would be
#
#   `data/cifar10_16x16/train/airplane/image.png`
#
# but only `cifar10` would be recorded here.
################################################################################
datasets = ["strawberry", "cifar10", "camnet3", "camnet3_deci", "camnet3_centi", "camnet3_milli", "miniImagenet", "miniImagenet_deci"]
no_val_split_datasets = ["cifar10"]
small_image_datasets = ["cifar10"]
data2split2n_class = {
    "camnet3": {"train": 3, "val": 3},
    "cifar10": {"train": 10, "val": 10, "test": 10},
    "miniImagenet": {"train": 64, "val": 16, "test": 20} # Not sure if these numbers are correct
}

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

def get_data_splits(data_str, eval_str, res=None, data_folder_path=f"{project_dir}/data"):
    """Returns data for training and evaluation. All Datasets returned are
    ImageFolders, meaning that another kind of dataset likely needs to be built
    on top of them.

    Args:
    data_str    -- string specifying the dataset to load. It must exactly exist
                    in the data directory, ie. data/data_str exists.
    eval_str    -- how to get validation/testing data
    resolutions -- resolutions in which to return data
                    - leave empty (no arguments) to get data for plain
                        contrastive learning
                    - if specified, validation data will include the minimum and
                        maximum resolutions
    """
    ############################################################################
    # CIFAR-10 has its own weird logic
    ############################################################################
    if data_str == "cifar10":
        if eval_str == "test":
            return (CIFAR10(root=data_folder_path, train=True, transform=None,
                            download=True),
                    CIFAR10(root=data_folder_path, train=False, transform=None,
                            download=True))
        else:
            return (CIFAR10(root=data_folder_path, train=True, transform=None,
                            download=True),
                    CIFAR10(root=data_folder_path, train=True, transform=None,
                            download=True))

    def paths_to_datasets(data_paths):
        """Returns ImageFolder(s) given [data_paths], a list of data paths for a
        list of ImageFolders, or a string for a single ImageFolder.
        """
        if len(data_paths) > 1 and isinstance(data_paths, (OrderedDict, list)):
            return [ImageFolder(data_path) for data_path in data_paths]
        elif isinstance(data_paths, (OrderedDict, list)):
            return ImageFolder(list(data_paths)[0])
        else:
            return ImageFolder(data_paths)

    data_path = f"{data_folder_path}/{data_str}"

    if eval_str == "test":
        eval_split_specifier = "test"
    elif eval_str == "val":
        eval_split_specifier = "val"
    elif eval_str == "cv":
        eval_split_specifier = "train"

    if res is None:
        data_paths_tr = f"{data_path}/train"
        data_paths_eval = f"{data_path}/{eval_split_specifier}"
    else:
        data_paths_tr = [f"{data_path}_{r}x{r}/train" for r in res]
        data_paths_eval = [
             f"{data_path}_{min(res)}x{min(res)}/{eval_split_specifier}",
             f"{data_path}_{max(res)}x{max(res)}/{eval_split_specifier}"]

        data_paths_tr = remove_duplicates(data_paths_tr)
        data_paths_eval = remove_duplicates(data_paths_tr)

        check_paths_exist([data_paths_tr, data_paths_eval])
    return paths_to_datasets(data_paths_tr), paths_to_datasets(data_paths_eval)

################################################################################
# Augmentations
################################################################################


def get_simclr_augs(crop_size=32, gaussian_blur=False, color_s=.5):
    """Returns a (SSL transforms, finetuning transforms, testing transforms)
    tuple based on [data_str].

    Args:
    data_str  -- a string specifying the dataset to get transforms for
    color_s   -- the strength of color distortion
    strong    -- whether to use strong augmentations or not
    """
    color_jitter = transforms.ColorJitter(0.8 * color_s,
         0.8 * color_s, 0.8 * color_s, 0.2 * color_s)
    color_distortion = transforms.Compose([
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2)])

    augs_tr_list = [transforms.RandomResizedCrop(crop_size)]

    if color_s > 0:
        augs_tr_list.append(color_distortion)
    if gaussian_blur:
        augs_tr_list.append(transforms.GaussianBlur(23, sigma=(.1, 2)))

    augs_tr_list += [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ]

    augs_tr = transforms.Compose(augs_tr_list)

    augs_te = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.ToTensor()])

    return augs_tr, augs_tr, augs_te

def get_gen_augs():
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

class XDataset(Dataset):
    """A wrapper over an iterable."""
    def __init__(self, data, transform=None):
        super(XDataset, self).__init__()
        self.data = data
        self.transform = transform

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        if self.transform is None:
            return self.data[idx]
        else:
            return self.transform(self.data[idx])

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

class CorruptedDataset(Dataset):
    """Dataset for returning corrupted images and their targets. The main
    difference from loading images from a GeneratorDataset and corrupting images
    on the fly is that this keeps the corruptions static, which is useful due to
    how we find the best codes for each image. Corrupted images don't change in
    between different iterations of code sampling.

    Args:
    source      -- source dataset, should be a GeneratorDataset
    corruptor   -- corruptor to corrupt a batch of images
    bs          -- batch size for getting corrupted versions of images
    """
    def __init__(self, source, corruptor):
        super(CorruptedDataset, self).__init__()
        loader = DataLoader(source, num_workers=num_workers, batch_size=8,
            collate_fn=collate_fn, pin_memory=True)
        self.corrupted_xs = []
        self.ys = []

        for x,y in loader:
            x = corruptor(x.to(device, non_blocking=True))
            self.corrupted_xs.append(x)
            self.ys += y

        self.ys = make_cpu(self.ys)
        self.corrupted_xs = torch.cat(self.corrupted_xs, axis=0).cpu()

    def __len__(self): return len(self.corrupted_xs)
    def __getitem__(self, idx): return self.corrupted_xs[idx], self.ys[idx]

class ExpandedDataset(Dataset):
    """Provides a view of [source_data] expanded by a factor of [expand_factor].
    More formally, the indices i * expand_factor ... (i+1) * expand_factor - 1
    map from the ith example of [source_data].
    """
    def __init__(self, source_data, expand_factor=1):
        self.source = source_data
        self.expand_factor = expand_factor

    def __len__(self): return self.expand_factor * len(self.source)

    def __getitem__(self, idx): return self.source[idx // self.expand_factor]


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
