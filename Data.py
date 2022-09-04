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
from torchvision.datasets.folder import default_loader
from torchvision import transforms

from Corruptions import *
from utils.Utils import *
from Augmentations import *
import glob
def is_valid_data(data_str):
    """Returns [data_str] if it is a valid data string, or False otherwise."""
    if os.path.dirname(data_str) in datasets:
        return data_str
    elif data_str.startswith("generated_data") or data_str.startswith("cifar") or data_str == "cv":
        return data_str
    else:
        raise argparse.ArgumentTypeError(f"--data can be one of {datasets} or a file in 'generated_data'")

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
    "cub": {"splits": ["train", "val", "test"],
        "res": [32, 64, 128, 256],
        "same_distribution_splits": True},
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
def get_imagefolder_data(*datasets, res=32, data_path=data_dir):
    """Returns a tranform-free dataset for each path in [*datasets].
    Args:
    datasets            -- list of strings that can be interpreted as a dataset,
                            or a path to a folder that can be interpreted as a
                            PreAugmentedDataset
    res                 -- list of resolutions or a list resolution interpreted
                            as a single-item list
    data_path           -- path to where datasets are found
    Returns:
    Roughly, [[d(p, r) for r in res] for p in datasets], where [d(.,.)] maps
    a string identifying a folder of data and a resolution to a PyTorch dataset
    over the data. If [res] contains only one item, the sublist is flattened.
    """
    ignored_data_strs = ["cifar10/train", "cifar10/test", "cv"]

    def contains_augs(data_str):
        """Returns if any images in [data_str] are augmentations."""
        for label in os.listdir(data_str):
            for image in os.listdir(f"{data_str}/{label}"):
                if "_aug" in image:
                    return True
        return False

    def data_str_with_resolution(data_str, res):
        """Returns [data_str] at resolution [res].
        Args:
        data_str -- a path to something that could be turned into an ImageFolder
        res     -- the desired resolution
        """
        if (data_str in ignored_data_strs
            or has_resolution(data_str)
            or os.path.exists(data_str)):
            return data_str
        else:
            dataset = data_str.strip("/")
            idx = data_str.rindex("/")
            return data_str[:idx] + f"_{res}x{res}" + data_str[idx:]

    def data_str_to_dataset(data_str):
        """Returns the dataset that can be built with [data_str].
        Args:
        data_str    -- One of 'cv', 'cifar10/train', 'cifar10/test', or a path
                        to a folder over which an ImageFolder can be constructed
        """
        if data_str == "cifar10/train":
            return CIFAR10(root=data_path, train=True, download=True)
        elif data_str == "cifar10/test":
            return CIFAR10(root=data_path, train=False, download=True)
        elif data_str == "cv":
            return "cv"
        else:
            if os.path.exists(data_str):
                data_str = data_str
            elif os.path.exists(f"{data_path}/{data_str}"):
                data_str = f"{data_path}/{data_str}"
            else:
                raise ValueError(f"Couldn't find a folder for data string {data_str}")

            if contains_augs(data_str):
                return PreAugmentedDataset(data_str, verbose=False)
            else:
                return ImageFolder(data_str)

    res = [res] if isinstance(res, int) else res
    datasets = [[data_str_with_resolution(d, r) for r in res] for d in datasets]
    datasets = [[data_str_to_dataset(d) for d in d_] for d_ in datasets]
    result = tuple([d[0] if len(d) == 1 else d for d in datasets])
    return result[0] if len(result) == 1 else result

def get_imagefolder_IMLE_data(*datasets, res=32, data_path=data_dir):
    """Returns a tranform-free dataset for each path in [*datasets].

    Args:
    datasets            -- list of strings that can be interpreted as a dataset,
                            or a path to a folder that can be interpreted as a
                            PreAugmentedDataset
    res                 -- list of resolutions or a list resolution interpreted
                            as a single-item list
    data_path           -- path to where datasets are found

    Returns:
    Roughly, [[d(p, r) for r in res] for p in datasets], where [d(.,.)] maps
    a string identifying a folder of data and a resolution to a PyTorch dataset
    over the data. If [res] contains only one item, the sublist is flattened.
    """
    ignored_data_strs = ["cifar10/train", "cifar10/test", "cv"]

    def contains_augs(data_str):
        """Returns if any images in [data_str] are augmentations."""
        for label in os.listdir(data_str):
            for image in os.listdir(f"{data_str}/{label}"):
                if "_aug" in image:
                    return True
        return False

    def data_str_with_resolution(data_str, res):
        """Returns [data_str] at resolution [res].

        Args:
        data_str -- a path to something that could be turned into an ImageFolder
        res     -- the desired resolution
        """
        if (data_str in ignored_data_strs
            or has_resolution(data_str)
            or os.path.exists(data_str)):
            return data_str
        else:
            dataset = data_str.strip("/")
            idx = data_str.rindex("/")
            return data_str[:idx] + f"_{res}x{res}" + data_str[idx:]

    def data_str_to_dataset(data_str):
        """Returns the dataset that can be built with [data_str].

        Args:
        data_str    -- One of 'cv', 'cifar10/train', 'cifar10/test', or a path
                        to a folder over which an ImageFolder can be constructed
        """
        if data_str == "cifar10/train":
            return CIFAR10(root=data_path, train=True, download=True)
        elif data_str == "cifar10/test":
            return CIFAR10(root=data_path, train=False, download=True)
        elif data_str == "cv":
            return "cv"
        else:
            if os.path.exists(data_str):
                data_str = data_str
            elif os.path.exists(f"{data_path}/{data_str}"):
                data_str = f"{data_path}/{data_str}"
            else:
                raise ValueError(f"Couldn't find a folder for data string {data_str}")

            if contains_augs(data_str):
                return PreAugmentedDataset(data_str, verbose=False)
            else:
                return data_str

    res = [res] if isinstance(res, int) else res
    datasets = [[data_str_with_resolution(d, r) for r in res] for d in datasets]
    datasets = [[data_str_to_dataset(d) for d in d_] for d_ in datasets]
    result = tuple([d[0] if len(d) == 1 else d for d in datasets])
    return (result[0], result[1]) if len(result) == 1 else result

################################################################################
# Datasets
################################################################################
class XYDataset(Dataset):
    """A simple dataset returning examples of the form (transform(x), y)."""

    def __init__(self, data, transform=transforms.ToTensor()):
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
        loader = DataLoader(data,
            batch_size=bs,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers)

        data_x, data_y = [], []
        F = F.to(device)
        F.eval()
        with torch.no_grad():
            for x,y in tqdm(loader,
                desc="Building FeatureDataset",
                leave=False):
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
    """Dataset that returns a (cx, codes, y) tuple where [cx] is a CxHxW
    corrupted image, [codes] is a list of codes for generating decorruptions of
    the image at progressively higher resolutions, and [y] is a list of the
    original image a progressively higher resolutions.

    To iterate over this dataset more than once, construct a DataLoader over it,
    and use the itertools.chain() method on a list of this DataLoader,
    duplicated a bunch.

    Args:
    cx      -- BSxCxHxW tensor of corrupted images
    codes   -- list of codes of shape BSxCODE_DIM. Elements in the list should
                be codes for sequentially greater resolutions
    ys      -- list of BSxCxHxW target images. Elements in the list should be
                for sequentially greater resolutions
    """
    def __init__(self, cx, codes, ys):
        super(CorruptedCodeYDataset, self).__init__()
        assert len(codes) == len(ys)
        assert all([len(c) == len(y) == cx.shape[0] for c,y in zip(codes, ys)])

        self.cx = cx.cpu()
        self.codes = [c.cpu() for c in codes]
        self.ys = [y.cpu() for y in ys]

    def __len__(self): return len(self.cx)

    def __getitem__(self, idx):
        cx = self.cx[idx]
        codes = [c[idx] for c in self.codes]
        ys = [y[idx] for y in self.ys]
        return cx, codes, ys

class InpaintingCorruptedCodeYDataset(Dataset):
    """Dataset that returns a (cx, codes, y) tuple where [cx] is a CxHxW
    corrupted image, [codes] is a list of codes for generating decorruptions of
    the image at progressively higher resolutions, and [y] is a list of the
    original image a progressively higher resolutions.

    To iterate over this dataset more than once, construct a DataLoader over it,
    and use the itertools.chain() method on a list of this DataLoader,
    duplicated a bunch.

    Args:
    cx      -- BSxCxHxW tensor of corrupted images
    codes   -- list of codes of shape BSxCODE_DIM. Elements in the list should
                be codes for sequentially greater resolutions
    ys      -- list of BSxCxHxW target images. Elements in the list should be
                for sequentially greater resolutions
    """
    def __init__(self, cx, codes, ys):
        super(InpaintingCorruptedCodeYDataset, self).__init__()
        assert len(codes) == len(ys)
        assert all([len(c) == len(y[0]) == cx.shape[0] for c,y in zip(codes, ys)])

        self.cx = make_cpu(cx)
        self.codes = [c.cpu() for c in codes]
        self.ys = make_cpu(ys)

    def __len__(self): return len(self.cx)

    def __getitem__(self, idx):
        cx = self.cx[idx]
        codes = [c[idx] for c in self.codes]
        ys = [y[0][idx] for y in self.ys]
        masks = [y[1][idx] for y in self.ys]
        return cx, codes, ys, masks

    @staticmethod
    def collate_fn(num_levels, samples):
        """Returns the batched up [samples]. [samples] should be a list where
        the ith element is a (x, [target at each level], [mask at each level])
        tuple. [num_levels] is the number of levels to return targets for.

        This function is best used as a partial function fixing [num_levels].
        """
        x = torch.stack([s[0] for s in samples])
        c = [torch.stack([s[1][level_idx] for s in samples])
            for level_idx in range(num_levels)]
        y = [torch.stack([s[2][level_idx] for s in samples])
            for level_idx in range(num_levels)]
        m = [torch.stack([s[3][level_idx] for s in samples])
            for level_idx in range(num_levels)]
        return x, c, list(zip(y, m))

class GeneratorDataset(Dataset):
    """A dataset for returning data for generative modeling. Returns data in as

        model_input, [model_output_1, ... model_output_n]

    where [model_input] is half the resolution of [model_output_1], and
    [model_output_i] is half the resolution of [model_output_i+1]. All returned
    images are CxHxW.

    **Apply corruptions at the minibatch level in the training loop directly.**

    Args:
    datasets    -- list of ImageFolders containing training data at sequentially
                    doubling or the same resolutions
    transform   -- transformation applied deterministically to both input and
                    target images
    """
    def __init__(self, datasets, transform=lambda x: x, validate=False):
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

    def to_val_dataset(self): return self

class IMLEDataset(Dataset):
    """A dataset for returning data for generative modeling. Returns data in as

        model_input, [model_output_1, ... model_output_n]

    where [model_input] is half the resolution of [model_output_1], and
    [model_output_i] is half the resolution of [model_output_i+1]. All returned
    images are CxHxW.

    **Apply corruptions at the minibatch level in the training loop directly.**

    Args:
    datasets    -- list of ImageFolders containing training data at sequentially
                    doubling or the same resolutions
    transform   -- transformation applied deterministically to both input and
                    target images
    """
    def __init__(self, datasets, transform=lambda x: x, validate=False):

        img_name = datasets + '/n02279972/*'
        images = [image for image in glob.glob(img_name)]
        
        self.datasets = images
        self.transform = transforms.Compose([
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                            ])

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

    def __len__(self): return len(self.datasets)

    def __getitem__(self, idx):
        # images = [d[idx][0] for d in self.datasets]
        # images = self.transform(images)
        # return images[0]
        from PIL import Image

        # read the image:
        image_name = self.datasets[idx]
        img = Image.open(image_name).convert('RGB')

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        if img.shape[0] >= 4:
            # ignore the alpha channel
            # in the image if it exists
            img = img[:3, :, :]

        # return the image:
        return img

    # def __repr__(self): return f"GeneratorDataset\n\tshapes {self.shapes}"

    def to_val_dataset(self): return self

class InpaintingGeneratorDataset(Dataset):
    """Dataset for inpainting images. This is unfortunately complicated. For
    inpainting, the model expects an (x, y) tuple where [x] is a BSxCxHxW tensor
    of images with certain portions masked out, and [y] is a
    (target_images, loss_masks) tuple. [target_images] is a list where the ith
    element is a BSxCxHxW tensor giving target images at the ith resolution, and
    [loss_masks] is a list where the ith element is a BSx1xHxW binary tensor
    giving where loss should be taken in the target images at the ith
    resolution. In these masks, a zero indicates a region where loss should be
    taken and a one a region where the loss should be ignored by setting model
    outputs to the ground truth value.

    The __getitem__() method returns an (x, y) identical to the one described
    above, but with the batch dimension removed, so all tensors are order three.

    NOTE: This dataset must be used with its collate_fn(). The normal one does
    work correctly with what the loss function expects!

    Args:
    datasets    -- list of ImageFolders containing training data at sequentially
                    doubling or the same resolutions
    transform   -- transformation applied deterministically to both input and
                    target images
    mask_size   -- number of pixels per mask size in the first resolution
    border_size -- number of pixels for the border around all possible masks in
                    the first resolution
    """
    def __init__(self, datasets, transform=lambda x: x, mask_size=12, border_size=2):
        super(InpaintingGeneratorDataset, self).__init__()
        self.datasets = datasets
        self.transform = transform
        self.border_size = border_size
        self.mask_size = mask_size
        self.res = [transforms.ToTensor()(d[0][0]).shape[-1] for d in self.datasets]

    def __len__(self): return len(self.datasets[0])

    def __getitem__(self, idx):
        images = [d[idx][0] for d in self.datasets]
        images = self.transform(images)
        images, masks = self.inpainting_fn(images)
        return images[0], images[1:], masks[1:]

    def inpainting_fn(self, images):
        """
        Args:

        """
        ul = torch.randint(self.border_size, self.res[0] - self.mask_size - self.border_size * 2, (2,))
        ul = [ul * (r // self.res[0]) for r in self.res]
        ul = [u + (self.mask_size * (r // self.res[0]) * (idx / (len(ul) - 1)) / 2)
            for idx,(r,u) in enumerate(zip(self.res, ul))]

        lr = [u + (self.mask_size * (r // self.res[0]) * (1 - (idx / (len(ul) - 1))))
            for idx,(r,u) in enumerate(zip(self.res, ul))]

        (uly,ulx),(lry,lrx) = ul[0], lr[0]

        uly = int(uly.item())
        ulx = int(ulx.item())
        lry = int(lry.item())
        lrx = int(lrx.item())

        images[0][:, uly:lry, ulx:lrx] = 0

        masks = [torch.zeros(1, r, r) for r in self.res]
        for m,(uly,ulx),(lry,lrx) in zip(masks, ul, lr):
            uly = int(uly.item())
            ulx = int(ulx.item())
            lry = int(lry.item())
            lrx = int(lrx.item())
            m[:, uly:lry, ulx:lrx] = 1

        masks = [m.bool() for m in masks]

        return images, masks

    def to_val_dataset(self):
        return ValidationInpaintingGeneratorDataset(
            self.datasets,
            transform=self.transform,
            border_size=self.border_size,
            mask_size=self.mask_size)

    @staticmethod
    def collate_fn(num_levels, samples):
        """Returns the batched up [samples]. [samples] should be a list where
        the ith element is a (x, [target at each level], [mask at each level])
        tuple. [num_levels] is the number of levels to return targets for.

        This function is best used as a partial function fixing [num_levels].
        """
        x = torch.stack([s[0] for s in samples])
        y = [torch.stack([s[1][level_idx] for s in samples])
            for level_idx in range(num_levels)]
        m = [torch.stack([s[2][level_idx] for s in samples])
            for level_idx in range(num_levels)]
        return x, list(zip(y, m))


    def __repr__(self): return f"InpaintingGeneratorDataset\n\tshapes {self.shapes}"

class ValidationInpaintingGeneratorDataset(InpaintingGeneratorDataset):
    """Identical to a InpaintingGeneratorDataset, but __getitem__() doesn't
    return masks.
    """
    def __init__(self, datasets, transform=lambda x: x, mask_size=12, border_size=2):
        super(ValidationInpaintingGeneratorDataset, self).__init__(datasets,
            transform=transform,
            mask_size=mask_size,
            border_size=border_size)

    def __getitem__(self, idx):
        images = [d[idx][0] for d in self.datasets]
        images = self.transform(images)
        images, _ = self.inpainting_fn(images)
        return images[0], images[1:]

def collate_fn(data):
    """Collate function for batching data."""
    return torch.stack([d[0] for d in data], axis=0), [d[1] for d in data]

class ManyTransformsDataset(Dataset):
    """A dataset that wraps a single source dataset, but returns a tuple of
    transformed items from it, with the ith item coming from the ith transform
    in [transforms].

    Args:
    source  -- the source dataset
    *transforms     -- the transforms to use
    """
    def __init__(self, data, *transforms):
        super(ManyTransformsDataset, self).__init__()
        self.data = data
        self.transforms = transforms

        if (isinstance(self.data, PreAugmentedDataset)
            and self.data.mode == "cl"
            and self.data.num_augs == len(self.transforms)):
            self.pre_augmented_cl = True
        elif (isinstance(self.data, PreAugmentedDataset)
            and self.data.mode == "cl"
            and not self.data.num_augs == len(self.transforms)):
            raise ValueError(f"Requested {len(self.transforms)}, but the wrapped PreAugmentedImagefolder will return {self.data.num_augs} images")
        else:
            self.pre_augmented_cl = False

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        if self.pre_augmented_cl:
            return tuple([t(x) for x,t in zip(self.data[idx], self.transforms)])
        else:
            x,_ = self.data[idx]
            return tuple([t(x) for t in self.transforms])


class PreAugmentedDataset(ImageFolder):
    """A drop-in replacement for an ImageFolder for use where some
    augmentations are pre-generated.

    Suppose we have a folder organized so an ImageFolder could read from it.
    Within each class, files that differ by only an 'augN' string are considered
    augmentations of each other, and shouldn't be returned in the same epoch. Examples of such files are 'butterfly_aug8.png' and 'butterfly_aug3.png'.

    Args:
    source              -- path to an folder of images laid out for an
                            ImageFolder. Files which differ by only an `_augN`
                            string are considered augmentations of each other.
    transform           -- transform to apply
    target_transform    -- target transform
    verbose             -- whether to print info about constructed dataset
    """
    def __init__(self, source, transform=None, target_transform=None,
        num_augs=1, mode="xy", verbose=True):

        def remove_aug_info(s):
            """Returns string [s] without information indicating which
            augmentation it is. Concretely, this means that the `_augN` where
            `N` is some (possibly multi-digit) number substring is removed.

            This requires images to be named without breaking this function.
            """
            if "_aug" in s:
                underscore_idx = s.rfind("_")
                dot_idx = s.rfind(".")
                return f"{s[:underscore_idx]}{s[dot_idx]}"
            else:
                return s

        super(PreAugmentedDataset, self).__init__(source,
            target_transform=target_transform)

        self.source = source
        self.num_augs = num_augs
        self.mode = mode
        self.transform = (lambda x: x) if transform is None else transform
        self.target_transform = target_transform
        self.image2aug_idxs = defaultdict(lambda: [])
        for idx,(path,target) in enumerate(self.samples):
            self.image2aug_idxs[remove_aug_info(path)].append(idx)
        self.image_idx2aug_idxs = [v for v in self.image2aug_idxs.values()]
        self.verbose = verbose

        if self.verbose:
            tqdm.write(str(self))

    @staticmethod
    def get_cl_mode(p, num_augs=2, verbose=True):
        """Returns a PreAugmentedDataset identical to PreAugmentedDataset [p]
        but with the mode set for contrastive learning. If [p] isn't a
        PreAugmentedDataset, returns [p] with not change.
        """
        if isinstance(p, PreAugmentedDataset):
            return PreAugmentedDataset(p.source,
                transform=p.transform,
                target_transform=p.target_transform,
                num_augs=num_augs,
                mode="cl",
                verbose=verbose)
        else:
            return p

    @staticmethod
    def get_xy_mode(p, num_augs=None, verbose=True):
        """Returns a PreAugmentedDataset identical to PreAugmentedDataset [p]
        but with the mode set for supervised learning. If [p] isn't a
        PreAugmentedDataset, returns [p] with not change. [num_augs] is ignored.
        """
        if isinstance(p, PreAugmentedDataset):
            return PreAugmentedDataset(p.source,
                transform=p.transform,
                target_transform=p.target_transform,
                num_augs=1,
                mode="xy",
                verbose=verbose)
        else:
            return p

    def __str__(self):
        length = len(self.image_idx2aug_idxs)
        min_augs = min([len(augs) for augs in self.image_idx2aug_idxs])
        avg_augs = np.mean([len(augs) for augs in self.image_idx2aug_idxs])
        max_augs = max([len(augs) for augs in self.image_idx2aug_idxs])
        return f"PreAugmentedDataset: source {self.source}\n\tlength {length} | min augs {min_augs} | avg augs {avg_augs:.2f} | max augs {max_augs} | mode {self.mode}"

    def __len__(self): return len(self.image_idx2aug_idxs)

    def __getitem__(self, idx):
        if self.mode == "cl":
            xs = random.sample(self.image_idx2aug_idxs[idx], k=self.num_augs)
            xs = [default_loader(self.samples[x][0]) for x in xs]
            return tuple([self.transform(x) for x in xs])
        else:
            idx = random.choice(self.image_idx2aug_idxs[idx])
            x, y = self.samples[idx]
            return self.transform(default_loader(x)), y