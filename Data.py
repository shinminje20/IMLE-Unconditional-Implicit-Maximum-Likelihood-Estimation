import PIL
import random
import sys
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split

from torchvision.datasets import CIFAR10, ImageFolder
from torchvision import transforms

from Utils import *

no_val_split_datasets = ["cifar10"]
small_image_datasets = ["cifar10"]
dataset2input_dim = {
    "cifar10": (3, 32, 32),
    "imagenet": None,
}

dataset2n_classes = {
    "cifar10": 10,
    "imagenet": None,
}

def get_data_splits(data_str, eval_str):
    """Returns training and evaluation data given [data_str] and [eval_str].

    Args:
    data_str    -- a string specifying the dataset to return
    eval_str    -- the kind of validation to do. 'cv' for cross validation,
                    'val' for using a validation split (only if it exists), and
                    'test' for using the test split
    """
    # Get train, validation, and test splits of the data
    if data_str == "cifar10":
        data_tr = CIFAR10(root=f"{project_dir}/Data", train=True, download=True)
        data_val = None
        data_te = CIFAR10(root=f"{project_dir}/Data", train=False, download=True)
    elif data_str == "miniImagenet":
        data_tr = ImageFolder(root=f"{project_dir}/Data/miniImagenet/train")
        data_val = ImageFolder(root=f"{project_dir}/Data/miniImagenet/val")
        data_te = ImageFolder(root=f"{project_dir}/Data/miniImagenet/default_test")
    else:
        raise ValueError(f"Unknown dataset {data_str}")

    if eval_str == "cv":
        eval_data = "cv"
    elif eval_str == "val":
        eval_data = data_val
    elif eval_str == "test":
        eval_data = data_te

    return data_tr, eval_data

################################################################################
# NData augmentations
################################################################################
def get_isicle_data_augs(in_size=32, out_size=128):
    """Returns data augmentations for ISICLE training. No augmentations here
    should take data off the real manifold!

    Args:
    in_size     -- size of images input to generator
    out_size    -- size of images that come out of generator
    """
    augs_tr = transforms.Compose([
        transforms.RandomResizedCrop(in_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()])
    augs_fn = transforms.Compose([
        transforms.RandomResizedCrop(out_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()])
    augs_te = transforms.Compose([
        transforms.RandomResizedCrop(out_size),
        transforms.ToTensor()])
    return augs_tr, augs_fn, augs_te

def get_ssl_data_augs(data_str, color_s=.5):
    """Returns a (SSL transforms, finetuning transforms, testing transforms)
    tuple based on [data_str].

    Args:
    data_str        -- a string specifying the dataset to get transforms for
    color_distorr   -- the strength of color distortion
    """
    color_jitter = transforms.ColorJitter(0.8 * color_s,
         0.8 * color_s, 0.8 * color_s, 0.2 * color_s)
    color_distortion = transforms.Compose([
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2)])

    if "cifar" in data_str:
        augs_tr = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            color_distortion,
            transforms.ToTensor()])

        # Validate using the same augmentations as for training, in line with
        # https://github.com/leftthomas/SimCLR
        augs_fn = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            color_distortion,
            transforms.ToTensor()])

        augs_te = transforms.Compose([
            transforms.ToTensor()])
    elif "miniImagenet" in data_str:
        augs_tr = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(p=0.5),
            color_distortion,
            transforms.GaussianBlur(23, sigma=(.1, 2)),
            transforms.ToTensor()])

        # Validate using the same augmentations as for training, in line with
        # https://github.com/leftthomas/SimCLR
        augs_fn = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(p=0.5),
            color_distortion,
            transforms.GaussianBlur(23, sigma=(.1, 2)),
            transforms.ToTensor()])

        augs_te = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.ToTensor()])
    else:
        raise ValueError("Unknown augmenta")

    return augs_tr, augs_fn, augs_te

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

class ImagesFromTransformsDataset(Dataset):
    """Wraps an internal dataset for which queries produce a tuple in which an
    image is the first result. This dataset computes transformations [x] and
    [y]; the object for a neural net using the generated data is to generate [y]
    conditioned on [x] and a random vector z, or to contrast [x] and [y].

    [x] might be a downsampled and greyscaled version of the image, while [y]
    might be a random crop. This trains a generator to simulteneously do
    super-resolution, colorization, and cropping.
    """
    def __init__(self, data, x_transform, y_transform):
        """
        Args:
        data        -- the wrapped dataset, should return tuples wherein the
                        first element is an image
        x_transform -- the transform giving input images
        y_transform -- the transform giving targets
        """
        super(ImagesFromTransformsDataset, self).__init__()
        self.data = data
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        return self.x_transform(image), self.y_transform(image)

class FeatureDataset(Dataset):
    """A dataset of model features."""

    def __init__(self, data, F, bs=500):
        """Args:
        F       -- a feature extractor
        data    -- a dataset of XY pairs
        bs      -- the batch size to use for feature extraction
        """
        super(FeatureDataset, self).__init__()
        loader = DataLoader(data, batch_size=bs, drop_last=False)

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
