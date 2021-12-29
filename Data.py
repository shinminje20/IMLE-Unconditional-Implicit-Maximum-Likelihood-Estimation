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

def seed_kwargs(seed=0):
    """Returns kwargs to be passed into a DataLoader to give it seed [seed]."""
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    return {"generator": g, "worker_init_fn": seed_worker}

def get_data_splits_ssl(data_str, eval_str):
    """Returns training and evaluation data given [data_str] and [eval_str].

    Args:
    data_str    -- a string specifying the dataset to return
    eval_str    -- the kind of validation to do. 'cv' for cross validation,
                    'val' for using a validation split (only if it exists), and
                    'test' for using the test split
    """
    if data_str == "cifar10":
        data_tr = CIFAR10(root=f"{project_dir}/data", train=True, download=True)
        data_val = None
        data_te = CIFAR10(root=f"{project_dir}/data", train=False, download=True)
    elif data_str == "miniImagenet":
        data_tr = ImageFolder(root=f"{project_dir}/data/miniImagenet/train")
        data_val = ImageFolder(root=f"{project_dir}/data/miniImagenet/val")
        data_te = ImageFolder(root=f"{project_dir}/data/miniImagenet/default_test")
    else:
        raise ValueError(f"Unknown dataset {data_str}")

    if eval_str == "cv":
        eval_data = "cv"
    elif eval_str == "val":
        eval_data = data_val
    elif eval_str == "test":
        eval_data = data_te

    return data_tr, eval_data

def get_data_splits_gen(data_str, eval_str, resolutions=[16, 32, 64, 128, 256]):
    """Returns training and evaluation data given [data_str] and [eval_str].
    The training and evaluation data are each returned as a list where each
    element is a dataset with the data at a given resolution.

    Args:
    data_str    -- a string specifying the dataset to return
    eval_str    -- the kind of validation to do. 'cv' for cross validation,
                    'val' for using a validation split (only if it exists), and
                    'test' for using the test split
    """
    if data_str == "miniImagenet":
        data_tr = OrderedDict(
            {r: ImageFolder(root=f"{project_dir}/data/miniImagenet_{r}/train") for r in resolutions})
        data_val = OrderedDict(
            {r: ImageFolder(root=f"{project_dir}/data/miniImagenet_{r}/val") for r in resolutions})
        data_te = OrderedDict(
            {r: ImageFolder(root=f"{project_dir}/data/miniImagenet_{r}/test") for r in resolutions})
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
# Data augmentations
################################################################################

class RandomHorizontalFlips(torch.nn.Module):
    """transforms.RandomHorizontalFlip but can be applied to multiple images."""
    def __init__(self, p=0.5):
        super(RandomHorizontalFlips, self).__init__()
        self.p = p

    def forward(self, images):
        """Returns [images] but with all elements flipped in the same direction,
        with the direction chosen randomly.

        Args:
        images  -- list of (PIL Image or Tensor): Images to be flipped
        """
        if torch.rand(1) < self.p:
            return [F.hflip(img) for img in images]
        return [images]

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def get_ssl_augs(data_str, color_s=.5, strong=True):
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

    if "cifar" in data_str:
        if strong:
            augs_tr = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                color_distortion,
                transforms.ToTensor()])
        else:
            augs_tr = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()])

        augs_fn = augs_tr
        augs_te = transforms.Compose([transforms.ToTensor()])
    elif "imagenet" in data_str.lower():
        if strong:
            augs_tr = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                color_distortion,
                transforms.GaussianBlur(23, sigma=(.1, 2)),
                transforms.ToTensor()])
        else:
            augs_tr = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                 transforms.ToTensor()])

        augs_fn = augs_tr
        augs_te = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor()])
    else:
        raise ValueError("Unknown augmenta")

    return augs_tr, augs_fn, augs_te

################################################################################
# Datasets
################################################################################
class MultiTaskDataset(Dataset):
    """A dataset for forcing a model a generative model to perform multiple
    tasks. Right now, only colorization + super-resolution are supported.
    """
    def __init__(self, res2data, task_transform, base_transform,
        intermediate_supervision=True):
        """
        Args:
        data                        -- list of input datasets. The first item in
                                        the list is a dataset giving the images
                                        fed to the model after corruption,
                                        remaining datasets specify sequentially
                                        higher resolutions of the original image
                                        from the first
        task_transform              -- transform giving the (x, (y1 ... yn))
                                        images—the model must decorrupt [x] to
                                        the sequence of ys
        base_transform              -- transform applied to all images in the
                                        (x, (y1 ... yn)) sequence returned from
                                        __getitem__(). This transform must be
                                        able to apply to a list of images, and
                                        be deterministic across this list.
        intermediate_supervision    -- whether to return targets for supervising
                                        the model at different resolutions
        """
        assert all([len(d) == len(data[0]) for d in data]), f"All input datasets must have equal length, but lengths were {[len(d) for d in data]}"
        self.data = data
        self.task_transform = task_transform
        self.base_transform = base_transform
        self.intermediate_supervision = intermediate_supervision

    def task_transform(self, idx):
        if self.task_transform == "col_sr":
            return self.data[0][idx][0], [d[idx][0] for for d in self.data[1:]]
        else:
            raise ValueError(f"Unknown task transform '{self.task_transform}'")

    def __len__(self): return len(self.data[0])

    def __getitem(self, idx):
        if self.intermediate_supervision:
            x,ys = task_transform(idx)
            transformed_images = self.base_transform([x] + ys)
            return (self.tensor_transforms(transformed_images[0]),
                    [self.tensor_transforms(y) for y in transformed_images[1:])
        else:
            x,ys = task_transform(idx)
            transformed_images = self.base_transform([x, ys[-1]])
            return (self.tensor_transforms(transformed_images[0]),
                    [self.tensor_transforms(y) for y in transformed_images[1:])


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
