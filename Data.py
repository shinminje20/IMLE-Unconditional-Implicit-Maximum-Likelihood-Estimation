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
data2split2n_class = {
    "cifar10": {"train": 10, "val": 10, "test": 10},
    "miniImagenet": {"train": 64, "val": 16, "test": 20} # Not sure if these numbers are correct
}

def seed_kwargs(seed=0):
    """Returns kwargs to be passed into a DataLoader to give it seed [seed]."""
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    return {"generator": g, "worker_init_fn": seed_worker}

################################################################################
# Baseline contrastive learning functions. Data is loaded directly, and image
# augmentations are basic/naive/not-necessarily-on-manifold.
################################################################################

def get_data_splits_ssl(data_str, eval_str):
    """Returns training and evaluation Datasets given [data_str] and [eval_str].

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
# ISICLE contrastive learning function—these are really data loading functions
# for a generative model that feeds data to a contrastive learner.
################################################################################

def get_data_splits_gen(data_str, eval_str, resolutions=[16, 32, 64, 128, 256]):
    """Returns training and evaluation Datasets given [data_str] and [eval_str].
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

class RandomHorizontalFlips(nn.Module):
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

class RandomResizedCrops(nn.Module):
    """This might become important later..."""
    pass

def get_base_augs(data_str):
    """Returns a list of base transforms for image generation. Each should be
    able to accept multiple input images and be deterministic between any two
    images input at the same time, and return a list of the transformed images.

    Args:
    data_str  -- a string specifying the dataset to get transforms for
    """
    return transforms.Compose([
        RandomHorizontalFlips()
    ])

def get_corruptions(corruptions_dict):
    """Returns a list of transforms acting as corruptions.

    Args:
    corruptions -- list containing any of 'grayscale',
    """
    corruptions = []

    if corruptions_dict["grayscale"]:
        corruptions.append(transforms.GrayScale(num_output_channels=3))
    elif corruptions_dict["crops"]:
        raise NotImplementedError()

    return transforms.Compose(corruptions)


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

    def __init__(self, data, F, bs=500):
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


class MultiTaskDataset(Dataset):
    """A dataset for forcing a model a generative model to perform multiple
    tasks. Returned images are as follows. Suppose x_1, x_2, ... x_N are
    versions of an image x at different (typically increasing) resolutions. Then
    this dataset can return

        [corruption(transform(x_2)),
         transform(x_1),
         ...
         transform(x_N)
        ]

    Args:
    data                        -- list of input datasets. The first item in
                                    the list is a dataset giving that will be
                                    corrupted; the rest should be sequentially
                                    higher resolutions of the original
                                    (uncorrupted) image
    corruption                  -- transform giving the (x, (y1 ... yn))
                                    images—the model must decorrupt [x] to
                                    the sequence of ys
    transform                   -- transform applied to all images in the
                                    (x, (y1 ... yn)) sequence returned from
                                    __getitem__(). This transform must be
                                    able to apply to a list of images, and
                                    be deterministic across this list.
    intermediate_supervision    -- whether to return targets for supervising
                                    the model at different resolutions
    """
    def __init__(self, res2data, b
        intermediate_supervision=True):
        assert all([len(d) == len(data[0]) for d in data]), f"All input datasets must have equal length, but lengths were {[len(d) for d in data]}"
        self.data = data
        self.corruption = corruption
        self.transform = transform
        self.intermediate_supervision = intermediate_supervision

    def __len__(self): return len(self.data[0])

    def __getitem(self, idx):
        if self.intermediate_supervision:
            images = [d[idx][0] for d in self.data]
        else:
            images = [self.data[0][idx][0], self.data[-1][idx][0]]

        images = [self.transform(image) for image in images]
        return self.corruption(images[0]), images[1:]

class ImagesFromTransformsDataset(Dataset):
    """Wraps an internal dataset for which queries produce a tuple in which an
    image is the first result. This dataset computes transformations [x] and
    [y]; the object for a neural net using the generated data is to generate [y]
    conditioned on [x] and a random vector z, or to contrast [x] and [y].

    [x] might be a downsampled and greyscaled version of the image, while [y]
    might be a random crop. This trains a generator to simulteneously do
    super-resolution, colorization, and cropping.

    Args:
    data        -- the wrapped dataset, should return tuples wherein the
                    first element is an image
    x_transform -- the transform giving input images
    y_transform -- the transform giving targets
    """
    def __init__(self, data, x_transform, y_transform):
        super(ImagesFromTransformsDataset, self).__init__()
        self.data = data
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        return self.x_transform(image), self.y_transform(image)
