"""Code for building datasets.

See data/SetupDataset.py for downloading their underlying data.

The important portions of this file are get_data_splits(), which returns
training and evaluation ImageFolders, and the various Dataset subclasses that
can be used to construct various useful datasets.
"""
from collections import OrderedDict
import PIL
import random
import sys
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms.functional import hflip

from Corruptions import *
from utils.Utils import *

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
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    return {"generator": g, "worker_init_fn": seed_worker}


################################################################################
# Functionality for loading datasets
################################################################################

def get_data_splits(data_str, eval_str, res=None):
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
    def paths_to_datasets(data_paths):
        """Returns ImageFolder(s) given [data_paths], a list of data paths for a
        list of ImageFolders, or a string for a single ImageFolder.
        """
        if isinstance(data_paths, (list, OrderedDict)):
            return [ImageFolder(data_path) for data_path in data_paths]
        else:
            return ImageFolder(data_paths)

    data_path = f"{project_dir}/data/{data_str}"

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
        data_paths_tr = [f"{data_path}_{r}/train" for r in res]
        data_paths_eval = [
             f"{data_path}_{min(res)}/{eval_split_specifier}",
             f"{data_path}_{max(res)}/{eval_split_specifier}"
        ]

    check_paths_exist([data_paths_tr, data_paths_eval])
    return paths_to_datasets(data_paths_tr), paths_to_datasets(data_paths_eval)

################################################################################
# Augmentations
################################################################################
def get_simclr_augs(crop_size=32, gaussian_blur=False, color_s=.5, strong=True):
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
            return [hflip(img) for img in images]
        return images

    def __repr__(self): return f"{self.__class__.__name__}(p={self.p})"

class ToTensors(nn.Module):
    def __init__(self):
        super(ToTensors, self).__init__()
        self.to_tensor = transforms.ToTensor()

    def forward(self, images): return [self.to_tensor(x) for x in images]

    def __repr__(self): return self.__class__.__name__


def get_gen_augs():
    """Returns a list of base transforms for image generation. Each should be
    able to accept multiple input images and be deterministic between any two
    images input at the same time, and return a list of the transformed images.
    """
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
    def __init__(self, datasets, transform):
        self.datasets = datasets
        self.transform = transform

        ########################################################################
        # Validate the sequence of datasets. The H and W dimensions of
        ########################################################################
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
        print("IIIII", type(images[0]))
        images = self.transform(images)
        return images[0], images[1:]

    def __repr__(self): return f"GeneratorDataset\n\tshapes {self.shapes}"







#
# class MultiTaskDataset(Dataset):
#     """A dataset for forcing a model a generative model to perform multiple
#     tasks. Returned images are as follows. Suppose x_1, x_2, ... x_N are
#     versions of an image x at different (typically increasing) resolutions. Then
#     this dataset can return
#
#         [corruption(transform(x_2)),
#          transform(x_1),
#          ...
#          transform(x_N)
#         ]
#
#     Args:
#     data                        -- list of input datasets. The first item in
#                                     the list is a dataset giving that will be
#                                     corrupted; the rest should be sequentially
#                                     higher resolutions of the original
#                                     (uncorrupted) image
#     corruption                  -- transform giving the (x, (y1 ... yn))
#                                     images—the model must decorrupt [x] to
#                                     the sequence of ys
#     transform                   -- transform applied to all images in the
#                                     (x, (y1 ... yn)) sequence returned from
#                                     __getitem__(). This transform must be
#                                     able to apply to a list of images, and
#                                     be deterministic across this list.
#     intermediate_supervision    -- whether to return targets for supervising
#                                     the model at different resolutions
#     """
#     def __init__(self, res2data, b
#         intermediate_supervision=True):
#         assert all([len(d) == len(data[0]) for d in data]), f"All input datasets must have equal length, but lengths were {[len(d) for d in data]}"
#         self.data = data
#         self.corruption = corruption
#         self.transform = transform
#         self.intermediate_supervision = intermediate_supervision
#
#     def __len__(self): return len(self.data[0])
#
#     def __getitem(self, idx):
#         if self.intermediate_supervision:
#             images = [d[idx][0] for d in self.data]
#         else:
#             images = [self.data[0][idx][0], self.data[-1][idx][0]]
#
#         images = [self.transform(image) for image in images]
#         return self.corruption(images[0]), images[1:]

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
