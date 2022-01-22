"""File containing utilities that provide corruptions that can be applied to
images. It can also be run to visualize what these corruptions actually do.
"""
import argparse
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from Data import *
from utils.NestedNamespace import NestedNamespace
from utils.Utils import *

################################################################################
# GENERATOR AUGMENTATIONS. These are for corrupting images before they are fed
# to the generator. The generator is responsible for outputting more than one
# image per input image.
################################################################################

# class LearnablePerPixelMaskCorruption(nn.Module):
#     """Learns a pixel-wise mask through gradients propagated back through the
#     network.
#
#     Args:
#     input_dim   -- the BSxCxHxW dimensionality of input image minibatches
#     n_channels  -- the number of channels internal to the ConvNets
#     """
#     def __init__(self, input_dim, n_channels=32, size=16):
#         super(LearnablePerPixelMaskCorruption, self).__init__()
#         _,c,h,w = input_dim
#         # This choice of kernel size and padding shouldn't change the dimension
#         # of the image while being run through the convolutional layers
#         self.model = nn.Sequential(
#             nn.Conv2D(c, n_channels, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2D(n_channels, n_channels, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2D(n_channels, n_channels, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Linear(h * w, h * w),
#             nn.ReLU(),
#             nn.Linear(h * w, h * w)
#         )
#
#     def forward(self, x):
#         mask = self.model(x) < 0
#         x[mask] = 0
#         return x

def get_non_learnable_image_corruption(size=256, mask_prob=0):
    corruptions = []
    corruptions.append(transforms.Resize((256, 256)))
    corruptions.append(transforms.ToTensor())
    if mask_prob > 0:
        corruptions.append(transforms.RandomErasing(p=mask_prob))
    c = transforms.Compose(corruptions)

    def f(x):
        print(type(x))
        return c(x)

    return f

def get_non_learnable_batch_corruption(
    rand_illumination=0,
    pixel_mask_frac=0,
    mask_prob=0,
    grayscale=False,
    ):
    """
    Args:
    pixel_mask_frac    -- the fraction of an image to randomly mask
    grayscale          -- whether or not to grayscale an image
    """
    corruptions = []
    if rand_illumination > 0:
        corruptions.append(RandomIllumination(sigma=rand_illumination))
    if pixel_mask_frac > 0:
        corruptions.append(RandomPixelMask(pixel_mask_frac))
    if grayscale:
        corruptions.append(transforms.Grayscale(num_output_channels=3))
    # if mask_prob > 0:
    #     corruptions.append(transforms.RandomErasing(p=mask_prob))
    return transforms.Compose(corruptions)

class RandomPixelMask(nn.Module):
    """Returns images with [pixel_mask_frac] of the pixels set to zero. (Feed
    images to this at 16x16 resolution.)
    """
    def __init__(self, pixel_mask_frac, size=16):
        super(RandomPixelMask, self).__init__()
        self.pixel_mask_frac = pixel_mask_frac
        self.size = size

    def forward(self, x):
        s = x.shape[-1] // self.size
        mask = torch.rand(size=(x.shape[0], x.shape[1], self.size, self.size))
        mask = F.interpolate(mask, scale_factor=(s, s), mode="nearest")
        x[mask < self.pixel_mask_frac] = 0
        return x

class RandomIllumination(nn.Module):

    def __init__(self, sigma=.1):
        super(RandomIllumination, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        sigmas = (torch.rand(x.shape[0], device=device) - .5) * 2 * self.sigma
        expanded_shape = tuple([len(x)] + ([1] * (len(tuple(x.shape)) - 1)))
        sigmas = sigmas.view(expanded_shape)
        return torch.clamp(x + sigmas.expand(x.shape), 0, 1)

################################################################################
# CONTRASTIVE LEARNING CORRUPTIONS. These corruptions can be applied to images
# decorrupted via a generator.
################################################################################

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="SimCLR training")
    P.add_argument("--data", choices=["cifar10", "miniImagenet", "camnet3"],
        default="cifar10",
        help="dataset to load images from")
    P.add_argument("--rand_illumination", default=0, type=float,
        help="amount of random illumination")
    P.add_argument("--pixel_mask_frac", default=0, type=float,
        help="amount of random masking")
    P.add_argument("--grayscale", action="store_true",
        help="whether or not to grayscale images")
    P.add_argument("--mask_prob", type=float, default=0,
        help="probability of adding transforms.RandomErasing")
    P.add_argument("--idxs", type=int, default=[-10], nargs="+",
        help="indices of images to corrupt, or negative number to sample that many randomly")
    args = NestedNamespace(P.parse_args())

    data, _ = get_data_splits(args.data, "cv")


    if len(args.idxs) == 1 and args.idxs[0] < 0:
        idxs = random.sample(range(len(data)), abs(args.idxs[0]))
    else:
        idxs = args.idxs

    transform = get_non_learnable_image_corruption(args.mask_prob)
    #
    # transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    #     transforms.RandomErasing(p=.5)
    # ])

    to_tensor = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    corruption = get_non_learnable_batch_corruption(
        rand_illumination=args.rand_illumination,
        grayscale=args.grayscale,
        pixel_mask_frac=args.pixel_mask_frac,
        mask_prob=args.mask_prob)

    images = [data[idx][0] for idx in idxs]
    image_grid = [deepcopy(images)] * 5
    images = [to_tensor(image) for image in images]
    image_grid = [[transform(image) for image in image_row] for image_row in image_grid]
    corrupted_images = [corruption(torch.stack(image_row)) for image_row in image_grid]
    image_grid = [images] + [[c for c in corrupted_images_row] for corrupted_images_row in corrupted_images]

    show_image_grid(image_grid)
