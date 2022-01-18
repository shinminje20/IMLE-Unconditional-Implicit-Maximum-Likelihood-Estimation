"""File containing utilities that provide corruptions that can be applied to
images. It can also be run to visualize what these corruptions actually do.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from Data import *
from utils.Utils import *

################################################################################
# GENERATOR AUGMENTATIONS. These are for corrupting images before they are fed
# to the generator. The generator is responsible for outputting more than one
# image per input image.
################################################################################

def get_non_learnable_corruption(mask_frac=0, grayscale=False):
    """
    Args:
    rand_pixel_mask_frac    -- the fraction of an image to randomly mask
    grayscale               -- whether or not to grayscale an image
    """
    corruptions = []
    if mask_frac > 0:
        corruptions.append(RandomMask(mask_frac))
    if grayscale:
        corruptions.append(transforms.Grayscale())
    return transforms.Compose(corruptions)

class RandomMask(nn.Module):
    """Returns images with [mask_frac] of the pixels set to zero. (Feed images
    to this at 16x16 resolution.)
    """
    def __init__(self, mask_frac, size=16):
        super(RandomMask, self).__init__()
        self.mask_frac = mask_frac
        self.size = size

    def forward(self, x):
        mask = torch.rand(size=(x.shape[0], x.shape[1], self.size, self.size)) * 100
        mask = F.interpolate(mask, scale_factor=(16, 16), mode="nearest")
        x[mask < self.mask_frac * 100] = 0
        return x

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

################################################################################
# CONTRASTIVE LEARNING CORRUPTIONS. These corruptions can be applied to images
# decorrupted via a generator.
################################################################################

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="SimCLR training")
    P.add_argument("--data", choices=["cifar10", "miniImagenet", "camnet3"],
        default="cifar10",
        help="dataset to load images from")
    P.add_argument("--mask_frac", default=0, type=float,
        help="amount of random masking")
    P.add_argument("--grayscale", action="store_true",
        help="whether or not to grayscale images")
    P.add_argument("--idxs", type=int, default=[-10], nargs="+",
        help="indices of images to corrupt, or negative number to sample that many randomly")
    P.add_argument("--seed", type=int, default=None,
        help="random seed")
    args = NestedNamespace(P.parse_args())


    data, _ = get_data_splits(args.data, "cv")

    if len(args.idxs) == 1 and args.idxs[0] < 0:
        idxs = random.sample(range(len(data)), abs(args.idxs[0]))
    else:
        idxs = args.idxs


    transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.ToTensor()])

    corruption = get_non_learnable_corruption(grayscale=args.grayscale,
                                              mask_frac=args.mask_frac)

    images = [transform(data[idx][0]) for idx in idxs]
    images = torch.stack(images)
    corrupted_images = corruption(images)
    corrupted_images = [c for c in corrupted_images]
    print(corrupted_images[0])
    show_images_grid(corrupted_images)
