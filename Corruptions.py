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

class CloneInput(nn.Module):

    def __init__(self):
        super(CloneInput, self).__init__()

    def forward(self, x): return x.clone()

class RandomPixelMask(nn.Module):
    """Returns images with in expectation [pix_mask_frac] of the pixels set to
    zero.

    Args:
    pix_mask_frac   -- expected fraction of pixels to mask out
    size            -- resolution (SIZExSIZE) at which to set pixels to zero.
                        Decreasing this makes for larger blotches of erased
                        image
    fill            -- how to fill the image, one of 'zero' or 'color'
    """
    def __init__(self, pix_mask_frac, size=16, fill="zero"):
        super(RandomPixelMask, self).__init__()
        self.pix_mask_frac = pix_mask_frac
        self.size = size
        self.fill = fill

    def forward(self, x):
        mask = torch.rand(size=(x.shape[0], self.size * self.size), device=device)
        _, indices = torch.sort(mask)
        indices = indices.view(x.shape[0], self.size, self.size).unsqueeze(1)
        cutoff = self.pix_mask_frac  * self.size * self.size
        mask = (indices < cutoff).float()

        mask = mask.expand(x.shape[0], x.shape[1], -1, -1)
        mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")

        if self.fill == "color":
            colors = torch.rand(size=(x.shape[0], x.shape[1], self.size, self.size), device=device)
            colors = F.interpolate(colors, size=x.shape[-2:], mode="nearest")
            x[mask.bool()] = colors[mask.bool()]
        elif self.fill == "zero":
            x[mask.bool()] = 0
        else:
            raise ValueError(f"Unknown fill type '{self.fill}'")
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

def get_non_learnable_batch_corruption(rand_illumination=0, pix_mask_frac=0,
    pix_mask_size=8, grayscale=False, fill="zero", **kwargs):
    """
    Args:
    pix_mask_frac    -- the fraction of an image to randomly mask
    grayscale          -- whether or not to grayscale an image
    """
    corruptions = []
    if grayscale:
        corruptions.append(transforms.Grayscale(num_output_channels=3))
        corruptions.append(CloneInput())
    if pix_mask_frac > 0:
        corruptions.append(RandomPixelMask(pix_mask_frac, size=pix_mask_size, fill=fill))
    if rand_illumination > 0:
        corruptions.append(RandomIllumination(sigma=rand_illumination))
    return transforms.Compose(corruptions)

################################################################################
# CONTRASTIVE LEARNING CORRUPTIONS. These corruptions can be applied to images
# decorrupted via a generator.
################################################################################

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="SimCLR training")
    P.add_argument("--data", choices=["cifar10", "miniImagenet", "camnet3"],
        default="cifar10",
        help="dataset to load images from")
    P.add_argument("--res", nargs="+", type=int,
        default=[64, 128],
        help="resolutions to see data at")
    P.add_argument("--grayscale", default=1, type=int, choices=[0, 1],
        help="grayscale corruption")
    P.add_argument("--pix_mask_size", default=8, type=int,
        help="fraction of pixels to mask at 16x16 resolution")
    P.add_argument("--pix_mask_frac", default=0, type=float,
        help="fraction of pixels to mask at 16x16 resolution")
    P.add_argument("--rand_illumination", default=0, type=float,
        help="amount by which the illumination of an image can change")
    P.add_argument("--idxs", default=[-10], type=int, nargs="+",
        help="amount by which the illumination of an image can change")
    args = P.parse_args()

    expand_factor = 10

    datasets, _ = get_data_splits(args.data, "cv", args.res)
    data = GeneratorDataset(datasets, get_gen_augs())

    if len(args.idxs) == 1 and args.idxs[0] < 0:
        idxs = random.sample(range(len(data)), abs(args.idxs[0]))
    else:
        idxs = args.idxs

    data = Subset(data, idxs)
    data_expanded = ExpandedDataset(data, expand_factor=expand_factor)

    corruptor = get_non_learnable_batch_corruption(**vars(args))
    corrupted_data = CorruptedDataset(data_expanded, corruptor)
    image_grid = [[d[1][-1]] for d in data]
    for idx,(c,_) in enumerate(corrupted_data):
        image_grid[idx // expand_factor].append(c)

    show_image_grid(make_cpu(image_grid))
