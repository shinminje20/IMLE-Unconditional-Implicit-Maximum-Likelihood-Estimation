"""File containing utilities that provide corruptions that can be applied to
images. It can also be run to visualize what these corruptions actually do.
"""
import argparse
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Data import *
from utils.NestedNamespace import NestedNamespace
from utils.Utils import *


import kornia as K
from kornia.augmentation import AugmentationBase2D


################################################################################
# GENERATOR AUGMENTATIONS. These are for corrupting images before they are fed
# to the generator. The generator is responsible for outputting more than one
# image per input image.
################################################################################
#
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
    def __init__(self, pix_mask_frac, pix_mask_size=16, fill="zero"):
        super(RandomPixelMask, self).__init__()
        self.pix_mask_frac = pix_mask_frac
        self.size = pix_mask_size
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

class Corruption(nn.Module):

    def __init__(self, pix_mask_frac=0, pix_mask_size=1, fill="zero", grayscale=1, **kwargs):
        super(Corruption, self).__init__()
        corruptions = []

        if grayscale:
            corruptions.append(K.augmentation.RandomGrayscale(p=1))
        if pix_mask_frac > 0:
            corruptions.append(RandomPixelMask(pix_mask_frac=pix_mask_frac,
                pix_mask_size=pix_mask_size, fill=fill))
        self.model = nn.Sequential(*corruptions)

    def forward(self, x): return self.model(x)


################################################################################
# CONTRASTIVE LEARNING CORRUPTIONS. These corruptions can be applied to images
# decorrupted via a generator.
################################################################################

# if __name__ == "__main__":
#     P = argparse.ArgumentParser(description="SimCLR training")
#     P.add_argument("--data", choices=["cifar10", "miniImagenet", "camnet3"],
#         default="cifar10",
#         help="dataset to load images from")
#     P.add_argument("--res", nargs="+", type=int,
#         default=[64, 128],
#         help="resolutions to see data at")
#     P.add_argument("--grayscale", default=0, type=int, choices=[0, 1],
#         help="grayscale corruption")
#     P.add_argument("--pix_mask_size", default=8, type=int,
#         help="fraction of pixels to mask at 16x16 resolution")
#     P.add_argument("--pix_mask_frac", default=0, type=float,
#         help="fraction of pixels to mask at 16x16 resolution")
#     P.add_argument("--rand_illumination", default=0, type=float,
#         help="amount by which the illumination of an image can change")
#     P.add_argument("--idxs", default=[-10], type=int, nargs="+",
#         help="amount by which the illumination of an image can change")
#     args = P.parse_args()

#     expand_factor = 10

#     datasets, _ = get_data_splits(args.data, "val", args.res)
#     data = GeneratorDataset(datasets, get_gen_augs())

#     if len(args.idxs) == 1 and args.idxs[0] < 0:
#         idxs = random.sample(range(len(data)), abs(args.idxs[0]))
#     else:
#         idxs = args.idxs

#     data = Subset(data, idxs)
#     data_expanded = ExpandedDataset(data, expand_factor=expand_factor)

#     corruptor = Corruption(**vars(args))
#     corrupted_data = CorruptedDataset(data_expanded, corruptor)
#     image_grid = [[d[1][-1]] for d in data]
#     for idx,(c,_) in enumerate(corrupted_data):
#         image_grid[idx // expand_factor].append(c)

#     show_image_grid(make_cpu(image_grid))
