import torch
import torch.nn as nn
from torchvision import transforms

################################################################################
# GENERATOR AUGMENTATIONS. These are for corrupting images before they are fed
# to the generator. The generator is responsible for outputting more than one
# image per input image.
################################################################################


class RandomPerPixelMaskCorruption(nn.Module):
    """Returns images with [mask_frac] of the pixels set to zero. (Feed images
    to this at 16x16 resolution.)
    """
    def __init__(self, mask_frac, size=16):
        super(RandomPerPixelMaskCorruption, self).__init__()
        self.mask_frac = mask_frac
        self.size = size

    def forward(self, x):
        mask = torch.rand(size=(x.shape[0], x.shape[1], self.size, self.size))
        mask = nn.functional.interpolate(mask, size=x.shape, mode="bilinear")
        x[mask < self.mask_frac] = 0
        return x

class LearnablePerPixelMaskCorruption(nn.Module):
    """Learns a pixel-wise mask through gradients propagated back through the
    network.

    Args:
    input_dim   -- the BSxCxHxW dimensionality of input image minibatches
    n_channels  -- the number of channels internal to the ConvNets
    """
    def __init__(self, input_dim, n_channels=32, size=16):
        super(LearnablePerPixelMaskCorruption, self).__init__()
        _,c,h,w = input_dim
        # This choice of kernel size and padding shouldn't change the dimension
        # of the image while being run through the convolutional layers
        self.model = nn.Sequential(
            nn.Conv2D(c, n_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(n_channels, n_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(n_channels, n_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Linear(h * w, h * w),
            nn.ReLU(),
            nn.Linear(h * w, h * w)
        )

    def forward(self, x):
        mask = self.model(x) < 0
        x[mask] = 0
        return x

def get_generator_corruptions(grayscale=True, rand_pixel_mask=True,
    learnable_pixel_mask=False):
    """Returns corruptions/augmentations for training a generator."""
    transforms = []
    if learnable_pixel_mask:
        raise NotImplementedError()
    elif rand_pixel_mask:
        transforms.append(RandomPerPixelMaskCorruption(.7))

    if grayscale:
        transforms.append(transforms.GrayScale())

    return transforms

################################################################################
# CONTRASTIVE LEARNING CORRUPTIONS. These corruptions can be applied to images
# decorrupted via a generator.
################################################################################
