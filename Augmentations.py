from copy import deepcopy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import hflip

def get_simclr_color_distortion(color_s=1):
    """Returns the color distortion transform used in SimCLR. [color_s] gives
    the strength of the distortion.
    
    Based on https://github.com/AndrewAtanov/simclr-pytorch/blob/master/configs/cifar_train_epochs1000_bs1024.yaml, this
    parameter should be 0.5 for the CIFAR-10.
    """
    color_jitter = transforms.ColorJitter(0.8 * color_s,
         0.8 * color_s, 0.8 * color_s, 0.2 * color_s)
    return transforms.Compose([
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2)])

def get_real_augs(res=32):
    """Returns (training, finetuning, testing) augmentations that ensure images 
    remain on the real manifold.
    """
    augs_tr = transforms.Compose([
        transforms.RandomResizedCrop(res,
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    return augs_tr, deepcopy(augs_tr), deepcopy(augs_tr)

def get_contrastive_augs(res=32, gaussian_blur=False, color_s=0,
    fn_with_tr_augs=False):
    """Returns an (SSL transforms, finetuning transforms, testing transforms)
    tuple.

    Args:
    res             -- the resolution of images output by the transforms
    gaussian_blur   -- whether to use Gaussian blur or not
    color_s         -- color distortion strength
    fn_with_tr_augs -- whether finetuning/linear probe augmentations should
                        match the training or testing ones
    """
    augs_tr_list = [transforms.RandomResizedCrop(res,
            interpolation=transforms.InterpolationMode.BICUBIC)]

    if color_s > 0:
        augs_tr_list.append(get_simclr_color_distortion(color_s=color_s))
    if gaussian_blur:
        augs_tr_list.append(transforms.GaussianBlur(res // 10, sigma=0.5))
    augs_tr = transforms.Compose(augs_tr_list + [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    augs_te = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(res,
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()])

    augs_fn = deepcopy(augs_tr) if fn_with_tr_augs else deepcopy(augs_te)

    return augs_tr, augs_fn, augs_te

def get_gen_augs(args):
    """Returns a list of base transforms for image generation. Each should be
    able to accept multiple input images and be deterministic between any two
    images input at the same time, and return a list of the transformed images.
    """
    class RandomHorizontalFlips(nn.Module):
        """RandomHorizontalFlip but can be applied to multiple images."""
        def __init__(self, p=0.5):
            super(RandomHorizontalFlips, self).__init__()
            self.p = p

        def forward(self, images):
            """Returns [images] but with all elements flipped in the same
            direction, with the direction chosen randomly.

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

    return transforms.Compose([
            RandomHorizontalFlips(),
            ToTensors()
        ])