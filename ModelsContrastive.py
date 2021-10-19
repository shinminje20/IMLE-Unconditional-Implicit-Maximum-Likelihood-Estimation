from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

def get_resnet_with_head(backbone, head_dim, is_cifar=False, head_type="none"):
    """Returns a resnet of [backbone] with a head of [head_type] attached."""
    if backbone == "resnet18":
        R = HeadlessResNet18(is_cifar=is_cifar)
    elif backbone == "resnet50":
        R = HeadlessResNet50(is_cifar=is_cifar)
    else:
        raise ValueError(f"Unknown backbone '{backbone}'")

    if head_type == "none":
        return nn.Sequential(OrderedDict([("backbone", R)]))
    elif head_type == "projection":
        H = ProjectionHead(R.out_dim, head_dim)
    elif head_type == "linear":
        H = nn.Linear(R.out_dim, head_dim)
    else:
        raise ValueError(f"Unknown head_type '{head_type}'")

    return nn.Sequential(OrderedDict([("backbone", R), ("head", H)]))

class HeadlessResNet18(nn.Module):
    """A class representing a ResNet18 with its head cut off.

    Code partially derived from https://github.com/leftthomas/SimCLR.
    """
    def __init__(self, is_cifar=False):
        super(HeadlessResNet18, self).__init__()

        ########################################################################
        # Construct the model architecture
        ########################################################################
        arch = models.resnet18(pretrained=False)
        if is_cifar:
            arch.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                bias=False)
            self.model = nn.Sequential(*[l for n,l in arch.named_children()
                if not n in ["fc", "maxpool"]])
        else:
            self.model = nn.Sequential(*[l for n,l in arch.named_children()])

        self.out_dim = 512

    def forward(self, x): return torch.flatten(self.model(x), 1)


class HeadlessResNet50(nn.Module):
    """A class representing a ResNet50 with its head cut off.

    Code partially derived from https://github.com/leftthomas/SimCLR.
    """
    def __init__(self, is_cifar=False):
        super(HeadlessResNet50, self).__init__()

        ########################################################################
        # Construct the model architecture
        ########################################################################
        arch = models.resnet50(pretrained=False)
        if is_cifar:
            arch.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                bias=False)
            self.model = nn.Sequential(*[l for n,l in arch.named_children()
                if not n in ["fc", "maxpool"]])
        else:
            self.model = nn.Sequential(*[l for n,l in arch.named_children()])

        self.out_dim = 2048

    def forward(self, x): return torch.flatten(self.model(x), 1)

class ProjectionHead(nn.Module):
    """A class implementing the projection head used for contrastive loss."""

    def __init__(self, in_dim, out_dim):
        """Args:
        in_dim  -- the input dimensionality
        out_dim -- the output dimensionality
        """
        super(ProjectionHead, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(in_dim, 2048, bias=False)),
            ("bn1", nn.BatchNorm1d(2048)),
            ("relu1", nn.ReLU(inplace=True)),
            ("fc2", nn.Linear(2048, out_dim, bias=False)),
            ("bn2", nn.BatchNorm1d(out_dim))]))

        # Turn off the bias for the bias term on this gradient
        self.model.bn2.bias.requires_grad = False

    def forward(self, x): return F.normalize(self.model(x), dim=1)

class DimensionedIdentity(nn.Identity):
    """nn.Identity() but with dimensions."""

    def __init__(self, out_dim):
        super(DimensionedIdentity, self).__init__()
        self.out_dim = out_dim
