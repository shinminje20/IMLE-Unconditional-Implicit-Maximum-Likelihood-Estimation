from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

def get_resnet_with_head(backbone, head_dim, head_type="none"):
    """Returns a resnet of [backbone] with a head of [head_type] attached."""
    if backbone == "resnet18":
        R = HeadlessResNet18()
    elif backbone == "resnet50":
        R = HeadlessResNet50()
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
    def __init__(self):
        super(HeadlessResNet18, self).__init__()
        arch = models.resnet18(pretrained=False)

        # Modification made in the SimCLR paper
        arch.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3,
            bias=False)
        self.model = nn.Sequential(*[l for n,l in arch.named_children()
            if not n in ["fc", "maxpool"]])
        self.out_dim = 512

    def forward(self, x): return torch.flatten(self.model(x), 1)


class HeadlessResNet50(nn.Module):
    """A class representing a ResNet50 with its head cut off.

    Code partially derived from https://github.com/leftthomas/SimCLR.
    """
    def __init__(self):
        super(HeadlessResNet50, self).__init__()
        arch = models.resnet50(pretrained=False)
        # Modification made in the SimCLR paper
        arch.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3,
            bias=False)
        self.model = nn.Sequential(*[l for n,l in arch.named_children()
            if not n in ["fc", "maxpool"]])
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
            ("fc1", nn.Linear(in_dim, in_dim, bias=False)),
            ("bn1", nn.BatchNorm1d(in_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("fc2", nn.Linear(in_dim, out_dim, bias=False)),
            ("bn2", nn.BatchNorm1d(out_dim))]))

        # Turn off the bias for the bias term on this gradient
        self.model.bn2.bias.requires_grad = False

    def forward(self, x): return F.normalize(self.model(x), dim=1)

class LinearHead(nn.Module):
    """A class implementing a linear head for classification."""

    def __init__(self, in_dim, out_dim):
        """Args:
        in_dim  -- the input dimensionality
        out_dim -- the output dimensionality
        """
        super(ProjectionHead, self).__init__()
        self.model = nn.Linear(in_dim, out_dim)

    def forward(self, x): self.model(x)

class ContrastiveLoss:

    def __init__(self, temp):
        self.temp = temp

    def __call__(self, fx1, fx2):
        """Returns the loss on [fx1] and [fx2].
        Code from https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/losses/self_supervised_learning.py.
        """
        out = torch.cat([fx1, fx2], dim=0)
        n_samples = len(out)

        # Full similarity matrix
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / self.temp)

        # Negative similarity
        mask = ~torch.eye(n_samples, device=sim.device).bool()
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        # Positive similarity :
        pos = torch.exp(torch.sum(fx1 * fx2, dim=-1) / self.temp)
        pos = torch.cat([pos, pos], dim=0)
        loss = -torch.log(pos / neg).mean()

        return loss
