from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F

from torchvision import models

from utils.Utils import *

class HeadedResNet(nn.Module):
    """A resnet of [backbone] with a head of [head_type] attached.

    Args:
    backone     -- backbone to use
    head_dim    -- output dimensionality of projection head
    small_image -- whether to modify the backbone to better work on small images
    head_type   -- the type of head to put on the ResNet
    """
    def __init__(self, backbone, head_dim, small_image, head_type="none"):
        super(HeadedResNet, self).__init__()
        R = HeadlessResNet(backbone=backbone, small_image=small_image)
        if head_type == "none":
            self.model = nn.Sequential(OrderedDict([("backbone", R)]))
        elif head_type == "projection":
            self.model = nn.Sequential(OrderedDict([
                ("backbone", R),
                ("projection", ProjectionHead(R.out_dim, head_dim))]))
        elif head_type == "linear":
            self.model = nn.Sequential(OrderedDict([
                ("backbone", R),
                ("linear", nn.Linear(R.out_dim, head_dim))]))
        else:
            raise ValueError(f"Unknown head type '{head_type}'")

    def forward(self x): return self.model(x)

class HeadlessResNet(nn.Module):
    """A class representing a ResNet with its head cut off."""
    def __init__(self, backbone, small_image=False):
        super(HeadlessResNet, self).__init__()

        if backbone == "resnet18":
            arch = models.resnet18(pretrained=False)
            self.out_dim = 512
        elif backbone == "resnet50":
            arch = models.resnet50(pretrained=False)
            self.out_dim = 2048
        else:
            raise ValueError(f"Unknown backbone '{backbone}'")

        # As per SimCLR, for the CIFAR datasets we change the first Conv2D layer
        # and ommit the MaxPooling layer to avoid losing too much information.
        # Really, we should do this for any dataset with small images.
        if small_image:
            arch.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.model = nn.Sequential(*[l for n,l in arch.named_children()
                if not n in ["fc", "maxpool"]])
        else:
            self.model = nn.Sequential(*[l for n,l in arch.named_children()
                if not n in ["fc"]])

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

class NTXEntLoss:
    """NT-XEnt loss, modified from PyTorch Lightning."""

    def __init__(self, temp=.5):
        """Args:
        temp    -- contrastive loss temperature
        """
        self.temp = temp

    def __call__(self, fx1, fx2):
        """Returns the loss from pre-normalized projections [fx1] and [fx2]."""
        out = torch.cat([fx1, fx2], dim=0)
        n_samples = len(out)
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / self.temp)
        mask = ~torch.eye(n_samples, device=sim.device).bool()
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)
        pos = torch.exp(torch.sum(fx1 * fx2, dim=-1) / self.temp)
        pos = torch.cat([pos, pos], dim=0)
        return -torch.log(pos / neg).mean()


def get_param_groups(model, lars_param_groups, weight_decay=1e-6):
    """Returns the param_groups for [model] based on whether [lars_param_groups]
    should be used or not. The result of this function can be used as a drop-in
    replacement for 'model.parameters()' when constructing an optimizer that
    will be wrapped in LARS.
    """
    if lars_param_groups:
        return [
            {"params": [p for n,p in model.named_parameters() if "bn" in n],
            "weight_decay": weight_decay, "layer_adaption": False
            },
            {"params": [p for n,p in model.named_parameters() if not "bn" in n],
            "weight_decay": weight_decay, "layer_adaption": True}]
    else:
        return model.parameters()

class LARS(Optimizer):
    """Code slightly modified from

    https://github.com/AndrewAtanov/simclr-pytorch/blob/master/utils/lars_optimizer.py

    which modified it from

    https://github.com/NVIDIA/apex/blob/d74fda260c403f775817470d87f810f816f3d615/apex/parallel/LARC.py
    """

    def __init__(self, optimizer, trust_coefficient=0.001):
        """
        Args:
        optimizer           -- the wrapped optimizer
        trust_coefficient   -- the trust coefficient
        """
        self.param_groups = optimizer.param_groups
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state): pass
        # self.optim.__setstate__(state)

    def __repr__(self):
        return self.optim.__repr__()

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue

                    if weight_decay != 0:
                        p.grad.data += weight_decay * p.data

                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)
                    adaptive_lr = 1.

                    if param_norm != 0 and grad_norm != 0 and group["layer_adaption"]:
                        adaptive_lr = self.trust_coefficient * param_norm / grad_norm

                    p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]

class CosineAnnealingLinearRampLR(_LRScheduler):
    """Cosine Annealing scheduler with a linear ramp."""

    def __init__(self, optimizer, T_0, n_ramp, T_mult=1, eta_min=0,
        last_epoch=-1, verbose=False):
        """
        Args:
        optimizer   -- the wrapped optimizer
        T_0         -- base COSINE period
        n_ramp      -- number of linear ramp epochs
        T_mult      -- multiplicative period change
        eta_min     -- minumum learning rate
        last_epoch  -- index of the last epoch run
        verbose     -- whether to have verbose output or not
        """
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if n_ramp < 0 or not isinstance(n_ramp, int):
            raise ValueError(f"Expected integer n_ramp >= 0, but got {n_ramp}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        self.T_0 = T_0
        self.n_ramp = n_ramp
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.ramped = (last_epoch >= self.n_ramp)
        self.T_cur = last_epoch

        super(CosineAnnealingLinearRampLR, self).__init__(optimizer, last_epoch,
            verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if not self.ramped:
            return [b * ((self.T_cur + 1) / self.n_ramp) for b in self.base_lrs]
        else:
            cos = (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            return [self.eta_min + (b - self.eta_min) * cos for b in self.base_lrs]

    def step(self):
        if not self.ramped and self.last_epoch + 1 < self.n_ramp:
            self.T_cur += 1
            self.last_epoch += 1
        elif not self.ramped and self.last_epoch + 1 >= self.n_ramp:
            self.last_epoch += 1
            self.T_cur = 0
            self.ramped = True
        elif self.ramped and self.T_cur >= self.T_i:
            self.last_epoch += 1
            self.T_cur = self.T_cur - self.T_i
            self.T_i = self.T_i * self.T_mult
        elif self.ramped and self.T_cur < self.T_i:
            self.last_epoch += 1
            self.T_cur += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group["lr"] = lr
                self.print_lr(self.verbose, i, lr, max(0, self.last_epoch))

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
