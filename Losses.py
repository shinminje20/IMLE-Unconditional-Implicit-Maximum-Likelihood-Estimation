"""Contains implementations of losses."""

import torch
import torch.nn as nn
from utils.UtilsLPIPS import LPIPSFeats
from utils.Utils import *
from torch.cuda.amp import autocast
import torch.nn.functional as F

def compute_loss_over_list(fxs, ys, loss_fn, list_reduction="sum"):
    """Returns [loss_fn] evaluated pairwise on lists [fxs] and [y]. It must be
    that [loss_fn] can be run on each sequential pair from the lists.

    Args:
    fxs             -- list of predictions
    ys              -- list of targets
    loss_fn         -- loss function
    list_reduction  -- reduction over the list. One of
                        'sum' to sum over the loss produced by each pair
                        'mean' for the mean loss produced by each pair
                        'batch' for the batch reduction
    """
    assert len(fxs) == len(ys)
    if list_reduction == "sum":
        losses = torch.cat([loss_fn(fx, y).mean().view(1) for fx,y in zip(fxs, ys)])
        return torch.sum(losses)
    elif list_reduction == "batch":
        losses = torch.cat([loss_fn(fx, y) for fx,y in zip(fxs, ys)])
        return torch.mean(losses.view(len(fxs), -1), axis=0)
    else:
        raise NotImplementedError()

class ResolutionLoss(nn.Module):
    def __init__(self, proj_dim=None, reduction="batch", alpha=.1):
        super(ResolutionLoss, self).__init__()
        self.lpips = ProjectedLPIPSFeats(proj_dim=proj_dim)
        self.alpha = alpha

    def forward(self, fx, y):
        lpips_loss = compute_loss_over_list(self.lpips(fx), self.lpips(y),
            batch_mse, list_reduction="batch")

        if fx.shape[-1] > 64:
            result = lpips_loss
        else:
            mse = batch_mse(fx.view(fx.shape[0], -1), y.view(y.shape[0], -1))
            result = lpips_loss + self.alpha * mse
        return result.squeeze()

class ProjectedLPIPSFeats(nn.Module):
    """Returns loss between LPIPS features of generated and target images."""
    def __init__(self, proj_dim=None):
        super(ProjectedLPIPSFeats, self).__init__()
        self.lpips = LPIPSFeats()
        self.proj_dim = proj_dim
        self.projections = nn.ModuleDict()

    def project_tensor(self, t):
        """Returns a projection matrix for a tensor with last size [dim]."""
        t = t if isinstance(t, list) else [t]

        results = [None] * len(t)
        for idx,t_ in enumerate(t):
            if not t_.shape[-1] in self.projections:
                t_shape = t_.shape[-1]
                projection = torch.randn(t_shape, self.proj_dim, requires_grad=True)
                projection = nn.Parameter(F.normalize(projection, p=2, dim=1))
                self.projections[t_shape] = projection

            results[idx] = torch.matmul(t_, self.projections[t_shape])

        return results

    def reset_projections(self): self.projections = nn.ModuleDict()

    def forward(self, x):
        """Returns a list of LPIPS features, with one element for each level of
        the VGG network making up the LPIPSFeats backbone. Each element is a
        BSxF tensor, where F is element-dependent and increases with the
        resolution of [x] and its position in the returned list.

        Args:
        x   -- BSxCxHxW tensor to get LPIPS features for
        """
        fx = self.lpips(x)
        if self.proj_dim is not None:
            fx = self.project_tensor(fx)
        return fx

def batch_mse(x, y):
    """Custom MSE loss that works on inputs that are larger than their targets.
    It uses a 'batch' reduction, meaning that a loss value exists for each input
    in the batch dimension. We divide by the dimensionality of each input so
    that the loss due to higher dimensionality inputs isn't increased.

    Args:
    x   -- a (B * K)xD tensor
    y   -- a BxD tensor

    Returns:
    A B * K tensor, where the ith element is MSE loss between the ith element
    of [x] and the (i // k) element of [y].
    """
    bs, d = x.shape
    y = y.unsqueeze(1).expand(-1, x.shape[0] // y.shape[0], -1)
    x = x.view(y.shape)
    return torch.sum(torch.square((x - y).view(bs, -1)), axis=1) / d

if __name__ == "__main__":
    r = nn.DataParallel(ResolutionLoss(proj_dim=None), device_ids=[0, 1]).cuda()
    with autocast():
        bs, c, h, w = 4, 3, 32, 32
        num_elements = bs * c * h * w
        x = torch.arange(bs * c * h * w).reshape(bs, c, h, w) / num_elements
        y = -1 * torch.arange(2 * c * h * w).reshape(2, c, h, w) / num_elements * 4
        x.requires_grad = True
    print(r(x.cuda(), y.cuda()))
    print(F.mse_loss(x.cuda(), y.cuda()))
