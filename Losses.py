"""Contains implementations of losses."""

import torch
import torch.nn as nn
from utils.UtilsLPIPS import LPIPSFeats
from utils.Utils import *
from torch.cuda.amp import autocast
import torch.nn.functional as F

def compute_loss_over_list(fx, y, loss_fn):
    """Returns the mean of [loss_fn] evaluated pairwise on lists [fx] and [y].
    It must be that [loss_fn] can be run on each sequential pair from the lists.

    Args:
    fx  -- list of predictions
    y   -- list of targets
    """
    loss = 0
    for fx_,y_ in zip(fx, y):
        loss += loss_fn(fx_, y_).mean()
    loss = loss / len(y)
    return loss

class ResolutionLoss(nn.Module):
    def __init__(self, proj_dim=None, reduction="batch", alpha=.1):
        super(ResolutionLoss, self).__init__()
        self.lpips = ProjectedLPIPSFeats(proj_dim=proj_dim)
        self.alpha = alpha
    
    def forward(self, fx, y):
        high_res = (fx.shape[-1] >= 64)
        fx_lpips = self.lpips(fx)
        y_lpips = self.lpips(y)

        fx = fx.view(fx.shape[0], -1)
        y = y.view(y.shape[0], -1)
        lpips_loss = batch_mse(fx_lpips, y_lpips)
        lpips_loss = torch.mean(lpips_loss.view(lpips_loss.shape[0], -1), axis=1)
        
        if high_res:
            result = lpips_loss
        else:
            mse = batch_mse(fx, y)
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
        if not t.shape[-1] in self.projections:
            projection = torch.randn(t.shape[-1], self.proj_dim, requires_grad=True)
            projection = nn.Parameter(F.normalize(projection, p=2, dim=1))
            self.projections[t.shape[-1]] = projection

        return torch.matmul(t, self.projections[t.shape[-1]])

    def reset_projections(self): self.projections = nn.ModuleDict()

    def forward(self, x):
        with autocast():
            x = self.lpips(x)
            if self.proj_dim is not None:
                x = self.project_tensor(x)
        return x

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
    return torch.sum(torch.square((x - y).float()).view(bs, -1), axis=1) / d

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
