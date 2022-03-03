import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def reduce_loss_over_batch(loss):
    """Returns unreduced loss [loss] reduced over the batch dimension."""
    return torch.mean(loss.view(loss.shape[0], -1), axis=1)

def compute_loss(fx, y, loss_fn, reduction="batch", list_reduction="mean"):
    """Returns the loss of output [fx] against target [y] using loss function
    [loss_fn] and reduction strategies [reduction] and [list_reduction].

    Args:
    fx              -- list of or single BSxCxHxW generated images
    y               -- list of or single BSxCxHxW ground-truth image
    loss_fn         -- unreduced loss function that acts on 4D tensors
    reduction       -- how to reduce across the images
    list_reduction  -- how to reduce across the list if inputs include lists
    """
    if isinstance(fx, list) and isinstance(y, list):
        losses = [compute_loss(f, t, loss_fn, reduction) for f,t in zip(fx, y)]
        if list_reduction == "mean":
            return torch.mean(torch.stack(losses), axis=0)
        else:
            raise ValueError(f"Unknown list_reduction '{list_reduction}'")
    else:
        if reduction == loss_fn.reduction:
            return loss_fn(fx, y)
        elif reduction == "batch" and loss_fn.reduction == "none":
            return reduce_loss_over_batch(loss_fn(fx, y))
        elif reduction == "mean":
            return torch.mean(loss_fn(fx, y))
        else:
            raise ValueError(f"Requested reduction '{reduction}' and/or loss_fn reduction '{loss_fn.reduction}' are invalid or can't be used together.")

class ResolutionLoss(nn.Module):
    """Loss function for computing MSE loss on low resolution images and LPIPS
    loss on higher resolution images.
    """
    def __init__(self, reduction="batch", alpha=.1):
        super(ResolutionLoss, self).__init__()
        self.mse = get_unreduced_loss_fn("mse")
        self.lpips = get_unreduced_loss_fn("lpips")
        self.alpha = alpha

        if not reduction == "batch":
            raise ValueError("ResolutionLoss can only be used with a batch reduction")
        self.reduction = "batch"

    def forward(self, fx, y):
        if fx.shape[-1] >= 64:
            result = reduce_loss_over_batch(self.lpips(fx, y))
            return result
        else:
            lpips_loss = reduce_loss_over_batch(self.lpips(fx, y))
            mse_loss = reduce_loss_over_batch(self.mse(fx, y))
            result = lpips_loss + self.alpha * mse_loss

class LPIPSLoss(nn.Module):
    """Returns loss between LPIPS features of generated and target images."""
    def __init__(self, reduction="mean", proj_dim=None):
        super(LPIPSLoss, self).__init__()
        self.lpips = LPIPSFeats()
        self.reduction = "none"
        self.loss = BroadcastMSELoss(reduction=self.reduction)

        self.proj_dim = proj_dim
        self.projections = {}

    def project_tensor(self, t):
        """Returns a projection matrix for a tensor with last size [dim]."""
        if not t.shape[-1] in self.projections:
            projection = torch.randn(t.shape[-1], self.proj_dim, device=device)
            projection = F.normalize(projection, p=2, dim=1)
            self.projections[t.shape[-1]] = projection

        return torch.matmul(t, self.projections[t.shape[-1]])

    def reset_projections(self): self.projections = {}

    def forward(self, fx, y):
        fx, y = self.lpips(fx), self.lpips(y)

        if self.proj_dim is not None:
            fx, y = self.project_tensor(fx), self.project_tensor(y)

        return self.loss(fx, y)

lpips_loss = None
def get_unreduced_loss_fn(loss_fn, proj_dim=None):
    """Returns an unreduced loss function of type [loss_fn]. The LPIPS loss
    function is memoized.
    """
    if loss_fn == "lpips":
        global lpips_loss
        if lpips_loss is None:
            lpips_loss = LPIPSLoss(reduction="batch", proj_dim=proj_dim).to(device)

        lpips_loss.reset_projections()
        return lpips_loss
    elif loss_fn == "mse":
        return nn.MSELoss(reduction="none")
    elif loss_fn == "resolution":
        return ResolutionLoss().to(device)
    else:
        raise ValueError(f"Unknown loss type {loss_fn}")

class BroadcastMSELoss(nn.Module):
    def __init__(self, reduction="batch"):
        super(BroadcastMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        # tqdm.write(f"INPUT BroadcastMSELoss SHAPES: x {x.shape} y {y.shape}")
        if len(y.shape) == 2:
            y = y.unsqueeze(1)
        if len(x.shape) == 2:
            x = x.view(y.shape[0], x.shape[0] // y.shape[0], x.shape[-1])
        if not (len(x.shape) == 3 and x.shape[0] == y.shape[0] and x.shape[2] == y.shape[2]):
            raise ValueError(f"Got invalid shapes for BroadcastMSELoss. x shape was {x.shape} and y shape was {y.shape}")

        result = torch.cdist(x, y)

        if self.reduction == "batch" or self.reduction == "none":
            return result.view(result.shape[0] * result.shape[1], 1)
        elif self.reduction == "mean":
            return torch.mean(result)
        else:
            raise ValueError(f"Unknown reduction {self.reduction}")
