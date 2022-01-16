import torch.nn as nn
from GeneratorArchitectures import *

class LPIPSLoss(nn.Module):
    """Returns MSE loss between LPIPS features of generated and target images."""
    def __init__(self):
        self.l_feats = LPIPSFeatsNet()
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, fx, y): return self.mse(self.l_feats(fx), self.l_feats(y))

class CodeGetter:
    """Class for maintaining the state of the latent codes used in IMLE. This
    wraps a bunch of functionality together so that we can
    """

    def __init__(self, hierarchical=False):
        self.hierarchical = hierarchical

    def generate_code(dim):
        """Returns a latent code sampled from N(0,1) with dimension [dim]."""
        if self.hierarchical:
            raise NotImplementedError()
        else:
            return torch.randn(size=dim)

def get_arch_args(args):
    """Returns the kwargs for building the architecture specified by [args]."""
    raise NotImplementedError()



class Generator(nn.Module):
    """
    Args:
    imle            -- whether to use the IMLE training methodology
    """

    def __init__(self, model, imle_options={}):
        if model == "camnet":
            self.model = CAMNet(**get_arch_args(args))
        else:
            raise ValueError(f"Unknown architecture '{arch}'")

    def forward(self, x, code=None):
        if code is None:
            return self.model(x)
        else:
            return self.model(x, code)




def one_epoch_gen_imle(corruptor, generator, optimizer, loader, loss_fn,
    hierarchical=False, iters_per_code=1000):
    """Trains [generator] and optionally [corruptor] for one epoch on data from
    [loader] via cIMLE.

    Args:
    corruptor       -- a corruption operator
    generator       -- a generative model that can undo corruptions
    optimizer       -- an optimizer for both the corruptor (if wanted) and
                        generator
    loader          -- DataLoader returning training and target data
    loss_fn         -- the loss function to use
    hierarchical    -- whether or not IMLE code generation should be
                        hierarchical
    iters_per_code  -- number of gradient steps for each code
    """
    C = CodeGetter(hierarchical=hierarchical)
    total_loss = 0
    for x,y in tqdm(loader, desc="Batches"):
        code = CodeGetter.generate_code()
        for _ in tqdm(range(iters_per_code), desc="Iterations"):
            with torch.no_grad():
                fx = corruptor(x.to(device, non_blocking=True))
            fx = generator(fx)
            loss = loss_fn(fx, y.to(device, non_blocking=True))
            loss.backward()
            total_loss += loss.item()

    return corruptor, generator, optimizer, total_loss / len(loader)
