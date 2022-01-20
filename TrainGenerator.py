import torch.nn as nn
from GeneratorArchitectures import *

def show_results(generator, images):
    

class CodeGetter:
    """Class for maintaining the state of the latent codes used in IMLE. This
    wraps a bunch of functionality together so that we can

    Args:
    iters       -- the number of iterations between new code generations
    hiearchical -- whether or not to use CAMNet's hierarchical sampling
    """

    def __init__(self, iters=1, num_samples=120, code_bs=1, hierarchical=False):
        self.num_samples = num_samples
        self.code_bs = code_bs
        self.iters = iters
        self.hierarchical = hierarchical
        self.cur_iter = 0
        self.cur_code = None

    def get_codes(z_dims, x_batch, y_batch, backbone, loss_fn):
        """Updates CodeGetter state and returns the correct latent code to use.

        Args:
        z_dims      -- list of shapes describing a latent code
        x_batch     -- the input of a batch wrapped in a list [BSxCxHxW]
        y_batch     -- the target of a batch wrapped in a list
                        [BSxCxH1xW1 ... BSxCxHNxWN]
        backbone    -- model backbone
        loss_fn     -- the means of determining distance
        """
        if self.cur_iter == iter or self.cur_iter == 0:
            self.cur_iter = 0

            bs = x_batch[0].shape[0]

            if self.hierarchical:
                raise NotImplementedError()
            else:
                best_codes = {i: torch.randn((bs,) + z_dim)
                    for i,z_dim in enumerate(z_dims)}

                for level,y_ in enumerate(y_batch):
                    for j in range(0, len(y_), self.code_bs):
                        with torch.no_grad():
                            x = [x[j:j+self.code_bs].to(device) for x in x_batch]
                            y = [y[j:j+self.code_bs].to(device) for y in y_]
                            c = [c[j:j+self.code_bs].to(device) for c in best_codes]
                            fx = backbone(x, c)
                            losses =











                self.cur_code = torch.randn(size=dim)

        self.cur_iter += 1
        return self.cur_code

class LPIPSLoss(nn.Module):
    """Returns loss between LPIPS features of generated and target images."""
    def __init__(self):
        self.l_feats = LPIPSFeatsNet()
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, fx, y):
        """Returns the loss between generated images [fx] and real images [y].
        [fx] and [y] can either be lists of images and loss is the sum of the
        loss computed on their pairwise elements, or simply batches of images.
        """
        get_loss = lambda fx_, y_: self.mse(self.l_feats(fx_), self.l_feats(y_))
        if isinstance(fx, list) and isinstance(y, list):
            return torch.sum([self.get_loss(fx_, y_) for fx_,y_ in zip(fx, y)])
        else:
            return self.get_loss(fx, y)

class GeneratorBackbone(nn.Module):
    """Base class for the architecture of any generator."""
    def __init__(self): pass
    def forward(self, x, code): pass
    def get_z_dims(self, x): pass

class Generator(nn.Module):
    """A generative model which undoes corruptions.

    Wraps a CodeGetter for use in getting latent codes and a Corruptor for
    getting corrupted versions of images. The intent is for the Generator to be
    the only thing that need be saved.

    Args:
    corruptor   -- a Corruptor
    backbone    -- a GeneratorBackbone
    code_getter -- a CodeGetter
    """
    def __init__(self, corruptor, code_getter, backbone):
        if arch_config["arch"] == "camnet":
            raise NotImplementedError()
        elif arch_config["arch"] == "rrdb_stacks":
            self.corruptor = corruptor
            self.backbone = backbone
            self.code_getter = code_getter
        else:
            raise ValueError(f"Unknown architecture '{arch}'")

    def forward(self, x):
        x = self.corruptor(x)
        codes = self.code_getter.get_codes(self.bacbone.get_z_dims(x))
        return self.backbone(x, codes)

def one_epoch_gen_imle(generator, optimizer, loader, loss_fn,
    iters_per_code_per_ex=1000, mini_bs=32):
    """Trains [generator] and optionally [corruptor] for one epoch on data from
    [loader] via cIMLE.

    This is slightly weird because for every batch of data returned from
    [loader], there is exactly one code. We train with mini batch size [mini_bs]
    for [iters_per_code_per_ex] iterations on the code before moving on to the
    next batch. Therefore, the actual number of steps against the gradient is

        len(loader) * iters_per_code_per_ex / mini_bs

    This means that for a fixed [iters_per_code_per_ex], each example will be
    included in training exactly the same number of times, and the degree of
    parallelization is controlled by [mini_bs].

    ****************************************************************************
    Note that in the typical terminology, a 'minibatch' and a 'batch' are
    synonymous, and here a 'minibatch' is a subset of a 'batch'.
    ****************************************************************************

    Args:
    generator               -- a generative model that can undo corruptions
    optimizer               -- an optimizer for both the corruptor
                                (if wanted) and generator
    loader                  -- DataLoader returning training and target data.
                                The data for both inputs and targets should be
                                a list of tensors
    loss_fn                 -- the loss function to use
    iters_per_code_per_ex   -- number of gradient steps for each code
    mini_bs                 -- the batch size to run per iteration. Must evenly
                                divide the batch size of [loader]
    """
    loss_fn = LPIPSLoss()
    total_loss = 0

    for x_batch,y_batch in tqdm(loader, desc="Batches"):

        for j in tqdm(range(0, iters_per_code_per_ex, mini_bs), desc="Iterations"):
            x = [input[j:j+mini_bs].to(device) for input in x_batch]
            y = [target[j:j+mini_bs].to(device) for input in y_batch]

            generator.zero_grad()
            fx = generator(x)
            loss = loss_fn(fx, y)
            loss.backward()
            total_loss += loss.item()

    return generator, optimizer, total_loss / len(loader)

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="CAMNet training")
    P.add_argument("--task", choices=["Colorization", "ColorizationSuperResolution"],
        default="ColorizationSuperResolution",
        help="task for training CAMNet")
    P.add_argument("--data", default="camnet_three",
        choices=["camnet_three"],
        help="data to train on")

    P.add_argument("--epochs", default=20, type=int,
        help="number of epochs (months) to train for")
    P.add_argument("--num_days", default=1e4, type=int,
        help="number of days per month. This is the number of iterations per minibatch, and may be larger than --bs / --bs_day, in which case training will loop over each batch multiple times.")
    P.add_argument("--bs", default=400, type=int,
        help="batch size. Across any minibatch, the latent code is constant"),
    P.add_argument("--bs_day", default=1, type=int,
        help="batch size for each iteration")
    P.add_argument("--suffix", default="",
        help="optional training suffix")
    P.add_argument("--options", default=[], nargs="+",
        help="options")
    P.add_argument("--gpu_ids", nargs="+", type=int, default=[0, 1],
        help="GPU IDs")

    P.add_argument("--use_dci", default=1, type=int,
        help="whether or not to use DCI")
