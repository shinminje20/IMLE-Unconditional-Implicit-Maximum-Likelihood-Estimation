"""File containing utilities."""
import math
import os
from tqdm import tqdm

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# Set up CUDA usage
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if "cuda" in device:
    torch.backends.cudnn.benchmark = True

################################################################################
# I/O Utils
################################################################################
state_sep_str = "=" * 40

def opts_str(args):
    """Returns the options string for [args]."""
    return f"-{'-'.join(args.options)}" if len(args.options) > 0 else "-"

def suffix_str(args):
    """Returns the suffix string for [args]."""
    return f"-{args.suffix}" if not args.suffix == "" else ""

def load_(file):
    """Returns a (model, optimizer, last_epoch, args, tensorboard results) tuple
    from [file].
    """
    data = torch.load(file)
    model = data["model"].to(device)
    optimizer = data["optimizer"]
    last_epoch = data["last_epoch"]
    args = data["args"]
    tb_results = data["tb_results"]
    return model, optimizer, last_epoch, args, tb_results

def save_(model, optimizer, last_epoch, args, tb_results, folder):
    """Saves input experiment objects to the [last_epoch].pt file [folder]."""
    tb_results.flush()
    tb_results.close()
    torch.save({"model": model.cpu(), "optimizer": optimizer,
        "last_epoch": last_epoch, "args": args, "tb_results": tb_results},
        f"{folder}/{last_epoch}.pt")
    model.to(device)

def generator_folder(args):
    """Returns the folder to save a generator trained with [args] to."""
    folder = f"Models/generator-{args.data}-{opts_str(args)}{suffix_str(args)}"
    if not os.path.exists(folder): os.makedirs(folder)
    return folder

def resnet_folder(args):
    """Returns the folder to to which to save a resnet trained with [args]."""
    folder = f"Models/resnets-{args.backbone}-{args.data}{opts_str(args)}{suffix_str(args)}"
    if not os.path.exists(folder): os.makedirs(folder)
    return folder

################################################################################
# SimCLR utilities
################################################################################

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

################################################################################
# Miscellaneous
################################################################################
def flatten(xs):
    """Returns collection [xs] after recursively flattening into a list."""
    result = []
    for x in xs:
        if isinstance(x, list) or isinstance(x, set) or isinstance(x, tuple):
            result += flatten(x)
        else:
            result.append(x)

    return result
