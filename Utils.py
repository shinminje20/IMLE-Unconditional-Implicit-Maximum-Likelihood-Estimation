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
    """Returns the options string of [args]."""
    return f"-{'-'.join(args.options)}-" if len(args.options) > 0 else "-"

def suffix_str(args):
    """Returns the suffix string of [args]."""
    return f"-{args.suffix}" if not args.suffix == "" else ""

def load_experiment(file):
    """Returns a (model, optimizer, last_epoch, args, results) tuple from
    [file].
    """
    data = torch.load(file)
    model = data["model"].to(device)
    last_epoch = data["last_epoch"]
    optimizer = data["optimizer"]
    args = data["args"]
    results = data["results"]
    return model, optimizer, last_epoch, args, results

def save_model(model, optimizer, last_epoch, args, results, folder):
    """Saves input experiment objects to the [last_epoch].pt file [folder]."""
    torch.save({
        "model": model.cpu(),
        "optimizer": optimizer,
        "last_epoch": last_epoch,
        "args": args,
        "results": results,
    }, f"{folder}/{last_epoch}.pt")

def generator_folder(args):
    """Returns the folder to save a generator trained with [args] to."""
    folder = f"Models/generator-{args.data}-{opts_str(args)}{suffix_str(args)}"
    if not os.path.exists(folder): os.makedirs(folder)
    return folder

def resnet_folder(args):
    """Returns the folder to save a resnet trained with [args] to."""
    folder = f"Models/resnets-{args.backbone}-{args.data}{opts_str(args)}{suffix_str(args)}"
    if not os.path.exists(folder): os.makedirs(folder)
    return folder

################################################################################
# Training utilities
################################################################################

class CosineAnnealingLinearRampLR(_LRScheduler):

    def __init__(self, optimizer, T_0, n_ramp, T_mult=1, eta_min=0,
        last_epoch=-1, verbose=False):
        """
        Args:
        optimizer   -- the wrapped optimizer
        T_0         -- base COSINE period
        n_ramp      -- number of linear ramp epochs
        T_mult      -- multiplicative period change ()
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

        super(CosineAnnealingLinearRampLR, self).__init__(optimizer, last_epoch, verbose)


    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if not self.ramped:
            return [b * ((self.T_cur + 1) / self.n_ramp) for b in self.base_lrs]
        else:
            return [self.eta_min + (b - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for b in self.base_lrs]

    def step(self):
        # Behave like linear ramp in this case
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





from torch.optim import Adam, SGD
from torch.autograd import Variable
from torch.nn.parameter import Parameter
class LARS(Optimizer):
    """Code slightly modified from

    https://github.com/AndrewAtanov/simclr-pytorch/blob/master/utils/lars_optimizer.py

    which modified it from

    https://github.com/NVIDIA/apex/blob/d74fda260c403f775817470d87f810f816f3d615/apex/parallel/LARC.py
    """

    def __init__(self, model, lr, momentum, weight_decay=1e-6,
        trust_coefficient=0.001):
        """
        Args:
        model               -- the model to optimize
        lr                  -- learning rate of wrapped SGD
        momentum            -- momentum of wrapped SGD
        weight_decay        -- weight_decay of wrapped SGD
        trust_coefficient   -- the trust coefficient
        """
        param_groups = [
            {"params": [p for n,p in model.named_parameters() if "bn" in n],
                "weight_decay": weight_decay,
                "layer_adaption": False
            },
            {"params": [p for n,p in model.named_parameters() if not "bn" in n],
                "weight_decay": weight_decay,
                "layer_adaption": True
            }
        ]
        optimizer = SGD(param_groups, lr=lr, momentum=momentum,
            weight_decay=weight_decay)
        self.param_groups = optimizer.param_groups
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

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
    """Returns set, list, or tuple [xs] after recursively flattening."""
    result = []
    for x in xs:
        if isinstance(x, list) or isinstance(x, set) or isinstance(x, tuple):
            result += flatten(x)
        else:
            result.append(x)

    return result
