
import functools
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

def expand_across_zero_dim(x, exp_factor):
    """Returns BSxAx...xZ tensor expanded across the zero dimension."""
    if exp_factor == 1:
        return x
    else:
        return torch.cat(exp_factor * [x], axis=0)

def init_weights(net, init_type='kaiming', scale=1, std=0.02):

    def weights_init_normal(m, std=0.02):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, std)
            nn.init.constant_(m.bias.data, 0.0)

    def weights_init_kaiming(m, scale=1):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

    # scale for 'kaiming', std for 'normal'.
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))

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
