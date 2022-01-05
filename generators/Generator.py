import torch.nn as nn
from GeneratorArchitectures import *

def get_arch_args(args):
    """Returns the kwargs for building the architecture specified by [args]."""
    pass

class CAMNetCodeGetter:
    """Class for maintaining the state of the latent codes used in IMLE. This
    wraps a bunch of functionality together so that we can
    """

    def __init__(self):

    def generate_code(): pass

class Generator(nn.Module):
    """
    Args:
    imle            -- whether to use the IMLE training methodology
    """

    def __init__(self, model, imle_options={}):
        if model == "camnet":
            self.model = CAMNet(**get_arch_args(args))
            self.imle = True

        else:
            raise ValueError(f"Unknown architecture '{arch}'")

    def forward(self, input):
        if self.imle:
            pass
        else:
            return self.model(input)
