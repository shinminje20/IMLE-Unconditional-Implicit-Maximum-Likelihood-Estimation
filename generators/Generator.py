import torch.nn as nn
from GeneratorArchitectures import *

def get_arch_args(args):
    """Returns the kwargs for building the architecture specified by [args]."""
    pass
    
class Generator(nn.Module):

    def __init__(self, arch, ):
        if arch == "camnet":
            self.model =
        else:
            raise ValueError(f"Unknown architecture '{arch}'")

    def forward(self, input):
