"""Downloads LPIPS data for use.
This avoids having to use the pip package, which is nice when <glares> pip
packages can't be installed.
Almost all code from the LPIPS repo:
https://github.com/richzhang/PerceptualSimilarity
Some is modified from CamNet:
https://github.com/niopeng/CAM-Net
"""
from os import path
import gdown
from tqdm import tqdm

import math
from collections import namedtuple
import torch
import torch.nn as nn
from torchvision import models as tv

def get_lpips_weights():
    """Downloads the LPIPS VGG16 weights. Our only contribution to this file!"""
    file = f"{path.dirname(f'{__file__}')}/vgg_lpips_weights.pth"
    if not path.exists(file):
        url = "https://drive.google.com/u/0/uc?id=1IQCDHxO-cYnFMx1hATjgSGQdO-_pB9nb&export=download"
        file = f"{path.dirname(f'{__file__}')}/vgg_lpips_weights.pth"
        gdown.download(url, file, quiet=False)

class LPIPSFeats(nn.Module):
    """Neural net for getting LPIPS features. Heavily modifed from CamNet."""

    def __init__(self):
        super(LPIPSFeats, self).__init__()
        self.scaling_layer = ScalingLayer()
        self.vgg = vgg16()

        get_lpips_weights()
        W = torch.load(f"{path.dirname(f'{__file__}')}/vgg_lpips_weights.pth")
        self.lin_layers = [NetLinLayer(torch.sqrt(W[k])) for k in W]

        self.eval()

    def forward(self, x, normalize=True):
        """Returns an n_samples x 124928 tensor where each the ith row is the
        LPIPS features of the ith example in [x].
        Args:
        x           -- input to get LPIPS features for with shape B x C x H x W
        normalize   -- whether to normalize the input in 0...1 to -1...1
        """
        x = 2 * x - 1 if normalize else x
        x = self.scaling_layer(x)
        vgg_feats = [normalize_tensor(v) for v in self.vgg(x)]
        feats = [l(v) for l,v in zip(self.lin_layers, vgg_feats)]
        feats = torch.cat([l.flatten(start_dim=1) for l in feats], axis=1)
        return feats

def normalize_tensor(x, eps=1e-10):
    """Returns tensor [x] after normalization."""
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + eps)
    return x / (norm_factor + eps)

class NetLinLayer(nn.Module):
    """A single linear layer used as placeholder for LPIPS learnt weights."""

    def __init__(self, weight):
        super(NetLinLayer, self).__init__()
        self.weight = weight

    def forward(self, inp): return self.weight * inp

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale

class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out

if __name__ == "__main__":
    get_lpips_weights()
