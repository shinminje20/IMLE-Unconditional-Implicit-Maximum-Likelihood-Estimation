"""
"""
import math
import os
from collections import namedtuple
import torch
import torch.nn as nn

from Utils import *

def act(act_type, inplace=True, neg_slope=0.2):
    """Returns an activation function of type [act_type].

    Args:
    act_type    -- activation type
    in_place    -- whether to do the activation in place or not
    neg_slope   -- negative slope for LeakyReLU
    """
    if act_type.lower() == "leakyrelu":
        return nn.LeakyReLU(neg_slope, inplace)
    elif act_type.lower() == "tanh":
        return nn.Tanh()
    else:
        raise NotImplementedError(f"Activation layer "{act_type}" is not found")

def get_valid_padding(kernel_size, dilation):
    """Returns the padding for a kernel with size [kernel_size] and dilation
    [dilation].
    """
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def normalize_tensor(in_feat, eps=1e-10):
    """Returns tensor [in_feat] after normalization."""
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True) + eps)
    return in_feat / (norm_factor + eps)


class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x, extra_x=None):
        if extra_x is None:
            output = x + self.sub(x)
        else:
            output = x + self.sub(x, extra_x)
        return output

    def __repr__(self):
        tmpstr = "Identity + \n|"
        modstr = self.sub.__repr__().replace("\n", "\n|")
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, dilation=1, bias=True, pad_type="zero", act_type="leakyrelu"):
    """
    Conv layer with weight normalization (NIPS 2016), activation
    """
    padding = get_valid_padding(kernel_size, dilation)
    padding = padding if pad_type == "zero" else 0

    c = weight_norm(nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias),
                    name="weight")
    a = act(act_type) if act_type else None
    return sequential(c, a)


def conv_block_noise(in_nc, out_nc, kernel_size, dilation=1, bias=True, pad_type="zero", act_type="leakyrelu"):
    """
    Conv layer with weight normalization (NIPS 2016), activation
    """
    padding = get_valid_padding(kernel_size, dilation)
    padding = padding if pad_type == "zero" else 0

    conv_v = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias)
    nn.init.trunc_normal_(conv_v.weight, std=0.15)
    c = weight_norm(conv_v, name="weight")

    a = act(act_type) if act_type else None
    return sequential(c, a)


def upconv_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, bias=True, pad_type="zero", act_type="leakyrelu",
                 mode="nearest"):
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, bias=bias, pad_type=pad_type, act_type=act_type)
    return sequential(upsample, conv)


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, nc, kernel_size=3, gc=32, bias=True, pad_type="zero", act_type="leakyrelu"):
        super(ResidualDenseBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, bias=bias, pad_type=pad_type, act_type=act_type)
        self.conv2 = conv_block(nc + gc, gc, kernel_size, bias=bias, pad_type=pad_type, act_type=act_type)
        self.conv3 = conv_block(nc + 2 * gc, gc, kernel_size, bias=bias, pad_type=pad_type, act_type=act_type)
        self.conv4 = conv_block(nc + 3 * gc, gc, kernel_size, bias=bias, pad_type=pad_type, act_type=act_type)
        self.conv5 = conv_block(nc + 4 * gc, nc, 3, bias=bias, pad_type=pad_type, act_type=None)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class ScalingLayer(nn.Module):
    # For rescaling the input to vgg16
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer("shift", torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class RerangeLayer(nn.Module):
    """Layer mapping inputs in [-1, 1] to [0, -1]."""
    def __init__(self): super(RerangeLayer, self).__init__()
    def forward(self, inp): return (inp + 1.) / 2.

class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    """
    def __init__(self, nc, kernel_size=3, gc=32, bias=True, pad_type="zero", act_type="leakyrelu"):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(nc, kernel_size, gc, bias, pad_type, act_type)
        self.RDB2 = ResidualDenseBlock(nc, kernel_size, gc, bias, pad_type, act_type)
        self.RDB3 = ResidualDenseBlock(nc, kernel_size, gc, bias, pad_type, act_type)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


class StyleBlock(nn.Module):
    """Style Block: Rescale each RRDB output."""
    def __init__(self, rrdbs, transformations, lr_conv):
        """
        Args:
        rrdbs           -- RRDB blocks
        transformations --
        lr_conv         --
        """
        super(StyleBlock, self).__init__()
        assert len(rrdbs) == len(transformations)
        self.nb = len(rrdbs)
        for i, rrdb in enumerate(rrdbs):
            self.add_module("%d" % i, rrdb)
            self.add_module("transform_%d" % i, transformations[i])
        self.lr_conv = lr_conv

    def forward(self, x, x_feat):
        for i in range(self.nb):
            rrdb_out = getattr(self, "%d" % i)(x)
            tran_out = getattr(self, "transform_%d" % i)(x_feat)
            bs, nc, w, h = rrdb_out.shape
            norm_layer = nn.InstanceNorm2d(nc, affine=False)
            x = (1. + tran_out[:, :nc].reshape(bs, nc, 1, 1)).expand(bs, nc, w, h) * norm_layer(rrdb_out) + \
                tran_out[:, nc:].reshape(bs, nc, 1, 1).expand(bs, nc, w, h)
        out = self.lr_conv(x)
        return out

class MappingNet(nn.Module):
    """A mapping network for one block of CAMNet."""

    def __init__(self, map_nc, latent_nc, act_type):
        """Args:
        map_nc      --
        latent_nc   --
        act_type    -- activation type to use
        """
        super(MappingNet, self).__init__()
        self.map_nc = map_nc
        self.model = nn.Sequential(*flatten(
            [nn.Linear(map_nc, latent_nc), act(act_type)] +
            [(nn.Linear(latent_nc, latent_nc), act(act_type)) for _ in range(7)]
        ))

    def forward(self, x): return self.model(x[:, :self.map_nc])

class CAMNetUpsampling(nn.Module):
    """Produces the output of a CAMNet level."""

    def __init__(self, resid_channels, upsample=False, upsample_kernel_mode="nearest",
        act_type="leakyrelu"):
        """Args:
        resid_channels                    -- number of residual channels
        upsample                -- whether to upsample or not
        upsample_kernel_mode    -- kernel mode for upsampling
        act_type                -- activation type to use
        """
        super(CAMNetLevelOutput, self).__init__()
        if upsample:
            self.model = nn.Sequential(
                B.upconv_block(resid_channels, resid_channels, act_type=act_type, mode=upsample_kernel_mode),
                B.conv_block(resid_channels, resid_channels, kernel_size=3, act_type=act_type))
        else:
            self.model = nn.Sequential(
                B.conv_block(resid_channels, resid_channels, kernel_size=3, act_type=act_type),
                B.conv_block(resid_channels, resid_channels, kernel_size=3, act_type=act_type))

    def forward(self, x): return self.model(x)

class CAMNetModule(nn.Module):

    def __init__(self, in_dim, out_dim):
        """
        Args:
        in_dim          -- number of dimensions of input CxWxH
        out_dim         -- number of dimensions of output CxWxH

        dense_channels  -- number of dense channels to use in RRDB blocks
        resid_channels  -- number of residual channels to use in RRDB blocks
        n_blocks        -- number of RRDB blocks in the level

        """
        super(CAMNetModule, self).__init__()
        self.map_channels = map_channels
        self.feat_scale = feat_scale
        self.mapping_net = MappingNet(map_channels, latent_channels, act_type)
        self.feat_net = B.conv_block(in_nc + code_nc + prev_resid_channels,
                                     resid_channels, kernel_size=3, act_type=None)
        self.style_block = B.ShortcutBlock(B.StyleBlock(
            [B.RRDB(resid_channels, gc=dense_channels, act_type=act_type) for _ in range(n_blocks)],
            [nn.Linear(latent_nc, 2 * resid_channels) for _ in range(n_blocks)],
            B.conv_block(resid_channels, resid_channels, kernel_size=3, act_type=None)
        ))
        self.upsample = CAMNetUpsampling(resid_channels,
            upsample=not (last_level and task == "colorization")))
        self.out = B.conv_block(resid_channels, out_nc, kernel_size=3, act_type="tanh")
        self.out_layer = RerangeLayer()

    def forward(self, net_input, code, feature=torch.tensor([])):
        """For now, equivalent to super-resolution.

        Args:
        net_input   -- the input to the network or the output of the prior level
        code        -- the code for the current level
        feature     -- the feature computed by the previous layer
        """
        bs, _, w, h = level_input.shape
        net_input = torch.cat([
            level_input,
            code[:, map_channels:]].reshape(bs, map_channels, w, h),
            feature * self.feat_scale], dim=1)

        mapped_code = self.mapping_net(code[:, :self.map_channels])
        feature = self.feat_net(net_input)
        feature = self.style_block(feature, mapped_code)
        feature = self.upsample(feature)
        out = self.out(feature)
        return feature, self.out_layer(out)

camnet_arch_defaults = {
    ### LEVEL COMPOSITION SPECIFICATION
    "n_levels": 4,                              # number of levels. Leave level does colorization, 2x super-resolution, and de-corrupting
    "base_size": 16,                            # size (height/width) of lowest resolution image fed to the network
    "out_channels": 2,                          # number of visual output channels to each level
    "tasks": ["colorization",                   # Each level performs a special task
        "super-resolution",
        "super-resolution",
        "super-resolution"],

    ### MappingNet SPECIFICATION
    "map_channels": 128,                        # number of channels on input to mapping network of each level
    "latent_channels": 512,                     # number of channels internal to mapping network of each level

    ### RRDB BLOCK SPECIFICATION
    "n_blocks": 6,                              # number of RRDB blocks in each level
    "n_resid_channels": [128, 64, 64, 64],      # number of residual channels in RRDB blocks
    "n_dense_channels": [256, 192, 128, 64],    # number of dense channels in RRDB blocks

    ### MISCELLANEOUS
    "act_type": "leakyrelu",                    # the activation unless otherwise intelligently hardcoded
}

class CAMNet(nn.Module):
    """CAMNet rewritten to be substantially stateless and better-documented."""

    def __init__(self, n_levels=4, base_size=16, out_channels=2,
        tasks=["colorization", "super-resolution", "super-resolution", "super-resolution"],
        map_channels=128, latent_channels=512, n_blocks=6, act_type="leakyrelu",
        n_resid_channels=[128, 64, 64, 64],
        n_dense_channels=[256, 192, 128, 64]
        feat_scales=None,
        ):
        """
        Args:

        ALL REMAINING ARGUMENTS DEFINED IN [camnet_arch_defaults] above.
        """
        ########################################################################
        # Build a list of dictionaries, each specifying the configuration for a
        # level. Then, build each level.
        ########################################################################
        level_info = [{
            "n_blocks": n_blocks,
            "n_dense_channels": n_dense_channels[i],
            "n_resid_channels": n_resid_channels[i],
            "map_channels": 128,
            "latent_channels": 512,
            "out_channels": 2,
            "size": base_size * 2 ** i,
            "task": tasks[i],
            "level": i,
            "feat_scale": (.1 * i) if feat_scales is None else feat_scales[i]
            } for i in range(n_levels)]

        self.levels = nn.Sequential(OrderedDict([
            (f"level {i}", CAMNetModule(**l)) for i,l in enumerate(level_info)]
        ))

    def forward(self, net_input, codes):
        """Returns a list of the outputs computed by each level.

        Unfortunately, it"s difficult to make this stateless in a clean way due
        to the fact that we need to return the output of each CAMNet level, so
        we need to collect them, and recursion isn"t going to make this much
        simpler.
        """
        level_output = net_input
        feature = torch.tensor([])

        # This loop has state in that [feature] is modified in each iteration
        outputs = []
        for code,level in zip(codes, self.levels):
            feature, level_output = level(level_output, code, feature=feature)
            outputs.append(level_output)

        return outputs
