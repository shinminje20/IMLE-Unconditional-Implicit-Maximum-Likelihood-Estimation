"""The CAMNet architecture rewritten for our purposes, and to be
 - better documented
 - (nearly) stateless
 - substantially functional
NOTES:
 - read 'nc' as 'n channels' and variable names will make some sense!
"""
import argparse
from collections import OrderedDict
import math
import torch
import torch.nn as nn

import Block as B
from utils.Utils import *
from utils.UtilsColorSpace import rgb2lab_with_dims, lab2rgb_with_dims
from utils.UtilsNN import *
from utils.NestedNamespace import NestedNamespace

def get_z_dims(model):
    """Returns a list of tuples, where the ith tuple is the shape of the
    latent code required for the ith level without expansion to the batch
    dimension.
    """
    model = model.module if isinstance(model, nn.DataParallel) else model
    return [(model.map_nc + model.code_nc * s ** 2,) for i,s in enumerate(model.in_sizes)]

class CAMNet(nn.Module):
    """CAMNet rewritten to be substantially stateless, better-documented, and
    optimized for ISICLE.

    Args:
    ARGUMENTS FOR OVERALL CAMNET STRUCTURE
    res        -- list of resolutions, starting with the input resolution to the
                    model and ending with the output resolution. Each resolution
                    should be equal to or 2x the last.
    n_levels   -- number of CAMNet levels. Must be at least the number of
                    resolutions

    ARGUMENTS FOR CONSTITUENT CAMNET LEVELS
    code_nc    -- number of code channels
    in_nc      -- number of input channels
    out_nc     -- number of output channels
    map_nc     -- number of input channels to mapping net
    latent_nc  -- number of channels inside the mapping net
    resid_nc   -- list of numbers of residual channels in RRDB blocks for each CAMNet level
    dense_nc   -- list of numbers of dense channels in RRDB  blocks for each CAMNet level
    n_blocks   -- number of RRDB blocks inside each level
    act_type   -- activation type
    feat_scales-- amount by which to scale features, or None
    """

    def __init__(self, res=[32, 64, 128, 256], levels=4, code_nc=5, in_nc=3,
        out_nc=3, map_nc=128, latent_nc=512, resid_nc=[128, 64, 64, 64],
        dense_nc=[256, 192, 128, 64], n_blocks=6, act_type="leakyrelu",
        feat_scales=None, color_space="rgb", init_scale=.1, init_type="kaiming",
        **kwargs):
        super(CAMNet, self).__init__()

        ########################################################################
        # Check input validity
        ########################################################################
        if not len(res) == levels + 1:
            raise ValueError(f"Got 'res' {res} and {levels} levels. 'res' should specify the input of each CAMNet level followed by the output resolution, and must therefore have one more item than 'levels'.")
        for r1,r2 in zip(res[:-1],res[1:]):
            if not (r1 == r2 or r1 * 2 == r2):
                raise ValueError(f"Got 'res' {res} btu sequential resolutions should have equal size or be 2x the last")
        if not all([len(x) >= levels for x in [resid_nc, dense_nc]]):
            raise ValueError(f"'dense_nc' and 'resid_nc' must have levels={levels} entries")

        ########################################################################
        # Create the network
        ########################################################################
        self.in_sizes, self.out_sizes= res[:-1], res[1:]
        level_info = [{
            "n_blocks": n_blocks,
            "map_nc": map_nc,
            "latent_nc": latent_nc,
            "code_nc": code_nc,
            "in_nc": in_nc,
            "out_nc": out_nc,
            "dense_nc": dense_nc[i],
            "resid_nc": resid_nc[i],
            "prev_resid_nc": 0 if i == 0 else resid_nc[i-1],
            "level": i,
            "feat_scale": (.1 * i) if feat_scales is None else feat_scales[i],
            "size": self.in_sizes[i],
            "upsample": self.out_sizes[i] > self.in_sizes[i],
            } for i in range(levels)]

        self.levels = nn.ModuleDict(
            {str(i): CAMNetModule(**l) for i,l in enumerate(level_info)})

        self.map_nc = map_nc
        self.code_nc = code_nc
        self.color_space = color_space

        init_weights(self, scale=init_scale, init_type=init_type)

        self.short_str = " ".join([f"[{i}->{o}]" for i,o in zip(self.in_sizes, self.out_sizes)])
        tqdm.write(f"Constructed CAMNet architecture: {self.short_str}")


    def forward(self, net_input, codes, loi=None, in_color_space="rgb", out_color_space="rgb"):
        """Returns a list of the outputs computed by each level.

        Args:
        net_input   -- BSxCxHxW tensor
        codes       -- list of latent codes
        loi         -- level of interest. Returns a list of outputs from only
                        this level
        """
        ########################################################################
        # Input color space conversion
        ########################################################################
        if in_color_space == self.color_space:
            level_output = net_input
        elif in_color_space == "rgb" and self.color_space == "lab":
            level_output = rgb2lab_with_dims(net_input)
        else:
            raise ValueError(f"in_color_space {in_color_space} and color_space {self.color_space} shouldn't be used")

        ########################################################################
        # Linear algebra soup
        ########################################################################
        bs = level_output.shape[0]
        feat = torch.tensor([], device="cuda:1")
        outputs = []
        for idx,(code,(_,level)) in enumerate(zip(codes, self.levels.items())):

            # This chunk of code allows parallelism across codes, by expanding
            # the inputs to the current level in the loop to match the size of
            # the codes for the current level in the zero dimension.
            level_bs = code.shape[0]
            if level_bs > bs:
                n = level_bs // bs
                if not evenly_divides(bs, level_bs):
                    raise ValueError(f"Batch size of old level {bs} must evenly divide batch size of the current level {level_bs}")

                level_output = torch.repeat_interleave(level_output, n, axis=0)
                feat = torch.repeat_interleave(feat, n, axis=0)
                bs = level_bs

            feat, level_output = level(level_output, code, feature=feat)
            outputs.append(level_output)
            if idx == loi: break

        result = outputs[loi] if loi is not None else outputs

        ########################################################################
        # Output color space conversion
        ########################################################################
        if self.color_space == out_color_space:
            return result
        elif self.color_space == "lab" and out_color_space == "rgb":
            return lab2rgb_with_dims(result)
        else:
            raise ValueError(f"color_space {self.color_space} and out_color_space {out_color_space} shouldn't be used")

class CAMNetModule(nn.Module):

    def __init__(self, map_nc=128, latent_nc=512, in_nc=3, out_nc=3, code_nc=5,
        feat_scale=1, prev_resid_nc=0, resid_nc=-1, dense_nc=-1, n_blocks=6,
        upsample_mode="nearest", act_type="leakyrelu", upsample=True,
        size=16, level=1):
        """
        Args:
        map_nc          -- number of channels on input to mapping network
        latent_nc       -- number of channels inside the mapping network
        in_nc           -- number of input channels (???)
        out_nc          -- number of output channels (???)
        code_nc         -- number of channels of input code (???)
        feat_scale      -- constant to scale input features by
        prev_resid_nc   -- number of residual channels in prior layer's RRDBs
        resid_nc        -- number of residual channels to use in RRDB blocks
        dense_nc        -- number of dense channels to use in RRDB blocks
        n_blocks        -- number of RRDB blocks in the level
        upsample_mode   -- upsampling mode
        act_type        -- activation type to use
        upsample        -- whether to upsample level features before or after
           running them through the output convolution
        """
        super(CAMNetModule, self).__init__()
        self.upsample = upsample
        self.map_nc = map_nc
        self.code_nc = code_nc
        self.feat_scale = feat_scale
        self.size = size
        self.mapping_net = MappingNet(map_nc, latent_nc, act_type)

        self.feat_net = B.conv_block(in_nc + code_nc + prev_resid_nc,
            resid_nc,
            kernel_size=3, act_type=None)

        self.style_block = B.ShortcutBlock(B.StyleBlock(
            [B.RRDB(resid_nc, kernel_size=3, gc=dense_nc, act_type=act_type, bias=True, pad_type="zero") for _ in range(n_blocks)],
            [nn.Linear(latent_nc, 2 * resid_nc) for _ in range(n_blocks)],
            B.conv_block(resid_nc, resid_nc, kernel_size=3, act_type=None)
        ))

        self.upsample_block = CAMNetUpsampling(resid_nc, upsample=upsample)
        self.out_conv = B.conv_block(resid_nc, out_nc, kernel_size=3,
            act_type="tanh")
        self.rerange_output = B.RerangeLayer()

    def forward(self, level_input, code, feature=torch.tensor([], device=device)):
        """Returns a (feature, level_output) tuple from the CAMNet module.
        [feature] is fed to the next CAMNet module in the stack, and
        [level_output] is too, but is also the output we ultimately want to make
        good.
        Args:
        net_input   -- the input to the network or the output of the prior level
        code        -- the code for the current level
        feature     -- the feature computed by the previous layer
        """



        bs, _, w, h = level_input.shape
        level_input = torch.cat([
            level_input,
            code[:, self.map_nc:].reshape(bs, self.code_nc, w, h),
            feature * self.feat_scale], dim=1)

        mapped_code = self.mapping_net(code[:, :self.map_nc])
        feature = self.feat_net(level_input)
        feature = self.style_block(feature, mapped_code)

        # If the second output should be 2x the dimension of the input, upsample
        # the input to self.out_conv, otherwise, do not!
        if self.upsample:
            feature = self.upsample_block(feature)
            out = self.out_conv(feature)
        else:
            out = self.out_conv(feature)
            feature = self.upsample_block(feature)

        return feature, self.rerange_output(out)

class MappingNet(nn.Module):
    """A mapping network for one block of CAMNet."""

    def __init__(self, map_nc, latent_nc, act_type="leakyrelu"):
        """Args:
        map_nc      -- number of channels on input to mapping network
        latent_nc   -- number of channels inside the mapping network
        act_type    -- activation type to use
        """
        super(MappingNet, self).__init__()
        self.map_nc = map_nc
        self.model = nn.Sequential(*flatten(
            [nn.Linear(map_nc, latent_nc), B.act(act_type)] +
            [(nn.Linear(latent_nc, latent_nc), B.act(act_type)) for _ in range(7)]
        ))

    def forward(self, x): return self.model(x[:, :self.map_nc])

class CAMNetUpsampling(nn.Module):
    """Produces the output of a CAMNet level."""

    def __init__(self, resid_nc, upsample=False, act_type="leakyrelu",
        upsample_kernel_mode="nearest"):
        """Args:
        resid_nc                -- number of residual channels
        upsample                -- whether to upsample or not
        upsample_kernel_mode    -- kernel mode for upsampling
        act_type                -- activation type to use
        """
        super(CAMNetUpsampling, self).__init__()
        if upsample:
            self.model = nn.Sequential(*flatten([
                B.upconv_block(resid_nc, resid_nc, act_type=act_type, mode=upsample_kernel_mode),
                B.conv_block(resid_nc, resid_nc, kernel_size=3, act_type=act_type)]))
        else:
            self.model = nn.Sequential(*flatten([
                B.conv_block(resid_nc, resid_nc, kernel_size=3, act_type=act_type),
                B.conv_block(resid_nc, resid_nc, kernel_size=3, act_type=act_type)]))

    def forward(self, x): return self.model(x)
