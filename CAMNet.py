"""The CAMNet architecture rewritten for our purposes, and to be
 - better documented
 - (nearly) stateless
 - substantially functional
NOTES:
 - read 'nc' as 'n channels' and variable names will make some sense!
"""
from collections import OrderedDict
import math
import torch
import torch.nn as nn

import Block as B
from Corruptions import *
from Data import *

def flatten(xs):
    """Returns collection [xs] after recursively flattening into a list."""
    result = []
    for x in xs:
        if isinstance(x, (list, tuple, set, nn.Sequential)):
            result += flatten(x)
        else:
            result.append(x)

    return result

camnet_arch_defaults = {
    ### LEVEL COMPOSITION SPECIFICATION
    "n_levels": 4,                        # number of levels. Leave level does colorization, 2x "super-resolution", and de-corrupting
    "base_size": 16,                      # size (height/width) of lowest resolution image fed to the network
    "in_nc": 1,                           # number of visual input channels to each level (????)
    "out_nc": 2,                          # number of visual output channels to each level (????)
    "feat_scales": None,                  # default scale of each feature
    "tasks": ["colorization",             # Each level performs a special task
        "super-resolution",
        "super-resolution",
        "super-resolution"],

    ### MappingNet SPECIFICATION
    "map_nc": 128,                        # number of channels on input to mapping network of each level
    "latent_nc": 512,                     # number of channels internal to mapping network of each level

    ### RRDB BLOCK SPECIFICATION
    "n_blocks": 6,                        # number of RRDB blocks in each level
    "resid_nc": [128, 64, 64, 64],      # number of residual channels in RRDB blocks
    "prev_resid_nc": [0, 128, 64, 64],    # number of residual channels in the previous set of RRDB block
    "dense_nc": [256, 192, 128, 64],    # number of dense channels in RRDB blocks

    ### MISCELLANEOUS
    "act_type": "leakyrelu",              # the activation unless otherwise intelligently hardcoded
    "upsample": True,
}

def original_to_rewritten_config(opt):
    """Returns a dictionary with the keys of [camnet_arch_defaults] built from
    an [opt] dictionary generated by the original version.
    """
    n_levels = int(math.log(opt["scale"], 2))
    result = camnet_arch_defaults
    result.update({
        "n_levels": n_levels,
        "base_size": 16,
        "in_nc": opt["network_G"]["in_nc"],
        "out_nc": opt["network_G"]["out_nc"],
        "tasks": [opt["task"].lower() for _ in range(n_levels)],
        "map_nc": opt["network_G"]["map_nc"],
        "latent_nc": opt["network_G"]["latent_nc"],
        "n_blocks": opt["network_G"]["num_blocks"],
        "resid_nc": opt["network_G"]["num_residual_channels"],
        "prev_resid_nc": [0] + opt["network_G"]["num_residual_channels"][:-1],
        "dense_nc": opt["network_G"]["num_dense_channels"],
        "act_type": "leakyrelu",
        "upsample": not opt["network_G"]["no_upsample"],
    })
    print(result)
    return result

class CAMNet(nn.Module):
    """CAMNet rewritten to be substantially stateless, better-documented, and
    optimized for ISICLE.

    Args:
    n_levels                    -- number of CAMNet levels
    base_size                   -- size of network input
    in_nc                       -- number of input channels
    out_nc                      -- number of output channels
    feat_scales                 -- amount by which to scale features, or None
    map_nc                      -- number of input channels to mapping net
    latent_nc                   -- number of channels inside the mapping net
    n_blocks                    -- number of RRDB blocks inside each level
    resid_nc                    -- list of numbers of residual channels in RRDB
                                    blocks for each CAMNet level
    dense_nc                    -- list of numbers of dense channels in RRDB
                                    blocks for each CAMNet level
    act_type                    -- activation type
    """

    def __init__(self, n_levels=4, base_size=16, in_nc=1, out_nc=2,
        feat_scales=None,
        map_nc=128, latent_nc=512, n_blocks=6,
        resid_nc=[128, 64, 64, 64], prev_resid_nc=[0, 128, 64, 64],
        dense_nc=[256, 192, 128, 64],
        act_type="leakyrelu",
        upsample=True):
        super(CAMNet, self).__init__()
        level_info = [{
            "n_blocks": n_blocks,
            "map_nc": 128,
            "latent_nc": 512,
            "in_nc": 1,
            "out_nc": 2,
            "dense_nc": dense_nc[i],
            "resid_nc": resid_nc[i],
            "prev_resid_nc": 0 if i == 0 else resid_nc[i-1],
            "level": i,
            "feat_scale": (.1 * i) if feat_scales is None else feat_scales[i],
            "upsample_output": True,
            "size": base_size * 2 ** i,
            } for i in range(n_levels)]

        self.levels = nn.ModuleDict({
            f"level {i}": CAMNetModule(**l) for i,l in enumerate(level_info)
        })

    def forward(self, net_input, codes):
        """Returns a list of the outputs computed by each level.
        Unfortunately, it"s difficult to make this stateless in a clean way due
        to the fact that we need to return the output of each CAMNet level, so
        we need to collect them, and recursion isn't going to make this much
        simpler.
        """
        level_output = net_input
        feature = torch.tensor([])
        outputs = []

        for code,level in zip(codes, self.levels.values()):
            feature, level_output = level(level_output, code, feature=feature)
            outputs.append(level_output)

        return outputs

class CAMNetModule(nn.Module):

    def __init__(self, map_nc=128, latent_nc=512, in_nc=1, out_nc=2, code_nc=5,
        feat_scale=1, prev_resid_nc=0, resid_nc=-1, dense_nc=-1, n_blocks=6,
        upsample_mode="nearest", act_type="leakyrelu", upsample_output=True,
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
        upsample_output  -- whether to upsample level features before or after
                            running them through the output convolution
        """
        super(CAMNetModule, self).__init__()
        self.upsample_output = upsample_output
        self.map_nc = map_nc
        self.feat_scale = feat_scale
        self.mapping_net = MappingNet(map_nc, latent_nc, act_type)
        self.feat_net = B.conv_block(in_nc + code_nc + prev_resid_nc,
            resid_nc,
            kernel_size=3, act_type=None)
        self.style_block = B.ShortcutBlock(B.StyleBlock(
            [B.RRDB(resid_nc, gc=dense_nc, act_type=act_type) for _ in range(n_blocks)],
            [nn.Linear(latent_nc, 2 * resid_nc) for _ in range(n_blocks)],
            B.conv_block(resid_nc, resid_nc, kernel_size=3, act_type=None)
        ))
        self.upsample = CAMNetUpsampling(resid_nc, upsample=True)
        self.out_conv = B.conv_block(resid_nc, out_nc, kernel_size=3,
            act_type="tanh")
        self.rerange_output = B.RerangeLayer()

    def forward(self, level_input, code, feature=torch.tensor([])):
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

        print("CODE SHAPE", code.shape, code[:, self.map_nc:])


        level_input = torch.cat([
            level_input,
            code[:, self.map_nc:].reshape(bs, self.map_nc, w, h),
            feature * self.feat_scale], dim=1)

        mapped_code = self.mapping_net(code[:, :self.map_nc])
        feature = self.feat_net(level_input)
        feature = self.style_block(feature, mapped_code)

        # If the second output should be 2x the dimension of the input, upsample
        # the input to self.out_conv, otherwise, do not!
        if self.upsample_output:
            feature = self.upsample(feature)
            out = self.out_conv(feature)
        else:
            out = self.out_conv(feature)
            feature = self.upsample(feature)

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

if __name__ == "__main__":
    data_tr, data_eval = get_data_splits("cifar10", "test", [16, 32])
    corruption = get_non_learnable_batch_corruption(grayscale=True)
    transform = get_gen_augs()
    data_tr = GeneratorDataset(data_tr, transform)
    G = CAMNet()

    loader = DataLoader(data_tr, batch_size=4)
    for x,ys in loader:
        code = torch.rand(x.shape[0], 16, 16)
        fx = G(x, code)
