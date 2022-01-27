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
from utils.NestedNamespace import NestedNamespace

def flatten(xs):
    """Returns collection [xs] after recursively flattening into a list."""
    result = []
    for x in xs:
        if isinstance(x, (list, tuple, set, nn.Sequential)):
            result += flatten(x)
        else:
            result.append(x)

    return result

# class SimpleNet(nn.Module):
#
#     def __init__(self, num_):


class CAMNet(nn.Module):
    """CAMNet rewritten to be substantially stateless, better-documented, and
    optimized for ISICLE.

    Args:
    n_levels                    -- number of CAMNet levels
    base_size                   -- size of network input
    code_nc                     -- number of code channels
    in_nc                       -- number of input channels
    out_nc                      -- number of output channels
    map_nc                      -- number of input channels to mapping net
    latent_nc                   -- number of channels inside the mapping net
    resid_nc                    -- list of numbers of residual channels in RRDB
                                    blocks for each CAMNet level
    dense_nc                    -- list of numbers of dense channels in RRDB
                                    blocks for each CAMNet level
    n_blocks                    -- number of RRDB blocks inside each level
    act_type                    -- activation type
    feat_scales                 -- amount by which to scale features, or None
    """

    def __init__(self, n_levels=4, base_size=16, code_nc=5, in_nc=3, out_nc=3,
        map_nc=128, latent_nc=512, resid_nc=[128, 64, 64, 64],
        dense_nc=[256, 192, 128, 64], n_blocks=6, act_type="leakyrelu",
        feat_scales=None):
        super(CAMNet, self).__init__()
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
            "size": base_size * 2 ** i,
            } for i in range(n_levels)]

        self.levels = nn.ModuleDict({
            f"level {i}": CAMNetModule(**l) for i,l in enumerate(level_info)
        })
        self.map_nc = map_nc
        self.code_nc = code_nc
        self.base_size = base_size

    def get_z_dims(self):
        """Returns a list of tuples, where the ith tuple is the shape of the
        latent code required for the ith level without expansion to the batch
        dimension.
        """
        return [(self.map_nc + self.code_nc * (self.base_size * (2 ** l)) ** 2,)
                for l in range(len(self.levels))]

    def parse_args_to_dict(args, resolutions):
        """Returns a dictionary containing , and a list of unparsed arguments.

        Args:
        args        -- command line arguments, *some* of which may be relevant
                        for CAMNet
        resolutions -- list of resolutions, one for each level
        """
        P = argparse.ArgumentParser(description="CAMNet argument parsing")
        P.add_argument("--code_nc", default=5, type=int,
            help="number of code channels")
        P.add_argument("--in_nc", default=3, type=int,
            help="number of input channels")
        P.add_argument("--out_nc", default=3, type=int,
            help=" number of output channels")
        P.add_argument("--map_nc", default=128, type=int,
            help="number of input channels to mapping net")
        P.add_argument("--latent_nc", default=512, type=int,
            help="number of channels inside the mapping net")
        P.add_argument("--resid_nc", default=[128, 64, 64, 64], type=int,
            nargs="+",
            help="list of numbers of residual channels in RRDB blocks for each CAMNet level")
        P.add_argument("--dense_nc", default=[256, 192, 128, 64], type=int,
            nargs="+",
            help="list of numbers of dense channels in RRDB blocks for each CAMNet level")
        P.add_argument("--n_blocks", default=6, type=int,
            help="number of RRDB blocks inside each level")
        P.add_argument("--act_type", default="leakyrelu",
            choices=["leakyrelu"],
            help="activation type")
        P.add_argument("--feat_scales", default=None, type=int,
            help="amount by which to scale features, or None")

        args, unparsed_args = P.parse_known_args(args)


        resolution_info = {
            "n_levels": len(resolutions) - 1, # Note that every level upsamples
            "base_size": resolutions[0]
        }

        args = NestedNamespace.leaf_union(args, resolution_info)

        # Ensure the number of levels checks out with the given numbers of
        # residual and dense channels
        if args.n_levels > len(args.resid_nc):
            raise ValueError(f"--resid_nc must contain {args.n_levels} values but contained {len(args.resid_nc)}")
        if args.n_levels > len(args.dense_nc):
            raise ValueError(f"--dense_nc must contain {args.n_levels} values but contained {len(args.dense_nc)}")
        if not len(args.resid_nc) == len(args.dense_nc):
            raise ValueError(f"Number of levels inconsistent between --resid_nc {len(args.resid_nc)} and --dense_nc {len(args.dense_nc)}")

        args.resid_nc = args.resid_nc[:args.n_levels]
        args.dense_nc = args.dense_nc[:args.n_levels]

        return NestedNamespace.to_dict(args), unparsed_args

    def forward(self, net_input, codes, loi=float("inf")):
        """Returns a list of the outputs computed by each level.

        Args:
        net_input   -- BSxCxHxW tensor
        codes       -- list of latent codes
        loi         -- level of interest. Returns a list of outputs from only
                        this level
        """
        level_output = net_input
        feat = torch.tensor([], device=device)
        outputs = []

        for idx,(code,(k,level)) in enumerate(zip(codes, self.levels.items())):
            feat, level_output = level(level_output, code, feature=feat)
            outputs.append(level_output)

            if idx == loi:
                return outputs[-1]

        return outputs

class CAMNetModule(nn.Module):

    def __init__(self, map_nc=128, latent_nc=512, in_nc=3, out_nc=3, code_nc=5,
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
        self.code_nc = code_nc
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

# if __name__ == "__main__":
#     data_tr, data_eval = get_data_splits("cifar10", "test", [16, 32])
#     corruption = get_non_learnable_batch_corruption(grayscale=True)
#     transform = get_gen_augs()
#     data_tr = GeneratorDataset(data_tr, transform)
#     G = CAMNet(n_levels=1)
#
#     loader = DataLoader(data_tr, batch_size=4)
#     for x,ys in loader:
#
#
#         s = (x.shape[0], 128 + 5 * 16 * (2 ** 0) * 16 * (2 ** 0))
#         print(s)
#         code = torch.rand(*s)
#         fx = G(x, [code])
