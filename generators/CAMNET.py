import math
import os
from collections import namedtuple
import torch
import torch.nn as nn
from . import blocks as B

from Utils import *

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

    def __init__(self, n_rc, upsample=False, upsample_kernel_mode="nearest",
        act_type="leakyrelu"):
        """Args:
        n_rc                    -- number of residual channels
        upsample                -- whether to upsample or not
        upsample_kernel_mode    -- kernel mode for upsampling
        act_type                -- activation type to use
        """
        super(CAMNetLevelOutput, self).__init__()
        if upsample:
            self.model = nn.Sequential(
                B.upconv_block(n_rc, n_rc, act_type=act_type, mode=upsample_kernel_mode),
                B.conv_block(n_rc, n_rc, kernel_size=3, act_type=act_type))
        else:
            self.model = nn.Sequential(
                B.conv_block(n_rc, n_rc, kernel_size=3, act_type=act_type),
                B.conv_block(n_rc, n_rc, kernel_size=3, act_type=act_type))

    def forward(self, x): return self.model(x)

class CAMNetLevel(nn.Module):

    def __init__(self, level, n_dc, n_rc, out_nc=2, map_nc=128, latent_nc=512, n_blocks=6, act_type="leakyrelu", last_level=False, prev_residual_channels=0):
        """
        Args:
        level                   -- the level of the module
        last_level              -- whether or not the module is the last level
        prev_residual_channels  -- the number of residual channels in the
                                    previous CAMNet module
        """
        super(CAMNetModule, self).__init__()
        self.mapping_net = MappingNet(map_nc, latent_nc, act_type)
        self.fea_conv = B.conv_block(in_nc + code_nc + prev_residual_channels,
                                     n_rc, kernel_size=3, act_type=None)
        self.style_block = B.ShortcutBlock(B.StyleBlock(
            [B.RRDB(n_rc, gc=n_dc, act_type=act_type) for _ in range(n_blocks)],
            [nn.Linear(latent_nc, 2 * n_rc) for _ in range(n_blocks)],
            B.conv_block(n_rc, n_rc, kernel_size=3, act_type=None)
        ))
        self.upsample = CAMNetUpsampling(n_rc,
            upsample=not (last_level and task == "colorization")))
        self.out = B.conv_block(n_rc, out_nc, kernel_size=3, act_type="tanh")

    def forward(self, level_input, codes, prev_feature=None):
        """
        Args:
        level_input     -- input to the given level
        codes           -- latent codes being learned
        prev_feature    -- the output of the previous layer
        """




class CAMNet(nn.Module):

    def __init__(self, in_nc, code_nc, out_nc, num_residual_channels,
        num_dense_channels, num_blocks, upscale=16, act_type='leakyrelu',
        upsample_kernel_mode="nearest", feat_scales=None, map_nc=128,
        latent_nc=512, no_upsample=False, use_noise_encoder=False, hr_width=256,
        hr_height=256, task='Colorization'):
        """
        in_nc
        code_nc
        out_nc
        num_residual_channels
        upscale                 -- the amount of upsampling, measured in change
                                    in side length
        """
        super(CAMNet, self).__init__()
        self.num_levels = int(math.log(upscale, 2))
        self.code_nc = code_nc
        self.feat_scales = [0.1] * (self.num_levels-1) if feat_scales is None else feat_scales
        self.out_layer = B.RerangeLayer()
        self.map_nc = map_nc
        self.use_noise_enc = use_noise_encoder
        self.task = task

        for i in range(self.num_levels):
            cur_num_dc = num_dense_channels[i]
            cur_num_rc = num_residual_channels[i]

            ####################################################################
            # Create the mapping network for the layer
            ####################################################################
            # mapping_net = ([B.sequential(nn.Linear(map_nc, latent_nc), B.act(act_type))] +
            #                [B.sequential(nn.Linear(latent_nc, latent_nc), B.act(act_type)
            #                     for _ in range(7)])
            # self.add_module(f"level_{i+1}_map", B.sequential(*mapping_net))


            if i == 0:
                fea_conv = B.conv_block(in_nc + code_nc, cur_num_rc, kernel_size=3, act_type=None)
            else:
                fea_conv = B.conv_block(in_nc + code_nc + num_residual_channels[i - 1], cur_num_rc, kernel_size=3, act_type=None)

            # RRDB blocks
            rb_blocks = [B.RRDB(cur_num_rc, kernel_size=3, gc=cur_num_dc, bias=True, pad_type='zero',
                                act_type=act_type) for _ in range(num_blocks)]
            transformations = [nn.Linear(latent_nc, 2 * cur_num_rc) for _ in range(num_blocks)]
            lr_conv = B.conv_block(cur_num_rc, cur_num_rc, kernel_size=3, act_type=None)

            style_block = B.StyleBlock(rb_blocks, transformations, lr_conv)
            # The layer that produces the feature to concatenate with the next level
            hr_conv = B.conv_block(cur_num_rc, cur_num_rc, kernel_size=3, act_type=act_type)
            out_conv = B.conv_block(cur_num_rc, out_nc, kernel_size=3, act_type="tanh")

            if i == self.num_levels - 1 and task == "Colorization":
                layer = B.conv_block(cur_num_rc, cur_num_rc, kernel_size=3, act_type=act_type)
            else:
                if no_upsample is not None and no_upsample:
                    layer = B.conv_block(cur_num_rc, cur_num_rc, kernel_size=3, act_type=act_type)
                else:
                    layer = B.upconv_block(cur_num_rc, num_rc, act_type=act_type, mode=upsample_kernel_mode)

            # noise encoder layer
            if self.use_noise_enc:
                scale_factor = 1 / (2 ** (self.num_levels - i))
                noise_en_block = B.NoiseEncoderBlock(in_ch=code_nc, out_ch=100,
                                                     input_size=[int(hr_height * scale_factor),
                                                                 int(hr_width * scale_factor)])

            self.add_module("level_%d_feat" % (i + 1), fea_conv)
            self.add_module("level_%d_style" % (i + 1), B.ShortcutBlock(style_block))
            self.add_module("level_%d_up" % (i + 1), B.sequential(layer, hr_conv))
            self.add_module("level_%d_out" % (i + 1), out_conv)
            if self.use_noise_enc:
                self.add_module("level_%d_noise_encoder" % (i + 1), noise_en_block)

    def forward(self, net_input, codes):
        if self.task == "Colorization":
            return self.forward_colorization(net_input=net_input, codes=codes)
        elif self.task == "Image_Synthesis":
            return self.forward_image_synthesis(net_input=net_input, codes=codes)
        elif self.task == "Super_Resolution":
            return self.forward_super_resolution(net_input=net_input, codes=codes)
        elif self.task == "Decompression":
            return self.forward_decompression(net_input=net_input, codes=codes)

    def forward_colorization(self, net_input, codes):
        assert len(codes) <= self.num_levels, "Number of codes should be no more than number of level of the network"
        outputs = []
        feature = None
        for i, code in enumerate(codes):
            bs, _, w, h = net_input[i].shape
            if i == 0:
                x = torch.cat((net_input[i], code[:, self.map_nc:].reshape(bs, self.code_nc, w, h)), dim=1)
            else:
                # concat with the previous level output and feature
                x = torch.cat((net_input[i], code[:, self.map_nc:].reshape(bs, self.code_nc, w, h),
                               feature * self.feat_scales[i - 1]), dim=1)

            mapped_code = getattr(self, "level_%d_map" % (i + 1))(code[:, :self.map_nc])
            feature = getattr(self, "level_%d_feat" % (i + 1))(x)
            feature = getattr(self, "level_%d_style" % (i + 1))(feature, mapped_code)
            # for the last backbone we pass through up_sample that actually preserve dimension
            if i == self.num_levels - 1:
                feature = getattr(self, "level_%d_up" % (i + 1))(feature)
            out = getattr(self, "level_%d_out" % (i + 1))(feature)

            if i < self.num_levels - 1:  # for all backbone except the last one, we double feature size
                feature = getattr(self, "level_%d_up" % (i + 1))(feature)

            outputs.append(out)

        for i in range(0, len(outputs)):
            outputs[i] = torch.cat((net_input[i], outputs[i]), 1)
        return outputs

    def forward_super_resolution(self, net_input, codes):
        assert len(codes) <= self.num_levels, "Number of codes should be no more than number of level of the network"
        outputs = []
        feature = None
        out = None
        for i, code in enumerate(codes):
            if i == 0:
                bs, _, w, h = net_input[0].shape
                x = torch.cat((net_input[0], code[:, self.map_nc:].reshape(bs, self.code_nc, w, h)), dim=1)
            else:
                bs, _, w, h = out.shape
                # concat with the previous level output and feature
                x = torch.cat((out, code[:, self.map_nc:].reshape(bs, self.code_nc, w, h), feature *
                               self.feat_scales[i - 1]), dim=1)
            mapped_code = getattr(self, "level_%d_map" % (i + 1))(code[:, :self.map_nc])
            feature = getattr(self, "level_%d_feat" % (i + 1))(x)
            feature = getattr(self, "level_%d_style" % (i + 1))(feature, mapped_code)
            feature = getattr(self, "level_%d_up" % (i + 1))(feature)
            out = getattr(self, "level_%d_out" % (i + 1))(feature)
            outputs.append(self.out_layer(out))
        return outputs

    def forward_image_synthesis(self, net_input, codes):
        assert len(codes) <= self.num_levels, "Number of codes should be no more than number of level of the network"
        outputs = []
        feature = None
        for i, code in enumerate(codes):
            bs, _, w, h = net_input[i].shape
            noise = code[:, self.map_nc:].reshape(bs, self.code_nc, w, h)
            if self.use_noise_enc:
                noise = getattr(self, "level_%d_noise_encoder" % (i + 1))(noise)
            if i == 0:
                x = torch.cat((net_input[i], noise), dim=1)
            else:
                # concat with the previous level output and feature
                x = torch.cat((net_input[i], noise, feature * self.feat_scales[i - 1]), dim=1)

            mapped_code = getattr(self, "level_%d_map" % (i + 1))(code[:, :self.map_nc])
            feature = getattr(self, "level_%d_feat" % (i + 1))(x)
            feature = getattr(self, "level_%d_style" % (i + 1))(feature, mapped_code)
            feature = getattr(self, "level_%d_up" % (i + 1))(feature)
            out = getattr(self, "level_%d_out" % (i + 1))(feature)
            outputs.append(self.out_layer(out))

        return outputs



    def forward_decompression(self, net_input, codes):
        assert len(codes) <= self.num_levels, "Number of codes should be no more than number of level of the network"
        outputs = []
        feature = None
        for level, inp in enumerate(codes):
            bs, _, w, h = net_input[level].shape
            if level == 0:
                x = torch.cat((net_input[level], codes[0][:, self.map_nc:].reshape(bs, self.code_nc, w, h)), dim=1)
            else:
                x = torch.cat((net_input[level], codes[level][:, self.map_nc:].reshape(bs, self.code_nc, w, h),
                               feature * self.feat_scales[level - 1]), dim=1)
            mapped_code = getattr(self, "level_%d_map" % (level + 1))(codes[level][:, :self.map_nc])
            feature = getattr(self, "level_%d_feat" % (level + 1))(x)
            feature = getattr(self, "level_%d_style" % (level + 1))(feature, mapped_code)
            out = getattr(self, "level_%d_out" % (level + 1))(feature)
            outputs.append(self.out_layer(out))
            if level < self.num_levels - 1:
                feature = getattr(self, "level_%d_up" % (level + 1))(feature)
        return outputs
