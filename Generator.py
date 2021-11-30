import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, n_features=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(n_features, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(n_features + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(n_features + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(n_features + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(n_features + 4 * gc, n_features, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, n_features, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(n_features, gc)
        self.RDB2 = ResidualDenseBlock_5C(n_features, gc)
        self.RDB3 = ResidualDenseBlock_5C(n_features, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_size, out_size, z_channels, in_channels=3, out_channels=3, n_features=128, n_blocks=6, gc=32):
        """
        Args:
        in_size         -- side length of input images
        out_size        -- side length of output images
        in_channels     -- number of input channels
        out_channels    -- number of output channels
        n_features      -- number of intermediate features
        n_blocks        -- number of RRDB blocks
        gc              -- channels for each growth(??)
        """
        super(RRDBNet, self).__init__()
        RRDB_block_f = fuchannelstools.partial(RRDB, n_features=n_features, gc=gc)

        self.conv_one = nn.Conv2d(in_channels + z_channels, n_features, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, n_blocks)
        self.trunk_conv = nn.Conv2d(n_features, n_features, 3, 1, 1, bias=True)

        ########################################################################
        # Create the upsampling layers. Each will scale the input
        # representation's size by a factor of two.
        ########################################################################
        n_upsamples = math.log2(out_size / in_size)
        if not int(n_upsamples) == n_upsamples:
            raise ValueError("'in_size' and 'out_size' must be powers of two")
        self.upconvs = [nn.Conv2d(n_features, n_features, 3, 1, 1, bias=True)
                        for _ in range(n_upsamples)]

        self.HRconv = nn.Conv2d(n_features, n_features, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(n_features, out_channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, z):
        x = torch.cat([x, z], 1)
        fea = self.conv_one(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        for upconv in self.upconvs:
            upsampled = F.interpolate(fea, scale_factor=2, mode='nearest')
            fea = self.lrelu(upconv(upsampled))

        return self.conv_last(self.lrelu(self.HRconv(fea)))
