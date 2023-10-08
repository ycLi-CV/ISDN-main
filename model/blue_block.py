'''
This repository is used to implement all upsamplers(only x4) and tools for Efficient SR
@author
    LI Zehyuan from SIAT
    LIU yingqi from SIAT
'''

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common
import math
import functools
from einops import rearrange,reduce
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.archs.arch_util import default_init_weights
from model import fattention as a
from basicsr.utils.registry import ARCH_REGISTRY
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)
def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_norm=False, bn_kwargs=None):
        super(DepthWiseConv, self).__init__()

        self.dw = torch.nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_ch,
                bias=bias,
                padding_mode=padding_mode,
        )

        self.pw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, input):
        out = self.dw(input)
        out = self.pw(out)
        return out


class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )
        #print(out_channels)

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ESA(nn.Module):
    def __init__(self, num_feat=64, conv=BSConvU, p=0.25):
        super(ESA, self).__init__()
        f = num_feat // 4
        BSConvS_kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': p}
        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv2 = conv(f, f, 3, 2, 0)
        self.conv3 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_ = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()
        #self.SMU=SMU()

    def forward(self, input):
        c1_ = (self.conv1(input))
        c1 = self.conv2(c1_)
        v_max = self.maxPooling(c1)
        v_range = self.GELU(self.conv_max(v_max))
        c3 = self.GELU(self.conv3(v_range))
        #v_range = self.SMU(self.conv_max(v_max))
        #c3 = self.SMU(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4((c3 + cf))
        m = self.sigmoid(c4)

        return input * m

def mean_block(F):
    F = reduce(
        F, 'b c bnum l -> b c bnum ', "mean"
    )
    F = torch.unsqueeze(F, 3)
    return F
def stdv_block(F, F_mean):
    _, _, l, _ = F.size()
    F = (F - F_mean).pow(3).sum(3, keepdim=True) / l
    return F


class DSDB(nn.Module):
    def __init__(self, in_channels,_, conv=BSConvU, conv1=common.default_conv, p=0.25):
        super(DSDB, self).__init__()

        self.add_inchannel=int(in_channels//2)

        self.c1_1 = conv(in_channels, in_channels//2, kernel_size=3,stride=1, padding=1, dilation=1)
        self.c1_2 = conv(in_channels//2, in_channels, kernel_size=3)

        self.c2_1 = conv(in_channels, in_channels//2, kernel_size=3,stride=1, padding=3, dilation=3)
        self.c2_2 = conv(in_channels//2, in_channels, kernel_size=3)

        self.c3_1 = conv(in_channels, in_channels//2, kernel_size=3,stride=1, padding=5, dilation=5)
        self.c3_2 = conv(in_channels//2, in_channels, kernel_size=3)

        self.c3 = conv(in_channels, in_channels//2, kernel_size=3,stride=1, padding=1, dilation=1)
    
        self.d1 = conv1(in_channels, in_channels//2 ,kernel_size=1)
        self.d2 = conv1(self.add_inchannel*3, in_channels//2, kernel_size=1)
        self.d3 = conv1(self.add_inchannel*3, in_channels// 2, kernel_size=1)

        self.act = nn.GELU()
        self.c5 = nn.Conv2d(in_channels*2, in_channels, 1)

        self.esa = ESA(in_channels)
        self.cca = CCALayer(in_channels)

    def forward(self, input):
        distilled_c1 = self.act(self.d1(input))
        r1_1 = (self.c1_1(input))
        r1_2 = (self.c1_2(r1_1))
        input1=self.act(input+r1_2)

        concat1=torch.cat([distilled_c1,input1],dim=1)
        distilled_c2 = self.act(self.d2(concat1))
        r2_1 = (self.c2_1(input1))
        r2_2 = (self.c2_2(r2_1))
        input2 = self.act(input1 + r2_2)

        concat2 = torch.cat([distilled_c2, input2], dim=1)
        distilled_c3 = self.act(self.d3(concat2))
        r3_1 = (self.c3_1(input2))
        r3_2 = (self.c3_2(r3_1))
        input3 = self.act(input2 + r3_2)

        r_c4 = self.act(self.c3(input3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)

        out_fused = self.esa(out)
        out_fused = self.cca(out_fused)

        return out_fused + input


if __name__ == "__main__":
    from thop import profile
    import time

    time_start = time.time()
    x = torch.randn(1,58, 100, 100)
    model =DSDB(58,58)
    print(model(x).shape)
    flops, params = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)

    time_end = time.time()
    print('totally cost', time_end - time_start)