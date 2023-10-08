import torch.nn as nn
from collections import OrderedDict
import torch
import torch.nn.functional as F
from model import common


# conv_layer is a convolution,The function is to keep the size of the feature map unchanged
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)



def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


# pad type is include constant reflect relicate
def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


# Select a different activation function
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


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# contrast-aware channel attention module
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
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m

# Classic modules in the paper

class mmIMDModule2(nn.Module):
    def __init__(self, in_channels):
        super(mmIMDModule2, self).__init__()
        # self.ddc = self.ddistilled_channels = in_channels // 4
        #self.dc = self.distilled_channels = in_channels // 2
        #self.rc = self.remaining_channels = in_channels
        self.conv1_1 = conv_layer(in_channels, in_channels, 1)
        self.conv1_2 = conv_layer(in_channels, in_channels, 1)
        self.conv1_3 = conv_layer(in_channels, in_channels, 1)
        self.conv1_4 = conv_layer(in_channels, in_channels//2, 1)
        self.conv2_1 = conv_layer(in_channels, in_channels, 3)
        self.conv2_2 = conv_layer(in_channels, in_channels, 3)
        self.conv2_3 = conv_layer(in_channels, in_channels, 3)
        self.conv2_4 = conv_layer(in_channels, in_channels//2, 3)
        # self.act = activation('lrelu', neg_slope=0.05)
        self.act = nn.GELU()
        self.c5 = conv_layer(224, in_channels, 1)
        # self.cca = CCALayer(self.distilled_channels * 8)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        out_branch1_1 = self.act(self.conv1_1(input)+input)
        distilled_branch1_1= out_branch1_1.split(30,1)
        # print(distilled_branch1_1[0].shape)
        out_branch1_2 = self.act(self.conv1_2(out_branch1_1)+out_branch1_1)
        distilled_branch1_2 = out_branch1_2.split(30,1)
        out_branch1_3 = self.act(self.conv1_3(out_branch1_2)+out_branch1_2)
        distilled_branch1_3 = out_branch1_3.split(30,1)
        out_branch1_4 = self.conv1_4(out_branch1_3)

        out_branch2_1 = self.act(self.conv2_1(input)+input)
        # distilled_branch2_1 = torch.split(out_branch2_1, (self.distilled_channels), dim=1)
        distilled_branch2_1 = out_branch2_1.split(30,1)
        out_branch2_2 = self.act(self.conv2_2(out_branch2_1)+out_branch2_1)
        # distilled_branch2_2 = torch.split(out_branch2_2, (self.distilled_channels), dim=1)
        distilled_branch2_2 = out_branch2_2.split(30,1)
        out_branch2_3 = self.act(self.conv2_3(out_branch2_2)+out_branch2_2)
        # distilled_branch2_3 = torch.split(out_branch2_3, (self.distilled_channels), dim=1)
        distilled_branch2_3 = out_branch2_3.split(30,1)
        out_branch2_4 = self.conv2_4(out_branch2_3)
        out = torch.cat([distilled_branch1_1[0],distilled_branch1_2[0], distilled_branch1_3[0], out_branch1_4, distilled_branch2_1[0], distilled_branch2_2[0], distilled_branch2_3[0], out_branch2_4], dim=1)
        # out_fused = self.c5(self.cca(out)) + input
        out_fused = self.esa(self.c5(out))+input

        return out_fused


if __name__ == "__main__":
    from thop import profile
    import time

    time_start = time.time()
    x = torch.randn(1, 40, 100, 100)
    model = mmIMDModule2(40)
    print(model(x).shape)
    flops, params, = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)

    time_end = time.time()
    print('totally cost', time_end - time_start)