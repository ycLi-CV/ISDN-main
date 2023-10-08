import torch
from model import common
import torch.nn as nn

from model.attention import *


class CRA_layer(nn.Module):
    def __init__(self, channel, conv=common.default_conv):
        super(CRA_layer, self).__init__()
        self.conv1_1 = conv(channel, channel, 1)
        self.conv1_2 = conv(channel, channel, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.body = nn.Sequential(*[
            conv(1, 2, 3),
            nn.ReLU(),
            conv(2, 1, 3),
            nn.Sigmoid()
        ])

    def forward(self, x):
        w = self.avg_pool(self.conv1_1(x)).squeeze(-1)
        x1 = self.conv1_2(x)
        w_ = w.transpose(-1, -2)
        q = torch.bmm(w,w_).unsqueeze(1)
        q = self.body(q).squeeze(1)
        k = torch.bmm(q,w).unsqueeze(-1)
        return x1 * k

class ChannelAttention_c(nn.Module):
    def __init__(self, channel, retio=8,conv = common.default_conv):
        super(ChannelAttention_c, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channel, channel // retio, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // retio, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x_ = self.avg_pool(x)
        n1 = self.body(x_ *(1+x_.sum()))
        return x * n1
class PA(nn.Module):
    def __init__(self,channel=64,conv = common.default_conv):
        super(PA, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        pass
    def forward(self,x):
        n1 = self.body(x)
        return x * n1
#
class ChannelAttention_c_1(nn.Module):
    def __init__(self, in_channel, retio=8, conv=common.default_conv):
        super(ChannelAttention_c_1, self).__init__()
        self.conv1_1 = conv(64, 64, 1)
        self.conv1_2 = conv(64, 64, 1)

        self.conv2_1 = conv(1, 4, 3)
        self.conv2_1 = conv(4, 4, 3)
        self.conv2_1 = conv(4, 1, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)

        B, c, h, w = x1.size()
        avg_out = self.avg_pool(x1)
        avg_out_1 = avg_out.view(B, 64, 1)
        avg_out_2 = avg_out.view(B, 1, 64)
        out = torch.bmm(avg_out_1, avg_out_2)
        out2 = torch.ones(B, c, 1).cuda()
        out3 = torch.bmm(out, out2)
        out3 = out3.view(B, c, 1, 1)

        return x2 * out3

#  SA   (common)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 6, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(6, 6, kernel_size, padding=padding, bias=False)
        self.conv3 = nn.Conv2d(6, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        op1 = self.conv1(x)
        op2 = self.conv2(op1)
        op3 = self.conv3(op2 + op1)
        return self.sigmoid(op3)

# CBAM
class CBAM(nn.Module):
    def __init__(self,in_channel=64,retio=4,kernel_size=3):
        super(CBAM, self).__init__()
        self.CA = CCALayer(in_channel)
        self.SA = SpatialAttention()
    def forward(self, x):
        x = self.CA(x)
        x = x * self.SA(x)
        return x
# CA (common)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


#  CCA  IMDN (CA)
def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))
def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
#channel attention module
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


# b_up
class B_up(nn.Module):
    def __init__(self, channels=64, scale=2, CAtype="CA",kernel_size=3,conv=common.default_conv):
        super(B_up, self).__init__()

        self.conv3_1 = conv(channels, channels * scale, kernel_size, groups=channels)
        self.conv3_2 = conv(channels, channels * scale, kernel_size, groups=channels)

        self.conv3_3 = conv(channels * scale, channels * (scale**2), kernel_size, groups=channels)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        if CAtype=="CRA":
            self.CA = CRA_layer(64)
        elif CAtype=="CCA":
            self.CA = CCALayer(64)
        else:
            self.CA = CALayer(64)

    def forward(self, x):

        op2_1 = self.conv3_1(x)
        op2_2 = self.conv3_2(self.CA(x))

        op2 = op2_1 + op2_2
        op4 = self.conv3_3(op2)

        op1 = self.pixel_shuffle(op4)
        return op1
# csb block
class CSB(nn.Module):
    def __init__(self, channels, kernel_size,conv=common.default_conv):
        super(CSB, self).__init__()
        self.conv3_1 = conv(channels, channels // 2, kernel_size)
        self.conv3_2 = conv(channels, channels // 2, kernel_size)
        self.conv3_3 = conv(channels, channels // 2, kernel_size)

        self.conv3_4 = conv(channels // 2, channels // 4, kernel_size)
        self.conv3_5 = conv(channels // 2, channels // 4, kernel_size)
        self.conv3_6 = conv(channels // 2, channels // 4, kernel_size)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        op1 = self.relu(self.conv3_1(x))
        op2 = self.relu(self.conv3_2(x))
        op3 = self.relu(self.conv3_3(x))

        ip1 = op1 + op2
        ip2 = op2 + op3

        op4 = self.relu(self.conv3_4(ip1))
        op5 = self.relu(self.conv3_5(ip2))

        ip3 = torch.cat((op4,op5),1)
        op6 = self.relu(self.conv3_6(ip3))

        ip4 = torch.cat((op6+op4, op5), 1)
        ip5 = torch.cat((ip4+op1, op2), 1)

        return ip5


# res block
class RCAB(nn.Module):
    def __init__(
        self, conv=common.default_conv, n_feat=64 , kernel_size=3,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        # modules_body.append(CALayer(n_feat))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)

        res += x
        return res

# EEB
class EEB(nn.Module):
    def __init__(self, channels=64, kernel_size=3,conv=common.default_conv):
        super(EEB, self).__init__()
        self.conv3_1 = conv(channels, 16, kernel_size)
        self.conv3_2 = conv(channels, 32, kernel_size)
        self.conv3_3 = conv(channels, 48, kernel_size)
        self.conv3_4 = conv(channels, 64, kernel_size)
        self.relu = nn.LeakyReLU(0.05)
        self.PA = CCALayer(64)

    def forward(self, x):

        op1 = self.relu(self.conv3_1(x))
        op1 = torch.cat((op1, x[:, 0:48, :, :]), 1)

        op2 = self.relu(self.conv3_2(op1))
        op2 = torch.cat((op2, x[:, 0:32, :, :]), 1)

        op3 = self.relu(self.conv3_3(op2))
        op3 = torch.cat((op3, x[:, 0:16, :, :]), 1)

        op4 = self.PA(self.relu(self.conv3_4(op3)))

        return op4 + x


# LEEB2
class EEB2(nn.Module):
    def __init__(self, channels=64, kernel_size=3,conv=common.default_conv):
        super(EEB2, self).__init__()
        self.conv1_1 = conv(channels, channels, 1)
        self.conv3_1 = conv(channels, channels//4, kernel_size)
        self.conv3_2 = conv(channels, channels//2, kernel_size)
        self.conv3_3 = conv(channels, channels-channels//4, kernel_size)

        self.conv1_2 = conv(channels, channels, 1)

        self.relu = nn.LeakyReLU(0.05)
        self.CA = CCALayer(64)


    def forward(self, x):
        op0 = self.relu(self.conv1_1(x))

        op1 = self.relu(self.conv3_1(op0))
        ip1 = torch.cat((op1, op0[:, 0:48, :, :]), 1)

        op2 = self.relu(self.conv3_2(ip1))
        ip2 = torch.cat((op2, op0[:, 0:32, :, :]), 1)

        op3 = self.relu(self.conv3_3(ip2))
        ip3 = torch.cat((op3, op0[:, 0:16, :, :]), 1)
        op3 = self.conv1_2(ip3)

        return self.CA(op3) + x
# eeb3
# class EEB3(nn.Module):
#     def __init__(self, c = 4,channels=64, kernel_size=3,conv=common.default_conv):
#         super(EEB3, self).__init__()
#         self.channels = channels
#         self.c = c
#
#         body = [ conv(channels, channels//c, kernel_size) for _ in range(c)]
#         self.body = nn.Sequential(*body)
#         self.conv1_1 = conv(channels, channels, 1)
#
#         self.relu = nn.LeakyReLU(0.05)
#
#         self.PA = CCALayer(64)
#
#
#     def forward(self, x):
#         op1 = x
#         for name, midlayer in self.body._modules.items():
#             op2 = self.relu(midlayer(op1))
#             op1 = torch.cat((op2,op1[:,:self.channels-self.channels//self.c,:,:]),1)
#
#         op3 = self.conv1_1(op1)
#         return self.PA(op3) + x


class EEB3(nn.Module):

    def __init__(self, channels=64, kernel_size=3,conv=common.default_conv):
        super(EEB3, self).__init__()
        self.conv3_0 = conv(channels, channels//4, kernel_size)
        self.conv3_1 = conv(channels, channels//4, kernel_size)
        self.conv3_2 = conv(channels, channels//4, kernel_size)
        self.conv3_3 = conv(channels, channels//4, kernel_size)

        self.conv1_1 = conv(channels, channels, 1)

        self.relu = nn.LeakyReLU(0.05)
        self.CA = CCALayer(64)

    def forward(self, x):
        op0 = self.relu(self.conv3_0(x))
        ip0 = torch.cat((op0, x[:, 0:48, :, :]), 1)

        op1 = self.relu(self.conv3_1(ip0))
        ip1 = torch.cat((op1, ip0), 1)

        op2 = self.relu(self.conv3_2(ip1))
        ip2 = torch.cat((op2, ip1[:, 0:48, :, :]), 1)

        op3 = self.relu(self.conv3_3(ip2))
        ip3 = torch.cat((op3, ip2[:, 0:48, :, :]), 1)

        op3 = self.conv1_1(ip3)
        return self.CA(op3) + x

# cs group
class CSG(nn.Module):
    def __init__(self, channels=64, kernel_size=3, conv=common.default_conv):
        super(CSG, self).__init__()
        # self.block1 = CSB(channels, kernel_size)
        # self.block2 = CSB(channels, kernel_size)
        self.block1 = EEB3(4,channels,kernel_size)
        self.block2 = EEB3(4,channels,kernel_size)
        self.block3 = EEB3(4, channels, kernel_size)
        self.block4 = EEB3(4, channels, kernel_size)
    def forward(self, x):
        op1 = self.block1(x)
        op2 = self.block2(op1)
        op3 = self.block3(op2)
        return op2 + x




def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1,conv_layer = common.default_conv):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(conv, pixel_shuffle)


class NGFF(nn.Module):
    def __init__(self, channels=64, GroupNum=4, conv=common.default_conv):
        super(NGFF, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(4)
        self.conv1_1 = conv((channels//16) * GroupNum, channels, 1)
        self.conv2_1 = conv(channels, channels , 2 ,stride=2)
        self.relu = nn.LeakyReLU(0.05)
    def forward(self,x):
        x_up = self.pixel_shuffle(x)
        x_up = self.conv1_1(x_up)
        x_d1 = self.relu(self.conv2_1(x_up))
        x_d2 = self.conv2_1(x_d1)
        return x_d2


class MK(nn.Module):
    def __init__(self, channels=64, GroupNum=16, conv=common.default_conv):
        super(MK, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(4)

        self.conv1_1 = conv((channels//16) * GroupNum, channels, 1)
        self.conv2_1 = conv(channels, channels , 2 ,stride=2)
        # self.conv2_2 = conv(channels, channels, 2, stride=2)

        self.relu = nn.LeakyReLU(0.05)


    def forward(self,x):
        x_up = self.pixel_shuffle(x)
        x_up = self.conv1_1(x_up)
        x_d1 = self.relu(self.conv2_1(x_up))
        x_d2 = self.conv2_1(x_d1)
        return x_d2

# asymmetric(al)   encode  decode
class ASEDBlocK(nn.Module):
    def __init__(self, channels=64,  conv=common.default_conv):
        super(ASEDBlocK, self).__init__()
        # encode
        self.conv3_1 = conv(channels, channels // 2, 3)
        self.conv3_2 = conv(channels, channels // 4, 3)
        self.conv3_3 = conv(channels // 2, channels // 4, 3)
        # decode
        self.conv3_4 = conv(channels // 2, channels // 2, 3)

        #
        self.relu = nn.LeakyReLU(0.5)

    def forward(self, x):
        # encode
        opBranch1 = self.conv3_1(x)
        opBranch2 = self.conv3_2(x)
        op2 =self.conv3_3(opBranch1)

        ip1 = torch.cat((opBranch2,op2),1)
        # decode
        op3 = self.conv3_4(ip1)
        res = torch.cat((op3, opBranch1),1)
        return res + x

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)

class GradConv(nn.Conv2d):
    def __init__(self):
        super(GradConv, self).__init__(3, 1, kernel_size=3,padding=3//2,bias=False)
        k = torch.FloatTensor([[[[1, 1, 1],
                              [1, -8, 1],
                              [1, 1, 1]]]])
        k = torch.cat((k, k, k), 1)
        k = torch.cat((k, k, k), 0)
        self.weight.data = k
        for p in self.parameters():
            p.requires_grad = False





if __name__ == "__main__":
    from thop import profile
    import time

    time_start = time.time()
    x = torch.randn(1, 64, 100, 100)
    model = ASEDBlocK()
    flops, params = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)


    time_end=time.time()
    print('totally cost',time_end-time_start)


  # zb = self.conv3_1_1(input)
        # for name, midlayer in self.body._modules.items():
        #     if name == '0':
        #         op2 = midlayer(op1)
        #         z = op2
        #     else:
        #         op2 = midlayer(op1)
        #         z = torch.cat((z,op2),1)
        #         op2 = self.conv1_1(op2)
        #         op2 = torch.cat((op2, zb[:, :8, :, :]), 1)
        #         op2 = midlayer(op2)
        #         zb = zb[:, 8:, :, :]

        # for name, midlayer in self.body._modules.items():
        #     if name == '0':
        #         op2 = midlayer(op1)
        #         z = op2
        #     else:
        #         op2 = midlayer(op2)
        #         z = torch.cat((op2,z),1)
        #
        # for name, midlayer in self.fn._modules.items():
        #     if name == '0':
        #         q = midlayer(z[:,:64,:,:])
        #         z = z[:,64:,:,:]
        #         op = "c"
        #     else:
        #         if op=="c":
        #             p = midlayer(z[:,:64,:,:])
        #             z = z[:, 64:, :, :]
        #             c = torch.cat((q,p),1)
        #             op = "f"
        #         else:
        #             q = rule(c)
        #             op = "c"