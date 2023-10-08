import torch
import torch.nn as nn
from einops import rearrange, reduce
import torch.nn.functional as F
from torch.nn import Softmax


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert (F.dim() == 4)
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


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class eca_layer(nn.Module):
    def __init__(self, channel, k_size):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.05)

    def forward(self, x):
        b, c, _, _ = x.size()  # x=[1,64,100,100]
        y = self.avg_pool(x)  # y=[1,64,1,1]
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)

        return x

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

class DeepWise_Pool(torch.nn.MaxPool1d):
    def __init__(self, channels):
        super(DeepWise_Pool, self).__init__(channels)
        self.kernel_size = channels//4
        self.stride = 1

    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        #print(input.shape)
        #pooled = torch.nn.functional.max_pool1d(input, self.kernel_size, self.stride,
                                                #self.padding, self.dilation, self.ceil_mode,
                                                #self.return_indices)
        pooled = nn.AdaptiveAvgPool1d(1)
        output=pooled(input)
        _, _, c = output.size()
        output = output.permute(0, 2, 1)
        return output.view(n, c, w, h)

class FSA(nn.Module):
    def __init__(self,inchannels=64):
        super(FSA,self).__init__()
        self.avg_pool = DeepWise_Pool(inchannels)
        self.conv = nn.Conv2d(inchannels, inchannels, 1)
        self.group_conv = nn.Sequential(
        nn.Conv2d(1 ,1, 3,padding=1,stride=1,dilation=1),
            nn.GELU(),
        nn.Conv2d(1, 1, 3,padding=3,stride=1,dilation=3),
            nn.GELU(),
        nn.Conv2d(1, 1, 3,padding=5,stride=1,dilation=5),
            nn.GELU(),
        nn.Conv2d(1, 1, 3,padding=7,stride=1,dilation=7),
            nn.GELU()
        )

        self.conv1=nn.Conv2d(1,inchannels,kernel_size=3,padding=1)
        self.act=nn.Sigmoid()


    def forward(self, x):
        opt1=self.conv(x)
        opt2=self.avg_pool(opt1)
        opt3=self.group_conv(opt2)
        #print(opt3.shape)
        opt4=self.act(self.conv1(opt3))
        return x * opt4

class FSA1(nn.Module):
    def __init__(self,inchannels=64,conv1=BSConvU):
        super(FSA1,self).__init__()
        self.avg_pool = DeepWise_Pool(inchannels//4)
        #self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv1 = nn.Conv2d(inchannels, 16, 1)
        #self.conv2 = nn.Conv2d(16,16, 3, 2, 0)
        self.group_conv = nn.Sequential(
        conv1(1,16, 3,padding=1,stride=1,dilation=1),
        conv1(16,16, 3,padding=3,stride=1,dilation=3),
        conv1(16, 16, 3, padding=5, stride=1, dilation=5),
        conv1(16, 1, 3, padding=1, stride=1, dilation=1),
            nn.GELU()
        )

        self.conv3=nn.Conv2d(1,inchannels,kernel_size=3,padding=1)
        self.act=nn.Sigmoid()


    def forward(self, x):
        opt1=self.conv1(x)
        opt2=self.avg_pool(opt1)
        opt3=self.group_conv(opt2)
        #print(opt3.shape)
        opt4=self.act(self.conv3(opt3))
        return x * opt4


class ESA(nn.Module):
    def __init__(self, num_feat=64, conv=nn.Conv2d, p=0.25):
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

    def forward(self, input):
        c1_ = (self.conv1(input))
        c1 = self.conv2(c1_)# print(c1.shape) c1[1,16,49,49]
        v_max = self.maxPooling(c1)#print(v_max.shape) v_max[1,16,15,15]
        v_range = self.GELU(self.conv_max(v_max))#print(v_range.shape) v_range[1,16,13,13]
        c3 = self.GELU(self.conv3(v_range))#c3=[1,16,11,11]
        c3 = self.conv3_(c3)#print(c3.shape) c3=[1,16,9,9]
        c3 = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)#print(c3.shape) c3=[1,16,100,100]
        cf = self.conv_f(c1_)#print(cf.shape)cf[1,16,100,100] unclear why do this step? to order c3 +
        c4 = self.conv4((c3 + cf))#c4[1,64,100,100]
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


class LDCA(nn.Module):
    def __init__(self, channels, window_size=4, calc_attn=True):
        super(LDCA, self).__init__()
        self.channels = channels  # 60
        self.window_size = window_size
        self.calc_attn = calc_attn
        self.inp = nn.Sequential(
            nn.Conv2d(self.channels, self.channels // 2, kernel_size=1),
            nn.ReLU()
        )
        self.split_chns = [self.channels // 4, self.channels // 4]
        self.act2 = nn.Sigmoid()
        self.conv1x1_2 = nn.Conv2d(self.channels // 4, self.channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_ = self.inp(x)
        wsize = self.window_size
        qx = rearrange(x_, 'b  c (h dh) (w dw) ->  b c (dh dw) (h w) ', dh=wsize, dw=wsize)
        _, vx = torch.split(qx, self.split_chns, dim=1)
        qkvmb = mean_block(qx)
        qkvstd = stdv_block(qx, qkvmb)
        qkv = (qkvmb + qkvstd)
        q, v = torch.split(qkv, self.split_chns, 1)
        atn = (q @ q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        y_ = atn @ v
        F = vx * self.act2(y_)
        F = rearrange(F, 'b c (dh dw) (h w) -> b  c (h dh) (w dw)', dh=4, dw=4, h=h // wsize, w=w // wsize)
        F = self.conv1x1_2(F)
        return F + x


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1)).to('cuda:0')

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).to('cuda:0')
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).to('cuda:0')

        a = (torch.bmm(proj_query_H, proj_key_H).to('cuda:0') + self.INF(m_batchsize, height, width).to('cuda:0'))
        energy_H = a.view(m_batchsize, width,
                          height,
                          height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width).to('cuda:0')
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1).to('cuda:0')
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3).to('cuda:0')
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=(3,3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=(3,3), stride=1, padding=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=(3,3), stride=1, padding=1)

    def forward(self, x):
        identity = x
        # [1,64,120,100]

        n, c, h, w = x.size()
        #Extract nchw separately

        x_h = self.pool_h(x)
        # x_h=[1,64,120,1]

        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        #self.pool_w(x)=[1,64,1,100] permute=[1,64,100,1]
        #so x_w=[1,64,100,1]
        #print(x_w.shape)

        y = torch.cat([x_h, x_w], dim=2)
        #print(y.shape) y=[1,64,220,1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        # print(y.shape) y=[1,8,220,1]

        x_h, x_w = torch.split(y, [h, w], dim=2)
        #print(x_h.shape) x_h=[1,8,120,1]
        #print(x_w.shape) x_w=[1,8,100,1]
        x_w = x_w.permute(0, 1, 3, 2)
        #print(x_w.shape)
        #x_w=[1,8,1,100]

        a_h = self.conv_h(x_h).sigmoid()
        #print(a_h.shape)
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        #out = identity +( a_w * a_h)

        return out

if __name__ == '__main__':
    from thop import profile
    import time

    time_start = time.time()
    model = FSA1(64)
    x = torch.randn(1,64,100,100)
    out = model(x)
    print(out.shape)
    flops, params = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)
    time_end = time.time()
    print('totally cost', time_end - time_start)
