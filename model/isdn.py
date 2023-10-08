import torch
import torch.nn as nn
from model import blue_block as B
from model import mmimdn as B1
import torch.nn.functional as F
import math
from model import upsamplers

def make_model(args, parent=False):
    model = ISDN()
    return model


class ISDN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=56, num_block=8, num_out_ch=3,conv=B.BSConvU):
        super(ISDN, self).__init__()
        #kwargs = {'padding': 1}
        #self.conv = conv
        #self.fea_conv = self.conv(num_in_ch * 4, num_feat, kernel_size=3, **kwargs)
        #self.fea_conv = nn.Conv2d(num_in_ch, num_feat, kernel_size=3,padding=1)
        self.fea_conv = conv(num_in_ch*4, num_feat, kernel_size=3, padding=1)

        self.B1 = B.DSDB(num_feat,num_feat)
        self.B2 = B.DSDB(num_feat,num_feat)
        self.B3 = B.DSDB(num_feat,num_feat)
        self.B4 = B.DSDB(num_feat,num_feat)
        self.B5 = B.DSDB(num_feat,num_feat)
        self.B6 = B.DSDB(num_feat,num_feat)
        self.B7 = B.DSDB(num_feat,num_feat)
        self.B8 = B.DSDB(num_feat,num_feat)
        #self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=2)
        #self.B8 = B.VABB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1,padding=1)
        self.GELU = nn.GELU()

        #self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)
        #self.c2 = nn.Conv2d(num_feat, num_feat, kernel_size=3,padding=0)
        self.c2 = conv(num_feat, num_feat, kernel_size=3, padding=0)
        #self.lkat=B.LKAT(num_feat)

        upsample_block = B1.pixelshuffle_block
        self.upsampler = upsample_block(num_feat, num_out_ch, upscale_factor=4)
        #self.upsampler=upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)


    def forward(self, input):
        input = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv(input)
        #print(out_fea.shape)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1)
        out_B = self.c1(trunk)
        #print(out_B.shape)
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea
        #out_lr = self.lkat(out_B) + out_fea
        output = self.upsampler(out_lr)


        return output


if __name__ == "__main__":
    from thop import profile
    import time

    time_start = time.time()
    x = torch.randn(1, 3, 100, 100)
    model = ISDN()
    print(model(x).shape)
    flops, params = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)

    time_end = time.time()
    print('totally cost', time_end - time_start)