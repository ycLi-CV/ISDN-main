import torch
import torch.nn as nn
from model import blue_block as B
from model import mmimdn as B1
import torch.nn.functional as F
import math
from model import upsamplers
from functools import partial
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

def make_model(args, parent=False):
    model = DEDRN()
    return model


class DEDRN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=58, num_block=8, num_out_ch=3,conv=B.BSConvU):
        super(DEDRN, self).__init__()
        self.fea_conv = conv(num_in_ch*4, num_feat, kernel_size=3, padding=1)

        self.B1 = B.B9(num_feat,num_feat)
        self.B2 = B.B9(num_feat,num_feat)
        self.B3 = B.B9(num_feat,num_feat)
        self.B4 = B.B9(num_feat,num_feat)
        self.B5 = B.B9(num_feat,num_feat)
        self.B6 = B.B9(num_feat,num_feat)
        self.B7 = B.B9(num_feat,num_feat)
        self.B8 = B.B9(num_feat,num_feat)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1,padding=1)
        self.GELU = nn.GELU()
        self.c2 = conv(num_feat, num_feat, kernel_size=3, padding=0)


        upsample_block = B1.pixelshuffle_block
        self.upsampler = upsample_block(num_feat, num_out_ch, upscale_factor=4)


    def forward(self, input):
        input = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4,out_B5, out_B6, out_B7, out_B8], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea
        output = self.upsampler(out_lr)


        return output


net=DEDRN()

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col

def hook(module, input, output, layer_num,d_num):
    #print(f"layer {layer_num},d{d_num},{output.shape}")
    output_image = output.squeeze(0).detach().cpu().numpy()  # (29, H, W)
    num_channels = output_image.shape[0]
    row, col = get_row_col(num_channels)
    
    plt.figure(figsize=(10, 10))
    
    # 定义归一化器，将特征图的值范围固定在-1到1之间
    normalize = colors.Normalize(vmin=-1, vmax=1)
    
    for i in range(num_channels):
        feature_map = output_image[i]
        plt.subplot(row, col, i+1)
        plt.imshow(feature_map, cmap='seismic', norm=normalize)
        plt.axis('off')
        plt.title(f'Channel {i+1}')
    
    plt.tight_layout()
    output_file = f"layer_{layer_num}_d{d_num}_output.png"
    plt.savefig(output_file)
    plt.close()
    
net.B1.d1.register_forward_hook( partial(hook, layer_num=1,d_num=1))
net.B1.d2.register_forward_hook( partial(hook, layer_num=1,d_num=2))
net.B1.d3.register_forward_hook( partial(hook, layer_num=1,d_num=3))
# net.B2.d1.register_forward_hook( partial(hook, layer_num=2))

#a=torch.randn(1,3,100,100)
#print("--------=")
#print(a.shape)
#net(torch.randn((1,3,100,100)))

image_path = "/home/mlt01/fmh/datasets/benchmark/Set5/LR_bicubic/X4/butterflyx4.png"
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_image = transform(image).unsqueeze(0)

net(input_image)

'''
image_path='/home/mlt01/fmh/datasets/benchmark/Set5/LR_bicubic/X4/butterflyx4.png'
input_image = Image.open(image_path)
transform = transforms.ToTensor()
input_tensor = transform(input_image)
input_tensor = input_tensor.unsqueeze(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)
net(input)
'''