"""U - Net模型的组成部分"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # 原始张量修改，节省空间，但无法追溯到relu前的值
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # kernel size 2  stride：默认：等于kernel size
            DoubleConv(in_channels,out_channels)
        )

    def forward(self,x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,bilinear=True):
        super().__init__()

        # 如果是双线性，则使用正常卷积来减少通道数
        if bilinear:
            #  align_corners = True:   像素被视为网格格子上的点，拐角处像素对齐。点与点等距。
            #  align_corners = False:  像素被视为网格的交叉线上的点，拐角处的点依然是原图像的拐角像素，但插值的点差距不等
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True) # 大两倍
        else:
            self.up = nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)

        self.conv = DoubleConv(in_channels,out_channels)

    def forward(self,x1,x2):
        x1 = self.up(x1)

        # input is CHW
        # x1,x2长宽的差，为了将两者尺寸变一致
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])   #输入数据构造张量
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])


        x1 = F.pad(x1,[diffX//2,diffX - diffX//2,
                       diffY//2,diffY - diffY//2,]) # 从最后一维开始，前后各加一项。 此处加W 加H

        x = torch.cat([x2,x1],dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OutConv,self).__init__()  # 继承父类的属性
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)

    def forward(self,x):
            return self.conv(x)
