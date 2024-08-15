# 定义unet各个模块，方便复用

import torch
import torch.nn.functional as F
import torch.nn as nn

# 卷积块
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Sequential: 将多个层组合成一个顺序的容器
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    # 前项传播
    def forward(self,x):
        return self.double_conv(x)

# 下采样
# 包含一个2x2最大池化和double_conv操作
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self,x):
        return self.maxpool_conv(x)

# 上采样
# 根据bilinear参数决定采用双线性插值还是转置卷积
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        # 继承/使用/改写nn.Module的方法
        super().__init__()
        if bilinear:
            # Upsample:双线性插值
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 转置卷积
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    # 不仅要做上采样，还有copy and crop之后的Double Conv
    def forward(self,x1,x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]]) # 保证尺寸相同
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        # 按尺寸填充
        x1 = F.pad(x1,[diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        # 拼接
        x = torch.cat([x2, x1], dim=1)
        # Double Conv
        return self.conv(x)

# unet最底层的conv结构，改变通道数
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self,x):
        return self.conv(x)

