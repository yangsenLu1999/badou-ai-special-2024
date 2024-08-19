import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        layers = nn.Sequential()
        layers.add_module('double_conv1',nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        layers.add_module('bn1',nn.BatchNorm2d(out_channels))
        layers.add_module('relu1',nn.ReLU(inplace=True))
        layers.add_module('double_conv2',nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
        layers.add_module('bn2',nn.BatchNorm2d(out_channels))
        layers.add_module('relu2',nn.ReLU(inplace=True))
        self.double_conv = layers
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels,bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels//2 , out_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1,[diffX//2,diffX - diffX//2,diffY//2,diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


























