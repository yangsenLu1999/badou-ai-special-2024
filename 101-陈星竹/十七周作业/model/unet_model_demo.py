
import torch.nn.functional as F
from .unet_parts_demo import *

class UNetDemo(nn.Module):
    def __init__(self, n_channels, n_classes,bilinear=True):
        super(UNetDemo, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear # 上采样

        # 定义unet的各个层
        self.inc = DoubleConv(n_channels, 64) #输入
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024,256,bilinear)
        self.up2 = Up(512,128,bilinear)
        self.up3 = Up(256,64,bilinear)
        self.up4 = Up(128,64,bilinear)
        # 卷积输出
        self.outc = OutConv(64,n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # 上采样部分一一拼接与裁剪
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    net = UNetDemo(n_channels=3,n_classes=1)
    print(net)


