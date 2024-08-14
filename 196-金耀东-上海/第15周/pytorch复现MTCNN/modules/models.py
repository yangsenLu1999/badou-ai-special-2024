from torch import nn

class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.pnet_body_conv = nn.Sequential(
            ConvPReluMaxpool(in_c=3, out_c=10, conv_ksize=3, pool_ksize=2),
            ConvPRelu(in_c=10, out_c=16, ksize=3),
            ConvPRelu(in_c=16, out_c=32, ksize=3)
        )
        self.pnet_head_cls = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1)
        self.pnet_head_roi = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)
        # self.pnet_head_landmask = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1)

    def forward(self,x):
        x = self.pnet_body_conv(x)
        cls = self.pnet_head_cls(x)
        roi = self.pnet_head_roi(x)
        # landmask = self.pnet_head_landmask(x)

        return cls, roi, None

class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.rnet_body_conv = nn.Sequential(
            ConvPReluMaxpool(in_c=3, out_c=28, conv_ksize=3, pool_ksize=3),
            ConvPReluMaxpool(in_c=28, out_c=48, conv_ksize=3, pool_ksize=3),
            ConvPRelu(in_c=48, out_c=64, ksize=2)
        )
        self.rnet_body_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 128)
        )
        self.rnet_head_cls = nn.Linear(in_features=128, out_features=2)
        self.rnet_head_roi = nn.Linear(in_features=128, out_features=4)
        # self.rner_head_landmask = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.rnet_body_conv(x)
        x = self.rnet_body_fc(x)
        cls = self.rnet_head_cls(x)
        roi = self.rnet_head_roi(x)
        # landmask = self.rner_head_landmask(x)
        return cls, roi, None

class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.onet_body_conv = nn.Sequential(
            ConvPReluMaxpool(in_c=3, out_c=32, conv_ksize=3, pool_ksize=3),
            ConvPReluMaxpool(in_c=32, out_c=64, conv_ksize=3, pool_ksize=3),
            ConvPReluMaxpool(in_c=64, out_c=64, conv_ksize=3, pool_ksize=2),
            ConvPRelu(in_c=64, out_c=128, ksize=2)
        )
        self.onet_body_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152, 256)
        )
        self.onet_head_cls = nn.Linear(in_features=256, out_features=2)
        self.onet_head_roi = nn.Linear(in_features=256, out_features=4)
        self.onet_head_landmask = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = self.onet_body_conv(x)
        x = self.onet_body_fc(x)
        cls = self.onet_head_cls(x)
        roi = self.onet_head_roi(x)
        landmask = self.onet_head_landmask(x)
        return cls, roi, landmask

class ConvPRelu(nn.Module):
    def __init__(self, in_c, out_c, ksize):
        super(ConvPRelu, self).__init__()
        self.conv_prelu = nn.Sequential(
            nn.Conv2d(in_c, out_c, ksize, stride=1, padding=0),
            nn.PReLU(num_parameters=1) # 与leakyRelue,a可学习，num_p=1:所有通道a相同; num_p=in_channles:所有通道的a不同
        )
    def forward(self,x):
        return self.conv_prelu(x)

class ConvPReluMaxpool(nn.Module):
    def __init__(self, in_c, out_c, conv_ksize, pool_ksize ):
        super(ConvPReluMaxpool, self).__init__()
        self.conv_prelu_maxpool = nn.Sequential(
            ConvPRelu(in_c, out_c, conv_ksize),
            nn.MaxPool2d(pool_ksize, stride=2, ceil_mode=True) # ceil_mode: size向上取值, floor_mode: size向下取整
        )
    def forward(self, x):
        return self.conv_prelu_maxpool(x)

