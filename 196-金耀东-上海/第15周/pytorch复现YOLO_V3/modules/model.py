import torch
from torch import nn

# ---------------------------------------#
#           YoloV3
# ---------------------------------------#
class YoloV3(nn.Module):
    def __init__(self, in_channels, num_achors_per_grid, num_cls):
        super(YoloV3, self).__init__()
        self.darknet53 = Darcknet53(in_channels=in_channels)
        self.yolo_body = YoloBody(out_channels=num_achors_per_grid*(num_cls+5))

    def forward(self, x):
        darknet53_out = self.darknet53(x)
        yolo_out = self.yolo_body(darknet53_out)
        return yolo_out

# ---------------------------------------#
#   YoloV3 Part I：Darcknet53
# ---------------------------------------#
class Darcknet53(nn.Module):
    def __init__(self, in_channels):
        super(Darcknet53, self).__init__()
        self.branch3_layer26 = nn.Sequential(
            CBL(in_channels, 32, (3, 3)), # layer 1
            ResidualBlock(32, 64, num_residual=1), # layer 4:1+3
            ResidualBlock(64, 128, num_residual=2), # layer 9:4+5
            ResidualBlock(128, 256, num_residual=8) # layer 26:9+17
        )
        self.branch2_layer43 = ResidualBlock(256, 512, num_residual=8) # layer 43:26+17
        self.branch1_layer52 = ResidualBlock(512, 1024, num_residual=4) # layer 52:43+9

    def forward(self, x):
        out3 = self.branch3_layer26(x)
        out2 = self.branch2_layer43(out3)
        out1 = self.branch1_layer52(out2)
        return out1, out2, out3

# ---------------------------------------#
#   YoloV3 Part II：YoloBody
# ---------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, out_channels, in1_channels=1024, in2_channels=512, in3_channels=256):
        super(YoloBody, self).__init__()
        self.branch1_cblx5_cblcovn = nn.Sequential(
            CBLx5(in1_channels, 512),
            CBLConv(512, 1024, out_channels)
        )
        self.branch2_catblock_cblx5 = nn.Sequential(
            ContactBlock(512, in2_channels, out_channels=256),
            CBLConv(256, 512, out_channels)
        )
        self.branch3_catblock_cblx5 = nn.Sequential(
            ContactBlock(256, in3_channels, out_channels=128),
            CBLConv(128, 256, out_channels)
        )

    def forward(self, x):
        in1, in2, in3 = x
        out1_route = self.branch1_cblx5_cblcovn[0](in1)
        out1 = self.branch1_cblx5_cblcovn[1](out1_route) # 512, 13, 13

        out2_route = self.branch2_catblock_cblx5[0](out1_route, in2) # 256, 26, 26
        out2 = self.branch2_catblock_cblx5[1](out2_route)

        out3_route = self.branch3_catblock_cblx5[0](out2_route, in3)
        out3 = self.branch3_catblock_cblx5[1](out3_route) # 128, 52, 52
        return out1, out2, out3

# ---------------------------------------#
#      卷积模块:Conv + BN + LeRelu
# ---------------------------------------#
class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="same"):
        super(CBL, self).__init__()
        self.cbl = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=1e-1)
        )
    def forward(self, x):
        return self.cbl(x)

# ---------------------------------------#
#          残差模块
# ---------------------------------------#
class Residual(nn.Module):
    def __init__(self, num_features):
        super(Residual, self).__init__()
        tmp = num_features // 2
        self.residual_branch = nn.Sequential(
            CBL(in_channels=num_features, out_channels=tmp, kernel_size=(1,1)),
            CBL(in_channels=tmp, out_channels=num_features,kernel_size=(3,3))
        )
    def forward(self,x):
        return x + self.residual_branch(x)

# ---------------------------------------#
#   多层残差模块：CBL + N * Residual
# ---------------------------------------#
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual):
        super(ResidualBlock, self).__init__()
        residual_moduls = [Residual(num_features=out_channels) for _ in range(num_residual) ]
        self.residual_block = nn.Sequential(
            CBL(in_channels, out_channels, (3, 3), stride=2, padding=1),
            *residual_moduls
        )
    def forward(self,x):
        return self.residual_block(x)

# ---------------------------------------#
#           5层卷积模块
# ---------------------------------------#
class CBLx5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBLx5, self).__init__()
        tmp_channels = out_channels*2
        self.cblx5 = nn.Sequential(
            CBL(in_channels, out_channels, (1,1)),
            CBL(out_channels, tmp_channels, (3,3)),
            CBL(tmp_channels, out_channels, (1, 1)),
            CBL(out_channels, tmp_channels, (3, 3)),
            CBL(tmp_channels, out_channels, (1, 1))
        )
    def forward(self,x):
        return self.cblx5(x)

# ---------------------------------------#
#         合并模块
# ---------------------------------------#
class ContactBlock(nn.Module):
    def __init__(self, in0_channels, in1_channels, out_channels):
        super(ContactBlock, self).__init__()
        self.catblock_branch_upsample = nn.Sequential(
            CBL(in0_channels, out_channels, (1,1)),
            nn.Upsample(scale_factor=2, mode="nearest")
        )
        self.catblock_out_cblx5 = CBLx5(in_channels=in1_channels+out_channels, out_channels=out_channels)
    def forward(self, in0, in1):
        out = self.catblock_branch_upsample(in0)
        out = torch.cat([out, in1], dim=1)
        out = self.catblock_out_cblx5(out)
        return out

# ---------------------------------------#
#         CBL + Conv
# ---------------------------------------#
class CBLConv(nn.Module):
    def __init__(self, in_channels, hidde_channels, out_channels):
        super(CBLConv, self).__init__()
        self.cbl_conv = nn.Sequential(
            CBL(in_channels, hidde_channels, (3, 3)),
            nn.Conv2d(hidde_channels, out_channels, (1, 1))
        )
    def forward(self, x):
       return self.cbl_conv(x)



