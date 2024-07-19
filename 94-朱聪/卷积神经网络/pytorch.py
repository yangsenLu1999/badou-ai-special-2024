import torch
import torch.nn as nn

# ======================== 自动求导功能 ====================

x = torch.randn((4, 4), requires_grad=True) # 建一个形状为 (4, 4) 的随机张量，并设置 requires_grad=True，以便后续可以跟踪它的梯度
y = 2 * x
z = y.sum()

print(z.requires_grad)  # True

z.backward() # 进行反向传播,计算 z 对 x 的梯度,并存储在x.grad中

print(x.grad)
'''
tensor([[ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.]])
'''

# ======================== Linear ======================

# 自己实现的简易Linear
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = x.mm(self.weight.t()) # 如不满足乘法条件，可以使用 self.weight.t() 转置以匹配矩阵乘法的要求
        if self.bias is not None:
            x = x + self.bias.expand_as(x) # expand_as(x) 方法会将 self.bias 扩展成与 x 张量相同的形状
        return x


if __name__ == '__main__':
    # train for mnist
    net = Linear(3, 2) # 3是输入张量的维度  2是输出张量的
    x = torch.randn(1, 3)
    output = net.forward(x)
    print(output)


# ======================== 容器 ==========================
# 方法1
model = nn.Sequential()
model.add_module('fc1', nn.Linear(3, 4)) # 输入特征为3
model.add_module('fc2', nn.Linear(4, 2)) # 要对应好，第二层的输入特征数量应该与上一层的输出特征量相同
model.add_module('output', nn.Softmax(2)) # 同样对应好

# 方法2
model2 = nn.Sequential(
    nn.Conv2d(1, 20, 5), # 定义卷积层 (Conv2d)，输入通道数为 1，输出通道数20，卷积核尺寸为5 * 5
    nn.ReLU(),
    nn.Conv2d(20, 64, 5), # 同样要对应好
    nn.ReLU()
)

# 方法3
model3 = nn.ModuleList([nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)])