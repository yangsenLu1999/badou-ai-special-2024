import torch


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias):
        super(Linear, self).__init__()  # 一般子类不继承父类的属性init，此方法可以继承父类的属性
        # Linear类继承nn.Module，super(Linear, self).__init__()
        # 就是对继承自父类nn.Module的属性进行初始化。而且是用nn.Module的初始化方法来初始化继承的属性
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]))  # 生成标准正态分布随机数
        # Parameter默认为有梯度，意义在于使得tensor变得可训练
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features, 1))

    def forward(self, x):
        x = torch.mm(self.weight, x.T)  # mm即为matmul需要引入 self.weight和x都是float
        # x = x.mm(self.weight)  # 简写方式，x * self.weight
        print(x.shape, self.bias.shape)
        print(x, self.bias)
        if self.bias:
            # x = x + self.bias  # 因为self.bias的tensor形式只有outfeatures, torch.Size([1, 2]) torch.Size([1])
            x = x + self.bias.expand_as(x)  # 将self.bias维度转为与x一致，方可相加
        return x


if __name__ == '__main__':
    linear = Linear(3, 1, True)
    data = torch.tensor([[5., 2., 3.],
                         [3., 2., 4.]])
    result = linear.forward(data)
    print(result)
