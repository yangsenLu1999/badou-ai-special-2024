import torch

x = torch.randn([2, 3], requires_grad=True)  # 默认requires_grad = False缺省，不计算梯度
# randn随机生成复核(0, 1)标准正态分布, 是random_normal(loc, scale, size)的特殊化
y = 2 * x
z = y.sum()
print(z.requires_grad)  # 由x需要计算梯度，传递至后端函数
z.backward()
print(x.grad)

s = x.sum()
print(s.requires_grad)
s.backward()
print(x.grad)  # 会将z和s对x的导数相加求和
