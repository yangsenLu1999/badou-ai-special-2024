# 导入 torch包
import torch

'''
tensor对象通过一系列的运算可以组成动态图，对于每个tensor对象，有下面几个变量控制求导的属性:
    requirs_grad: 默认值为false, 表示变量是否需要计算导数
    grad_fn: 变量的梯度函数
    grad: 变量的梯度
'''
# 创建tensor x , requires_grad 自动求带，默认值false
x = torch.randn((4, 4), requires_grad=True)
y = 2 * x
z = y.sum()

# 这里只有一个变量x， 而且后面进行的常量操作， 所以requires_grad为true
print(z.requires_grad)  # True

# 反向传播(梯度求导，权值更新)
z.backward()

print(x.grad)
'''
tensor([[ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.]])
'''
