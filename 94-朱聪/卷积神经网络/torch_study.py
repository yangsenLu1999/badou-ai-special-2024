import torch
import torch.nn as nn
import numpy as np


# pytorch数据结构，默认的整数是int64(LongTensor)，默认的浮点数是float32(FloatTensor)
a = torch.tensor([1,2,3])
b = torch.tensor([1.,2.,3.])
# 可以通过dtype修改类型，一般情况下，直接使用tensor()就行，因为整数和浮点数刚好是我们在深度学习中标签、输入数据的类型
c = torch.tensor([1,2,3], dtype=torch.int8)
# 除了dtype，还可以使用torch.IntTensor([1,2,3])
# LongTensor常用在深度学习中的标签值， FloatTensor常用做深度学习中可学习参数或者输入数据的类型
d = torch.DoubleTensor([1,2,3])

# 使用torch.float()等方法实现数据类型转换  a.float()  a.double() a.long()
e = a.float()
# 也可以使用type转换
f = a.type(torch.float32)

# 张量的数据类型和numpy.array基本一一对应，除了不支持str
# torch基本上是实现了numpy的大部分必要的功能，并且tensor是可以利用GPU进行加速训练的
# 两者转换也非常简单
np_data = np.array([1.,2.,3.])
torch_data = torch.tensor(np_data)
np_data = torch_data.numpy() # 注意使用上的区别  一个是 tensor(x)  一个是 x.numpy()

# 共享内存和内存复制
# tensor()，也是最常用的，不管输入类型是什么，torch.tensor都会进行数据拷贝，不共享内存
# .numpy() 是共享内存的， 不希望共享可以使用  .numpy().copy()

# 什么叫张量
'''
标量：数据是一个数字
向量：数据是一串数字,也是一维张量
矩阵：数据二维数组，也是二维张量
张量：数据的维度超过2的时候，就叫多维张量
'''
# tensor, 张量修改尺寸 reshape  view  都是共享内存的
a = torch.arange(0, 6)
b = a.reshape((2, 3))
print(b)
c = a.view((2, 3))
# numpy, 张量修改尺寸 reshape  resize, resize是直接修改变量  reshape是会返回新的变量

# 矩阵相乘的3个方法  torch.mm()   torch.matmul()   @
# 假如参与运算的是一个多维张量，那么只有torch.matmul()可以使用。可以使用该方法进行兼顾

a = torch.tensor([1.,2.])
b = torch.tensor([2.,3.]).view(2, 1) # (2, 1) 2行一列
print(torch.matmul(a, b))
# 多维张量中，参与矩阵运算的其实只有后两个维度，前面的维度其实就像是索引一样
a = torch.rand((3, 2, 64, 32))
b = torch.rand((1, 2, 32, 64))
print(torch.matmul(a, b).shape) # torch.Size([1, 2, 64, 64]) 这里还涉及自动传播，会把b的第一维度复制3次
# clamp，[min,max],小于min的话就被设置为min，大于max的话就被设置为max
a = torch.rand(5)
print(a) # tensor([0.5271, 0.6924, 0.9919, 0.0095, 0.0340])
print(a.clamp(0.3,0.7)) # tensor([0.5271, 0.6924, 0.7000, 0.3000, 0.3000])


# 构建输入集，看怎么样方便
x = np.mat('0 0;'
           '0 1;'
           '1 0;'
           '1 1')

# x = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

x = torch.tensor(x).float() # 必须通过该方法将数据转为张量
y = np.mat('1;'
           '0;'
           '0;'
           '1')
y = torch.tensor(y).float()

# 搭建网络,Sequential允许你按顺序定义神经网络的层,使用简单。
myNet = nn.Sequential( 
    nn.Linear(2,10), # 会默认进行权重的初始化
    nn.ReLU(),
    nn.Linear(10,1),
    nn.Sigmoid()
    )
print(myNet)


# 设置优化器

# parameters()返回权重矩阵参数和偏置参数，对网络进行训练时需要将parameters()作为优化器optimizer的参数
optimzer = torch.optim.SGD(myNet.parameters(), lr=0.05) # SGD(随机梯度下降)
loss_func = nn.MSELoss() # 损失函数

# 训练网络
for epoch in range(5000):
    out = myNet(x)
    loss = loss_func(out, y)
    optimzer.zero_grad() # 清除上一次所求出的梯度
    loss.backward() # 反向传播
    optimzer.step() # 更新参数


# 推理
print(myNet(x).data) # .data取的是数据，myNet(x)除了数据外还有自动求导数据


'''
PyTorch 读取其他的数据，主要是通过 Dataset 类
所有的 datasets 都需要继承它并实现__len__和__getitem__方法
'''
from torch.utils.data import Dataset, DataLoader

# 定义自己的一个DataSet类
class MyDataSet(Dataset):
    def __init__(self):
        # 初始化中，一般是把数据直接保存在这个类的属性中
        self.data = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
        # 一般标签值应该是Long整数的，所以标签的tensor可以用torch.LongTensor(数据)或者用.long()来转化成Long整数的形式
        self.label = torch.LongTensor([1, 1, 0, 0])

    # index是一个索引
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.data)


# 在梯度下降的过程中，一般是需要将多个数据组成batch，这个需要我们自己来组合吗？不需要的，所以PyTorch中存在DataLoader这个迭代器
mydataset = MyDataSet()

# batch_size控制了每次读取的条目数
mydataloader = DataLoader(dataset=mydataset, batch_size=1)

# 每个batch2个样本，且是乱序的
# mydataloader = DataLoader(dataset=mydataset, batch_size=2, shuffle=True)

for i,(data,label) in enumerate(mydataloader):
    # 总共输出了4个batch，每个batch都是只有1个样本（数据+标签），这个输出过程是顺序的。
    print(data,label)


# 如果要使用PyTorch的GPU训练的话，一般是先判断cuda是否可用，然后把数据标签都用to()放到GPU显存上进行GPU加速
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for i,(data,label) in enumerate(mydataloader):
    data = data.to(device)
    label = label.to(device)
    print(data,label)
'''


# 前面使用了Sequential实现一个简单的神经网络，还有另一种方式
'''
模型三要素

必须要继承nn.Module这个类，要让PyTorch知道这个类是一个Module
在__init__(self)中设置好需要的组件，比如conv，pooling，Linear，BatchNorm等等
最后在forward(self,x)中用定义好的组件进行组装，就像搭积木，把网络结构搭建出来，这样一个模型就定义好了
'''

# init中
# Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1)) -> 输入3通道，输出6通道，卷积核,步长1，padding=0
self.conv1 = nn.Conv2d(3, 6, 5) # nn.Conv2d就是一般图片处理的卷积模块 参数分别是输入通道数，输出通道数，卷积核尺寸
self.pool1 = nn.MaxPool2d(2, 2) # 池化层

# forward中  def forward(self, x)
x = self.pool1(F.relu(self.conv1(x))) # x为模型的输入，表示x经过conv1，然后经过激活函数relu，然后经过pool1操作

# 全连接层的输入之前，需要将图片或者其他形状的数据展平成一维向量
x = x.view(-1, 28 * 28) # 对x进行reshape，为后面的全连接层做准备。转化为一个向量

# 使用
net = Net()
outputs = net(inputs) # 类似于使用了net.forward(inputs)这个函数

# modules()可以递归的返回网络的各个module（深度遍历）,从最顶层直到最后的叶子的module
# named_modules()和module()类似，只是同时返回name和module


'''
torchvision 由流行的数据集、模型体系结构和通用的计算机视觉图像转换组成
简单地说就是常用数据集+常见模型+常见图像增强方法
主要组成： torchvision.datasets   orchvision.models   torchvision.transforms
'''

# torchvision.datasets下有很多的数据集，如MNIST ImageNet COCO USPS ...
# 每一个数据集的API都是基本相同的。他们都有两个相同的参数：transform和target_transform
# 用最经典最简单的MNIST手写数字数据集为例，包含了5个参数
'''
root：就是你想要保存MNIST数据集的位置，如果download是False的话，则会从目标位置读取数据集；
download：True的话就会自动从网上下载这个数据集，到root的位置；
train：True的话，数据集下载的是训练数据集；False的话则下载测试数据集（真方便，都不用自己划分了）
transform：这个是对图像进行处理的transform，比方说旋转平移缩放，输入的是PIL格式的图像（不是tensor矩阵）；
target_transform：这个是对图像标签进行处理的函数
'''

import torchvision
mydataset = torchvision.datasets.MNIST(root='./',
                                      train=True,
                                      transform=None,
                                      target_transform=None,
                                      download=True)

myloader = DataLoader(dataset=mydataset, batch_size=16)
# 会抛出错误：这个dataloder只能把tensor或者numpy这样的组合成batch，而现在的数据集的格式是PIL格式
# 所以定义的dataset中，transform不能是None，我们需要将PIL转化成tensor才可以
'''
transform = transforms.Compose( # 将多个数据转换操作组合成一个序列
        [transforms.ToTensor(), # ToTensor() 转换操作，用于将 PIL 图像或 numpy 数组转换为 PyTorch 的张量
         transforms.Normalize([0, ], [1, ])] # 对张量进行标准化,[0, ] 表示每个通道的均值为0，[1, ] 表示每个通道的标准差为1
    )

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
'''

for i,(data,label) in enumerate(myloader):
    print(data.shape)
    print(label.shape)
    break


'''
torchvision.models 提供了多种预训练模型，大体分为：分类模型，语义模型，目标检测模型，视频分类模型

alexnet = models.alexnet()  构建模型，权重值是随机的，只有结构是保存的
alexnet = models.alexnet(pretrained=True) 取预训练的模型，则需要设置参数pretrained

'''


# torch.nn.Module 构建模型，之前讲过，要继承基类并实现相应的方法

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3) # 定义
        self.conv2 = nn.Conv2d(64, 64, 3)

        # 也可以使用add_module完成
        self.add_module('conv1', nn.Conv2d(3, 64, 3)) # 可以用字符串来定义变量名字，更灵活
        self.add_module('conv2', nn.Conv2d(64, 64, 3))


    def forward(self, x):
        x = self.conv1(x) # 正向传播中调用
        x = self.conv2(x)

        return x
    
# 上述方式如果网络复杂的话，就会有很多重复代码。可以使用 torch.nn.ModuleList和torch.nn.Sequential

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(10, 10) for _ in range(5)]
        )


    def forward(self, x):
        for l in self.linears:
            x = l(x)

        return x

# ModuleList主要是用在读取config文件来构建网络模型中的
vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
           512, 512, 512, 'M']

# 如cfg配置，我们就可以通过循环，针对不同内容生成不同的layer，然后添加到一个列表中，最后放入ModuleList，来构建网络
# 也可以通过修改cfg，快速修改网络结构


# Sequential

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3), # 这样是使用默认的数字标号(从0开始)
            nn.Conv2d(64, 64, 3)
        )

        # 如何修改Sequential中网络层的名称,collections.OrderedDict
        '''
            self.conv = nn.Sequential(OrderedDict([
                ('conv1',nn.Conv2d(3,64,3)),
                ('conv2',nn.Conv2d(64,64,3))
            ]))
        '''

        '''
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
            self.conv2 = nn.Conv2d(64, 64, 3)
            self.maxpool1 = nn.MaxPool2d(2, 2)
    
            self.features = nn.Sequential(OrderedDict([
                ('conv3', nn.Conv2d(64, 128, 3)),
                ('conv4', nn.Conv2d(128, 128, 3)),
                ('relu1', nn.ReLU())
            ]))
        
        '''


    def forward(self,x):
        x = self.conv(x)

        return x


'''
总结
- 单独增加一个网络层或者子模块，可以用add_module或者直接赋予属性；
- ModuleList可以将一个Module的List增加到网络中，自由度较高。
- Sequential按照顺序产生一个Module模块。这里推荐习惯使用OrderedDict的方法进行构建。对网络层加上规范的名称，这样有助于后续查找与遍历
'''


# 保存与载入  torch.save和torch.load
