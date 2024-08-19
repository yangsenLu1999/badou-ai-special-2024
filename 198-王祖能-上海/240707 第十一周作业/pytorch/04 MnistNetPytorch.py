'''
pytorch框架手写mnist数字识别
'''
import torch
import torch.nn as nn
import torch.optim as optim  # 定义优化器的模块，optim.SGD()
import torchvision
import torchvision.transforms as transforms  # torchvison主要实现对数据集的预处理、数据增强、数据转换成tensor等
from torch.utils.data import DataLoader
import time


class MnistNet(torch.nn.Module):  # 第二步，组建网络层次关系,必须继承父类torch.nn.Module
    def __init__(self):
        super(MnistNet, self).__init__()  # 继承了父类的init
        self.fc1 = nn.Linear(28*28, 512)  # 形成的权重矩阵是(512 ， 28*28) in_features在后方便 w * x
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)  # 一般将可训练的权重参数放置到init层

    def forward(self, x):  # 应该也可以按照容器概念修改网络结构
        x = x.view(-1, 28*28)  # view相当于reshape，改变形状，-1 表示其他维度自动计算
        x = self.fc1(x)  # 全连接层与激活层的串联处理数据，最后softmax分类
        x = nn.functional.relu(x)  # 是否一定需要引入torch.nn.functional as F, x = F.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.softmax(x)  # torch.nn.Softmax()作为一个层添加到模型中， torch.nn.functional.softmax()作为一个函数直接调用
        # x = nn.softmax(x, dim=1) 当输入数据是二维时，dim=0表示对列进行归一化，dim=1表示对行进行归一化
        # print(x.shape)
        return x


'''
网络基类：nn.module用于所有神经网络层的基类，定义了神经网络中前向传播和反向传播需要的方法，并要用super_init方法继承
常用网络层结构：nn.Linear全连接层； nn.Conv2d二维卷积层； nn.MaxPool2d二维最大池化层； nn.Dropout随机失活层，减少全连接过程可能过拟合
常用激活函数：nn.ReLU(), nn.Sigmoid(), nn.Tanh()
常用损失函数：nn.CrossEntropyLoss()：交叉熵损失函数，通常用于分类问题;nn.MSELoss()：均方差损失函数，通常用于回归问题
常用优化器：torch.optim.SDG()：随机梯度下降优化器;torch.optim.RMSprop; torch.optim.Adam
'''


class Model:  # 第三步，在网络结构返回结果的基础上搭建模型结构，优化器损失函数的设置，以及训练和推理过程
    def __init__(self, net, cost, optimist):  # 模型的损失函数和优化器初始化
        self.net = net
        self.cost = self.create_cost(cost)  # init直接调用了内部函数
        self.optimist = self.create_optimist(optimist)

    def create_cost(self, cost):
        support_cost = {
            'CrossEntropy': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]

    def create_optimist(self, optimist, **rests): # 可能有其他参数**rests
        support_optimist = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),  # parameters是调回MnistNet中的init下面fc构造函数里定义各个层
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)  # 这个学习率重要改为0.01计算，很有可能都是一个结果，不好
        }
        return support_optimist[optimist]

    def train(self, train_data, epochs):
        for e in range(epochs):
            lossing = 0
            print('epochs:%d' % (e + 1))
            for i, data in enumerate(train_data, 0):  # 按batch数读进来，0表示第一个index从0开始标注，若为1，表示第一个value对应的index为1
                inputs, labels = data  # data是60000/32个 ( [32, 1, 28, 28]的inputs tensor 以及 [32, ]的labels tensor) 组成的list

                self.optimist.zero_grad()  # 一般用于loss.backward前，清除累计梯度，每个epoch更新一次累计梯度

                outputs = self.net(inputs)  # pytorch框架封装，就是调用forward，其他函数不行
                loss = self.cost(outputs, labels)
                # print(outputs.shape, labels.shape)
                loss.backward()
                '''
                先将梯度归零（optimizer.zero_grad()），然后反向传播计算得到每个参数的梯度值（loss.backward()），最后通过梯度下降执行一步参数更新（optimizer.step()）
                '''
                self.optimist.step()  # 用于优化器更新w, 以SGD为例，w = w - lr * w_grad

                lossing += loss.item()  # 取出张量的数字，高精度显示，且减少内存占用，用于loss， accuracy
                if (i+1) % 100 == 0:
                    print('loss', lossing)
                    print('progress:%.2f %%, loss:%.3f' % ((i+1)/len(train_data)*100, lossing/100))  # i表示batch, 每100个batch显示平均损失
                    lossing = 0
        print('--------Training finished--------')

    def evaluate(self, test_data):
        print('--------Evaluating--------')
        correct, total = 0, 0
        with torch.no_grad():  # 表示以下内容不需要求梯度，因为是推理，权重已根据训练得到
            print(test_data)
            for data in test_data:
                inputs, labels = data

                outputs = self.net(inputs)
                predicts = torch.argmax(outputs, 1)  # 排序取出top1的outputs
                total += labels.size(0)
                correct += (predicts == labels).sum().item()  # sum()结果是tensor(49),item转换为int(49)
        print('准确率为：', correct / total * 100)


def load_data():  # 第一步，读取下载数据文件并进行预处理。共60000+10000的训练+测试的样本集
    '''
    torchvision.transforms是pytorch中的图像预处理包,主要实现对数据集的预处理、数据增强、数据转换成tensor等。
    一般用Compose把多个步骤整合到一起.
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
         # 把PIL图像或[0, 255]范围的numpy.ndarray形状(H W C)转化成torch.FloatTensor，张量形状(C x H x W)，范围在[0.0, 1.0]
        transforms.Normalize([0, ], [1, ])])
        # 平均值和标准差标准化输入图片，给定n个通道的平均值(M1,…,Mn)和标准差(S1,…,Sn)，这一变换会在输入图片的每一个通道上进行标准化，
        # 即input[channel] = (input[channel] - mean[channel]) / std[channel]。
        # transforms.Resize() # 输入图片大小调成为指定大小
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    '''
    torchvision中datasets中所有封装的数据集都是torch.utils.data.Dataset的子类，它们都实现了__getitem__和__len__方法。
    因此，它们都可以用torch.utils.data.DataLoader进行数据加载。
    train (bool, optional)： 如果为True，则从training.pt创建数据集，否则从test.pt创建数据集
    download (bool, optional)： 如果为True，则从internet下载数据集并将其放入根目录。如果数据集已下载，则不会再次下载
    transform (callable, optional)： 接收PIL图片并返回转换后版本图片的转换函数
    '''
    train_data = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    '''
    主要是对数据进行 batch 的划分， shuffle是否打乱抓取数据
    num_workers是Dataloader的概念，默认值是0,告诉DataLoader实例要使用多少个子进程进行数据加载(和CPU有关，和GPU无关)
    num_worker多线程，设置得大，寻batch速度快，因为下轮迭代batch可能在上轮/上上轮…迭代时已经加载好。
    坏处是内存开销大，加重CPU负担（worker加载数据到RAM的进程是CPU复制）。经验值是自己电脑/服务器的CPU核心数，如CPU、RAM充足，可设置更大。
    '''
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_data = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    return train_data, test_data


if __name__ == '__main__':
    start = time.time()
    net = MnistNet()  # 网络实例化
    model = Model(net, 'CrossEntropy', 'RMSP')  # 神经网络损失函数、优化器属性定义
    train_data, test_data = load_data()  # 下载训练数据
    model.train(train_data, epochs=5)  # 数据传入模型中训练
    middle = time.time()
    model.evaluate(test_data)
    end = time.time()
    train_time = middle - start
    evaluate_time = end - middle
    print(train_time, evaluate_time)
