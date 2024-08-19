import torch
import torch.nn as nn      # 包含构建神经网络的模块和函数
import torch.optim as optim  # 包含多种优化算法（如SGD、Adam等）
import torch.nn.functional as F   # 提供各种函数（如激活函数、损失函数等），通常用于模型的前向传播
import torchvision                # 提供计算机视觉相关的工具和数据集
import torchvision.transforms as transforms    # 据预处理的功能，例如图像转换和归一化

class Model:   # 创建模型类，Model类用于封装神经网络、损失函数和优化器
    def __init__(self, net, cost, optimist):   # 这个net就是贡菜下面构建的那个
        self.net = net
        self.cost = self.create_cost(cost)    # 根据输入去create损失函数
        self.optimizer = self.create_optimizer(optimist)
        pass

    def create_cost(self, cost):   # create_cost方法根据输入的字符串选择相应的损失函数（如交叉熵损失或均方误差损失）
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),   # 是nn里写好的
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):  # create_optimizer方法根据输入的字符串选择相应的优化器，并使用网络参数进行初始化
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    def train(self, train_loader, epoches=3):    # train: 方法名，表示训练模型。self: 指向当前类的实例。train_loader: PyTorch的DataLoader实例，用于提供数据批次。epoches: 训练的轮数，默认为3。
        for epoch in range(epoches):
            running_loss = 0.0  # running_loss: 一个变量，用于累计当前epoch中每个batch的损失值，以便在输出时计算平均损失
            for i, data in enumerate(train_loader, 0):  # 开始一个循环，用于遍历训练数据集中的每一个batch。
# enumerate(train_loader, 0): 返回每一个batch的数据及其索引，i是当前的batch索引，data是当前batch的数据。
                inputs, labels = data

                self.optimizer.zero_grad()   # 在每次迭代前清空之前计算的梯度，以避免梯度累加。PyTorch中的梯度默认是累加的

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()  # 计算损失对每个参数的梯度（反向传播）。这一步会更新网络中每个参数的梯度信息
                self.optimizer.step()     # 更新网络中每个参数的值。使用之前计算的梯度来更新参数

                running_loss += loss.item()   # 将当前batch的损失值累加到running_loss中。loss.item(): 获取损失值的数值（从Tensor中提取出一个Python数字
                if i % 100 == 0:  # 每100个batch输出一次训练状态
                    print('[epoch %d, %.2f%%] loss: %.3f' %     # 输出当前epoch的编号，当前训练进度的百分比，以及当前100个batch的平均损失
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')   # 在所有epoch训练完成后，输出“Finished Training”以表示训练结束

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # 这句话下面的所有变量都不需要自动求导
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)   # 排序，取最大的概率的值
                total += labels.size(0)   # 计算一共推理了多少张图
                correct += (predicted == labels).sum().item()   # 对了就加

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
    return trainloader, testloader


class MnistNet(torch.nn.Module):   # 这块完成了网络模型的构建，完了还要定义损失函数、优化项
    def __init__(self):  # init里面放需要训练的
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)     # 相当于reshape，把（28，28）变成28*28，-1表示这个维度不变
        x = F.relu(self.fc1(x))     # F是import进来发Function,都是一些不需要训练的函数，比如激活函数
        # 先过一个relu,接的是fc1(x),x是输入，先过一个fc1的lineary,再池化
        x = F.relu(self.fc2(x))   # 再过fc2,再池化
        x = F.softmax(self.fc3(x), dim=1)   # 再过fc3,再softmax
        return x

if __name__ == '__main__':
    # train for mnist
    net = MnistNet()   # 按顺序看上面的定义函数，网络结构
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')   # 优化项
    train_loader, test_loader = mnist_load_data()  # 把训练数据、测试数据load进来分别用于训练和测试
    model.train(train_loader)
    model.evaluate(test_loader)
