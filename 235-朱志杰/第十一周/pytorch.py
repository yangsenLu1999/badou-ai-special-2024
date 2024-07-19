import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Model:
    def __init__(self, net, cost, optimist):
        # 保存传入的神经网络
        self.net = net
        # 调用损失函数方法
        self.cost = self.create_cost(cost)
        # 调用优化器函数方法
        self.optimizer = self.create_optimizer(optimist)
        pass
    # 创建损失函数，定义字典来保存多种损失函数
    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    # 创建优化器，定义字典来保存多种优化器，**rests用于接受额外的关键参数
    def create_optimizer(self, optimist, **rests):
        support_optim = {
            # lr是学习率，控制权重步长
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def mnist_load_data():
    # 数据预处理，归一化，把图像像素值都归到0，1之间
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])
    # 从torchvision下载数据集
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    # 批量加载数据集
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
    return trainloader, testloader

# 构建神经网络
class MnistNet(torch.nn.Module):
    # 存放需要训练的参数
    def __init__(self):
        # 初始化函数，mnistNet类是最简单的神经网络模型
        super(MnistNet, self).__init__()
        # 定义线性层，输入层得和下一个输出层数量保持一致
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)
    # 存放不需要训练的参数
    def forward(self, x):
        # 输入数据x，并重塑形状，因mnist数据集图片大小都为28*28。-1 表示自动计算该维度的大小，以确保总元素数量不变。
        x = x.view(-1, 28*28)
        # 数据通过线性层，激活函数使用relu
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 应用 Softmax 激活函数。Softmax 函数用于多分类问题，它可以将输出转换为概率分布。
        x = F.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':
    # train for mnist
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
