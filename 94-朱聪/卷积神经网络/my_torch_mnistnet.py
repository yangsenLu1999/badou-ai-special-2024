import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt

# 构建普通神经网络
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义每层内容
        self.linear = nn.ModuleList([
            nn.Flatten(), # 将输入展平成一维向量
            nn.Linear(28 * 28, 512), # 这里会默认进行权重的初始化
            nn.ReLU(), # 是一个类，在__init__中初始化  而 F.relu是一个函数，要在单独在forward中调用
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10), # 最后一个全连接层，将输入维度 512 转换为输出维度 10，对应于最终的数量(分类任务下)
            nn.Softmax(dim=1) # dim=1 表示在第一个维度（即每个样本的维度）上进行 Softmax 操作
            # 最后输出的数据是一个具有 10 个元素的向量，这个向量表示了模型对每个类别的预测概率
        ])

    def forward(self, x):
        # 依次执行
        # x = x.view(-1, 28 * 28) # 也可以转为向量，因为要接入全连接层
        for l in self.linear:
            x = l(x)

        return x

    # def reset_parameters(self) ,如需要特定的权重初始化方式，重写该方法


# 准备数据
# 下载或者读取
def mnist_load_data():
    # 定义数据转换： PIL -> Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0, ], [1, ])
    ])

    # torchvision.datasets下有很多的数据集
    train_set = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=False
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                              shuffle=True, num_workers=2)

    test_set = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transform,
        download=False
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32,
                                               shuffle=True, num_workers=2)

    return train_loader, test_loader


# 训练
def train(net, train_loader, epoches=3):
    optimizer = optim.RMSprop(net.parameters(), lr=0.001) # 优化器
    for epoch in range(epoches):
        running_loss = 0.0
        # log = []
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()  # 梯度清零，在每个小批量数据训练之前执行
            outputs = net(inputs)
            criterion = nn.CrossEntropyLoss() # 损失函数，计算误差
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # 更新参数

            running_loss += loss.item()
            if i % 100 == 0:
                print('[epoch %d, %.2f%%] loss: %.3f' %
                      (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                # log.append(running_loss / 100)

                running_loss = 0.0

        # plt.plot(log)
        # plt.show()

# 推理
def evaluate(net, test_loader):
    print('Evaluating ...')
    correct = 0
    total = 0
    with torch.no_grad():  # no grad when test and predict
        for data in test_loader:
            images, labels = data

            outputs = net(images)
            predicted = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    train_loader, test_loader = mnist_load_data()
    net = MyNet()
    train(net, train_loader)
    evaluate(net, test_loader)
