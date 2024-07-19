import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

'''第三步： 把数据送入网络进行训练  和  测试'''
class Model:  # 创建模型训练的过程
    def __init__(self, net, cost, optimist):  # self 现在表示的是 这个model的（在主函数里） 实例。需要确定训练中的（模型架构，损失函数，和优化项），这些需要定义一个实例传入这些实参
        self.net = net
        self.cost = self.create_cost(cost)    #  cost的属性 等于 方法createcost 的值
        self.optimizer = self.create_optimizer(optimist)
        pass
    def create_cost(self, cost):            #cost实参输入之后，返回到这个键对应的值
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]  #cost实参输入之后，返回到这个键对应的值

    def create_optimizer(self, optimist, **rests):
        support_optim = {
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
                # .zero_grad() 方法的作用是将所有模型参数的梯度归零，以确保每次计算的梯度是正确的，即只反映了当前迭代的损失相对于参数的梯度。
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)   #  计算损失函数
                loss.backward()  #  反向传播
                #step 是优化器对象的一个方法，在调用这个方法之前，通常需要先计算模型参数的梯度，这通常通过调用.backward()方法在损失函数上完成
                # .step()方法会根据这些梯度和优化器的配置（如学习率）来更新模型的参数。这个过程称为梯度下降。
                self.optimizer.step()

                running_loss += loss.item()
                if i %100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0
        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating............')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data

                outputs =self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct +=(predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

'''第二步：编写数据的标签和路径索引 '''
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

'''第一步：编写好网络'''
# super中（mnist ）参数表示这个子类调用了父类的方法，可以告诉是哪个类调用了父类的方法，因为可能有多个父类
 # self指向当前实例的引用，self代表正在创建的对象
class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)    #  每个模块（如全连接层、卷积层等）都是nn.Module的一个实例
        self.fc2 = torch.nn.Linear(512, 512)      # self.fc1是 一个全连接层的实例。通过调用self.fc1(input)，实际上是在调用这个全连接层实例的forward方法，将输入数据input传递给这个层进行处理
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)    # -1表示在这个维度不变化
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)  # 输入张量的形状是 (N, C)，当dim=1 时，意味着 Softmax 函数将在输入张量的第二个维度上进行计算

        return x

if __name__ == '__main__':
    # train for mnist
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')  # 创建model的实例，这个实例的属性 self.net=net  是就是创建一个 MnistNet 的一个实例
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)  # 调用类Model 的方法 （train），需要传入实参 ——训练数据，epochs是函数的默认参数，则不需要重新传入
    model.evaluate(test_loader)

