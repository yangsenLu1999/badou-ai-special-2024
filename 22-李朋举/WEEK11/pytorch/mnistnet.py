import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Model:
    # net-网络模型  cost-损失函数  optimist-优化项
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    """
    用于创建损失函数： 损失函数用于描述模型预测值与真实值的差距大小
        1. `support_cost` 是一个字典，其中键是损失函数的名称，值是对应的损失函数对象。    (如果输入的实数、无界的值, 使用MSE比较合适)
        2. `nn.CrossEntropyLoss()` 是 PyTorch 中提供的交叉熵损失函数，用于多分类问题。 (如果输入标签是 矢量(分类标志) , 使用交叉熵比较合适)
        3. `nn.MSELoss()` 是均方误差损失函数，用于回归问题。        
        这样，在其他地方可以通过调用 `create_cost` 方法并传入损失函数名称来创建相应的损失函数对象，以便在训练模型时使用。        
    """
    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),  # 交叉熵
            'MSE': nn.MSELoss()  # MSE
        }
        return support_cost[cost]

    # 优化算法 (优化项本身可选)
    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),  # 随机梯度下降 精度要求不高
             # momentum解决SGD优化算法摆动幅度大的问题  RMSP进一步解决百度幅度大的问题  收敛速度快
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests),  # 比momentum进一步解决SGD优化算法摆动幅度大的问题, 收敛速度快  √
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),  # 先做momentum再做RMSP 收敛速度慢
        }
        return support_optim[optimist]


    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0   #在每次训练轮开始时，将运行损失初始化为 0
            """
            `enumerate(train_loader, 0)` 是 Python 中的一个函数，用于将一个可迭代对象转换为包含索引和元素的元组序列。
             在这个例子中，enumerate(train_loader, 0)` 将从索引 0 开始，为 `train_loader` 中的每个元素生成一个包含索引和元素的元组。
            """
            for i, data in enumerate(train_loader, 0):  # 遍历训练数据加载器 train_loader 中的数据批次
                inputs, labels = data  # 获取输入数据 inputs 和标签数据

                self.optimizer.zero_grad() # 在每次迭代之前，将优化器的梯度清零，以避免梯度累积。

                # forward + backward + optimize
                outputs = self.net(inputs)  # 正向过程:  输入  经过正向过程  输出 （softmax结果）
                loss = self.cost(outputs, labels)  # 计算损失函数  (计算输出结果与真实标签之间的损失函数值 loss)
                """
                `loss.backward()` 是 PyTorch 中的一个函数，用于计算损失函数的梯度。
                    在深度学习中，我们通过计算损失函数来衡量模型的预测结果与真实标签之间的差异。然后，我们使用反向传播算法来计算损失函数对模型参数的梯度，以便更新模型的参数，从而使模型能够更好地拟合数据。
                    `loss.backward()` 函数的作用是将损失函数的梯度传播回模型的参数。具体来说，它会计算损失函数对每个参数的导数，并将这些导数存储在参数的 `grad` 属性中。 
                                      然后，我们可以使用优化器（如随机梯度下降）来更新模型的参数，以减小损失函数的值。                    
                    需要注意的是，在使用 `loss.backward()` 函数之前，我们需要确保模型的参数已经正确地初始化，并且模型已经被正确地训练。此外，我们还需要在每次迭代中清零模型的梯度，以避免梯度的累积。
                """
                loss.backward()  # 进行反向传播，计算梯度。
                """
                `self.optimizer.step()` 这行代码的作用是根据计算得到的梯度，使用优化器对模型的参数进行更新。
                    在深度学习中，训练过程通常涉及到优化算法，用于调整模型的参数以最小化损失函数。优化器（如随机梯度下降 SGD、RMSP、ADAM 等）负责根据损失函数的梯度来更新模型的参数。                     
                        1. `self.optimizer` 是一个优化器对象，它已经在代码的其他地方被创建和初始化，负责根据计算得到的梯度来更新模型的参数。
                        2. `step()` 是优化器对象的一个方法，用于执行参数更新的步骤。
                        3. 在每次训练迭代中，模型会根据输入数据进行前向传播，计算损失函数。然后，通过反向传播算法计算损失函数对模型参数的梯度。
                        4. 最后，调用 `self.optimizer.step()` 会根据计算得到的梯度来更新模型的参数。优化器会根据其内部的算法和策略来调整参数的值，以朝着损失函数减小的方向进行优化。
                    通过不断重复这个过程，模型的参数会逐渐调整，从而使模型能够更好地拟合训练数据，并提高其性能。                    
                """
                self.optimizer.step() # 根据计算得到的梯度，使用优化器进行参数更新。

                """
                将当前批次的损失值累加到运行损失中。
                    在训练神经网络时，通常会计算每个批次数据的损失值。通过将这些损失值累加起来，可以得到整个训练过程中的总损失。
                        1. `running_loss` 是一个变量，用于存储运行损失。在每次训练迭代开始时，`running_loss` 会被初始化为 0。
                        2. `loss.item()` 表示当前批次的损失值。`loss` 是通过计算模型的输出与真实标签之间的差异得到的。
                        3. `running_loss += loss.item()` 将当前批次的损失值累加到 `running_loss` 中。
                    通过累加每个批次的损失值，可以得到训练过程中的总损失。这对于监控训练进度、调整超参数以及评估模型性能都非常有用。
                """
                running_loss += loss.item() # 将当前批次的损失值累加到运行损失中。
                if i % 100 == 0:  # 每经过 100 个批次，打印一次训练信息，包括当前训练轮数、进度百分比、平均损失等。
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')


    def evaluate(self, test_loader):
        print('Evaluating ...')
        # 初始化两个变量，分别用于记录正确预测的数量和总预测数量。
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict  推理时不需要计算导数,所有变量、函数都不需要求导
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)  # 正向过程:  输入  经过正向过程  输出 （softmax结果）
                """
                在这段代码中，训练过程中没有直接使用 `argmax` 函数。
                        `argmax` 函数通常用于在给定的一组值中找到最大值的索引。在训练过程中，可能会涉及到对输出结果的处理，但具体的处理方式取决于模型的设计和任务需求。
                        在这个例子中，可能的情况是：
                        1. 模型的输出是经过 Softmax 激活函数处理的概率分布。Softmax 函数将输出值转换为概率形式，使得每个输出值都在 0 到 1 之间，并且所有输出值的和为 1。
                        2. 在训练过程中，通常会使用损失函数来衡量模型的预测结果与真实标签之间的差异。常见的损失函数如交叉熵损失函数（CrossEntropyLoss）会自动处理概率分布和标签之间的关系，并不需要显式地使用 `argmax`。
                        3. 在评估模型性能或进行预测时，可能会根据输出的概率分布来做出决策。例如，可以选择概率最大的类别作为预测结果，或者根据概率分布进行其他的分析和处理。
                        总之，虽然代码中没有直接使用 `argmax`，但它可能在模型的其他部分或后续的处理中被隐式地使用，具体取决于模型的实现和应用场景。
                """
                predicted = torch.argmax(outputs, 1)  # argmax 排序 取概率最高的
                total += labels.size(0)  # 统计推理了多少张图
                correct += (predicted == labels).sum().item()  # 统计正确率

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


"""
加载 MNIST 数据集: 
    1. `transforms.Compose`：这是 PyTorch 中的一个函数，用于将一系列数据变换组合在一起。在这里，我们使用了两个变换：
        - `transforms.ToTensor()`：将数据转换为张量。
        - `transforms.Normalize([0, ], [1, ])`：对数据进行标准化，将均值设置为 0，标准差设置为 1。
        
    2. `torchvision.datasets.MNIST`：这是 PyTorch 中提供的 MNIST 数据集加载器。 通过设置 `root` 参数指定数据集的保存路径，`train=True` 表示加载训练集，
                                    `download=True` 表示如果数据集不存在则自动下载，`transform=transform` 应用上述定义的变换。
    
    3. `torch.utils.data.DataLoader`：这是 PyTorch 中的数据加载器，用于将数据集分成批次并进行加载。
                                     通过设置 `batch_size` 参数指定每个批次的大小，`shuffle=True` 表示在每个 epoch 时打乱数据顺序，`num_workers=2` 指定使用的线程数。
    
    4. 函数返回两个数据加载器：`trainloader` 用于加载训练集，`testloader` 用于加载测试集。
    
"""
def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0, ], [1, ])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader


# 网络模型的构建
class MnistNet(torch.nn.Module):
    # 定义需要训练的层  通常将需要训练的层写在 init 函数中 (卷积、全连接 ...)
    def __init__(self):
        """
        `Linear` 类继承自 `torch.nn.Module` 类, 调用父类的 `__init__` 方法的主要作用是确保父类的属性和方法被正确初始化。
         这样就可以在 `Linear` 类中使用父类提供的功能，同时也可以根据需要添加自己的定制化逻辑。
         这段代码使用 PyTorch 库定义了一个全连接层（fully connected layer），将输入维度为 28×28 的数据映射到 512 个神经元。

        - `self.fc1 = torch.nn.Linear(28 * 28, 512)`：这行代码创建了一个名为 `fc1` 的全连接层对象。
            - `torch.nn.Linear`：这是 PyTorch 中用于定义线性层的类。
            - `28 * 28`：表示输入数据的维度，即 28 乘以 28 的图像大小。
            - `512`：表示输出神经元的数量，即全连接层将把输入数据映射到 512 个神经元上。

        全连接神经网络的特点是每个神经元都与前一层的所有神经元相连，这种连接方式可以捕捉输入数据中的全局特征，但对于高维数据可能会存在参数过多的问题。
        相比之下，卷积神经网络（Convolutional Neural Network，CNN）通常在图像识别等任务中表现出色，它利用卷积核在输入数据上进行滑动窗口操作，以提取局部特征，从而减少了参数数量。
        判断一个神经网络是否为全连接，可以根据以下几个特征：
            连接方式：在全连接神经网络中，每一个神经元都与前一层的所有神经元相连。也就是说，对于每一个神经元，它的输入来自于前一层的所有神经元的输出。
            权重矩阵：全连接神经网络的权重矩阵是一个二维矩阵，其行数等于当前层的神经元数量，列数等于前一层的神经元数量。
            计算过程：在全连接神经网络中，计算当前层的神经元输出时，需要将前一层的所有神经元输出与对应的权重相乘，然后将结果相加，再加上偏置项。
        """
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    # 定义参数不需要训练的层   将参数不需要训练的层在 forward 方法里(激活函数、Relu、+ - * /  ...)
    def forward(self, x):
        """
        将输入数据 x 进行形状重塑，将其转换为一个二维张量，其中第一维的大小可以根据输入数据的数量自动确定，第二维的大小为 28 * 28。
        这样的重塑操作通常用于将输入数据展平为一维向量，以便后续的线性层处理。
        """
        x = x.view(-1, 28 * 28)  # reshape
        """
        使用了 PyTorch 中的激活函数 ReLU 对线性层 self.fc1 的输出进行激活。
        ReLU 函数将输入值限制为非负值，对于小于 0 的输入值，输出为 0，对于大于 0 的输入值，输出保持不变。这样的激活函数可以引入非线性，增加模型的表达能力。
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        """
        使用 Softmax 函数对线性层 self.fc3 的输出进行归一化处理。
        Softmax 函数将输入值转换为概率分布，使得每个输出值都在 0 到 1 之间，并且所有输出值的和为 1。 dim=1 参数指定了在第二维上进行归一化。
        """
        x = F.softmax(self.fc3(x), dim=1)
        return x

    # backward 反向自动求导


if __name__ == '__main__':
    # train for mnist
    # [1] 写好网络
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    # [2] 编写好数据的标签和路径索引
    train_loader, test_loader = mnist_load_data()
    # [3] 把数据送到网络
    model.train(train_loader)
    model.evaluate(test_loader)
