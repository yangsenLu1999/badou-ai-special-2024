import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from mode_resnet18 import Resnet18

# 使用GPU: 需要添加的地方-->模型--损失函数-- .to(device)
# 使用第 0 个GPU, 判断语句，能使用GPU则使用。
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载数据
# 参数：下载保存路径、train=训练集(True)或者测试集(False)、download=在线(True) 或者 本地(False)、数据类型转换
train_data = torchvision.datasets.CIFAR10("./dataset",
                                          train=True,
                                          download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./dataset",
                                         train=False,
                                         download=True,
                                         transform=torchvision.transforms.ToTensor())
train_len = len(train_data)
val_len = len(test_data)
print("训练数据集合{} = 50000".format(train_len))
print("测试数据集合{} = 10000".format(val_len))
# 格式打包
# 参数：数据、1组几个、下一轮轮是否打乱、进程个数、最后一组是否凑成一组
train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=2, shuffle=True, num_workers=0, drop_last=True)

# 导入网络
tudui = Resnet18(10)
# 使用GPU
tudui = tudui.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 使用GPU
loss_fn = loss_fn.to(device)

# 优化器
# 学习率
learning_rate = 1e-4
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 记录训练次数
train = 0
# 记录测试次数
val = 0
# 训练轮数
epoch = 1000

# writer = SummaryWriter("logs")


for i in range(epoch):
    print()
    print("第{}轮训练开始".format(i + 1))

    # 训练开关-->针对与过拟合的操作层才有效，例如：Dropout，BatchNorm，etc等
    tudui.train(mode=True)
    # 准确率总和
    acc_ = 0
    # 训练
    for data in train_loader:
        imgs, targets = data
        # 使用GPU
        imgs = imgs.to(device)
        targets = targets.to(device)

        # 数据输入模型
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化模型  清零、反向传播、优化器开始优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累计训练次数
        train += 1
        # loss现在看不出来，但应该加 loss.item() 这可让其直接显示数值
        print("\r训练次数:{}，Loss:{}".format(train, loss), end="")

        # 准确率
        accuracy = (outputs.argmax(1) == targets).sum()
        acc_ += accuracy

        if train % 4000 == 0:
            print("训练次数:{}，Loss:{}".format(train, loss))
            # writer.add_scalar("train", loss, train)
    print()
    print("Loss:{}, 准确率：{}".format(loss, acc_/train_len))

    # 测试开关
    tudui.eval()

    # 测试
    total_test_loss = 0
    acc_val = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            # 使用GPU
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)

            # 准确率
            accuracy_val = (outputs.argmax(1) == targets).sum()
            acc_val += accuracy_val

            total_test_loss += loss
            print("\r测试集的Loss:{}".format(total_test_loss), end="")
    print()
    print("整体测试集的Loss:{}, 准确率{}".format(total_test_loss, acc_val/val_len))
    # writer.add_scalar("val", loss, val)
    val += 1

    # 每轮保存模型
    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")
# writer.close()

