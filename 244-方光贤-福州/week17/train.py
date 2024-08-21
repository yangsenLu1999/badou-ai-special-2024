from model.unet_model import UNet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch


# 设置训练网络过程 batch_size为1 训练轮次为40 学习率为0.00001 这是因为样本较小
def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    # 定义RMSprop算法 设置衰减率 学习率和动量
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # 将best_loss初始化为正无穷
    best_loss = float('inf')
    # 训练迭代
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练 将标签和数据对应
        for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数 输出预测结果
            pred = net(image)
            # 计算预测值和真实值的损失值
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            # 更新权重参数
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    # 选择cuda加速 没有cuda就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络 图片单通道且类别只有1类
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce里
    net.to(device=device)
    # 指定数据集地址 开始训练
    data_path = "data/train/"
    train_net(net, device, data_path)