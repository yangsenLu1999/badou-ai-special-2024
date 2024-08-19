from model.unet_model_demo import UNetDemo
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch

def train_net(net,device,data_path,epochs=40,batch_size=1,lr=0.00001):
    # 加载数据集
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 优化器和损失函数
    '''
    Momentum：加速收敛，尤其是在平缓或震荡的损失函数区域，
        通过结合当前梯度和之前的梯度方向，使得优化过程更平滑、更快速。
    Weight Decay：通过在损失函数中添加正则项，避免模型参数过大，
        从而防止模型过拟合，提高泛化能力。
    '''
    optimizer = optim.RMSprop(net.parameters(), lr=lr,weight_decay=1e-5,momentum=0.9)
    criterion = nn.BCEWithLogitsLoss() #使用二分类交叉熵损失函数
    best_loss = float('inf')

    # 循环训练
    for epoch in range(epochs):
        net.train(True) # 训练模式
        for image,label in train_loader:
            optimizer.zero_grad() # 梯度清零
            # 将图像和标签数据移动到指定设备
            image = image.to(device=device,dtype=torch.float32)
            label = label.to(device=device,dtype=torch.float32)
            pred = net(image) #前向传播生成预测
            loss = criterion(pred, label) # 计算loss
            print('Loss/train',loss.item()) # 输出当前loss
            if loss.item() < best_loss:
                best_loss = loss.item() #更新最佳loss
                torch.save(net.state_dict(),'best_model.pth') # 保存模型参数
            loss.backward() # 反向传播
            optimizer.step() # 更新参数

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNetDemo(n_channels=1,n_classes=1)
    net.to(device=device)
    data_path = "data/train/"
    train_net(net,device,data_path) # 开始训练
