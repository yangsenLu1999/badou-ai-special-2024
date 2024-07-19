import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Model:
    def __init__(self,net):
        self.net=net
        self.loss=nn.CrossEntropyLoss()
        self.optim=optim.Adam(self.net.parameters(),lr=0.0001)

    def train(self,train_data,epochs):
        #此处使用小批量梯度下降法，即一个batch之后，更新一次权重
        for epoch in range(epochs):
            losses=0
            for i,data in enumerate(train_data):        #data是list，里面有数据和标签
                input=data[0]                           #input是32*1*28*28
                label=data[1]
                #1.首先进行梯度清零
                self.optim.zero_grad()
                #2.正向传播
                output=self.net(input)
                #3.计算loss
                loss=self.loss(output,label)
                #4.反向传播
                loss.backward()
                #5.更新权重
                self.optim.step()

                losses+=loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_data)*100, losses / 100))
                    losses = 0.0

    def evaluate(self,test_data):
        total = 0
        correct = 0
        with torch.no_grad():                   #训练集的数据不参与运算
            for data in test_data:
                input=data[0]
                label=data[1]
                output = self.net(input)
                predicted = torch.argmax(output, 1)       #找到最大值的下标
                total += label.size(0)
                correct += (predicted == label).sum().item()  #item()函数用于将一个标量张量（只有一个元素的张量）转换为Python的标准数值类型（如int或float）
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))





class net(nn.Module):
    def __init__(self,input_dim):
        super(net,self).__init__()
        self.input_dim=input_dim
        self.l1=nn.Linear(input_dim,512)
        self.l2=nn.Linear(512,512)
        self.l3=nn.Linear(512,10)   #10分类任务

    def forward(self,x):
        x=x.reshape(-1,self.input_dim)   #与x.view等价
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.softmax(self.l3(x),dim=1)    #最后的输出是batch_size*10
        return x

def data_loader():
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0,], [1,])])
    '''
    transforms.Compose([transforms.ToTensor(), transforms.Normalize([0,], [1,])]) 是一个用于图像预处理的变换序列，它包含了两个变换操作：
    transforms.ToTensor(): 这个变换将输入的PIL图像或者NumPy数组转换为PyTorch张量（tensor）。在这个过程中，它会将图像的数据类型从uint8转换为float，并将像素值范围从[0, 255]缩放到[0.0, 1.0]。
    transforms.Normalize([0,], [1,]): 这个变换对输入的张量进行标准化。它接受两个参数，分别是均值（mean）和标准差（std），用于对每个通道进行标准化。
    在这个例子中，均值为[0,]，标准差为[1,]，这意味着每个通道的值将被减去0并除以1，实际上不会改变张量的值。通常情况下，均值和标准差会根据训练数据集的统计信息来确定，以便在训练过程中更好地适应数据分布
    '''
    trainset = torchvision.datasets.MNIST(root='./data', train=True,                   #根据train是什么来区分是训练集还是测试集
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,                 #生成一个迭代器，trainset是一个类，num_workers 参数表示在加载数据时使用的子进程数量。
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader

if __name__=="__main__":
    #1.读取数据
    train_data,test_data=data_loader()
    #print(len(train_data))             #1875,即60000/32=1875，len(train_data)代表有多少个batch
    #2:初始化模型
    dim=np.array(train_data.dataset[0][0]).shape              #1*28*28
    myNet=net(dim[1]*dim[2])
    model=Model(myNet)
    #3.训练数据
    model.train(train_data,10)
    #4.评测模型
    model.evaluate(test_data)