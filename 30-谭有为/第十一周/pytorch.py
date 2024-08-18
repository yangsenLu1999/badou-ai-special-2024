import torch
import torch.nn
import torchvision
import torchvision.transforms
import torch.nn.functional
import torch.optim


#定义神经网络模型
class Mnistnet(torch.nn.Module):
    def __init__(self):
        super(Mnistnet,self).__init__()   #继承自父类nn.Module的属性进行初始化
        #定义输入、隐藏、输出层结点数,使用全连接
        self.fc1=torch.nn.Linear(28*28,512)
        self.fc2=torch.nn.Linear(512,512)
        self.fc3=torch.nn.Linear(512,10)

    #正向传播过程
    def forward(self,x):
        x=x.view(-1,28*28)
        x=torch.nn.functional.relu(self.fc1(x))
        x=torch.nn.functional.relu(self.fc2(x))
        x=torch.nn.functional.softmax(self.fc3(x),dim=1)   #softmax函数： 将一组实时转换为概率分布的实数  dim跟tensor的维度有关
        return x


#定义神经网络
class Model:
    #参数初始化
    def __init__(self,net,cost,optimist):
        self.net=net
        self.cost=self.create_cost(cost)   #损失函数
        self.optimizer=self.support_optimizer(optimist)   #优化器
        pass

   #损失函数
    def create_cost(self,cost):
        support_cost={
            'Cross_EntropyLoss':torch.nn.CrossEntropyLoss(), #交叉熵
            'MSE':torch.nn.MSELoss()   #均方误差
        }
        return support_cost[cost]

#优化器选择   SGD--随机梯度下降，使用一批样本来计算梯度，更新权重    ADAM--- Momentum算法和RMSProp算法结合起来使用的一种算法    RMSP--一种优化摆动幅度过大的方式
    def support_optimizer(self,optimist,**rests):
        support_optim={
            'SGD':torch.optim.SGD(self.net.parameters(),lr=0.1,**rests),
            'ADAM':torch.optim.Adam(self.net.parameters(),lr=0.01,**rests),
            'RMSP':torch.optim.RMSprop(self.net.parameters(),lr=0.001,**rests)
        }
        return support_optim[optimist]

    #训练
    def train(self,train_datas,epoches=3):
        for e in range(epoches):
            _loss=0.0
            for i,data in enumerate(train_datas,0):
                inputs,labels=data
                self.optimizer.zero_grad()  #梯度归0
                outputs=self.net(inputs)
                loss=self.cost(outputs,labels)
                loss.backward()
                self.optimizer.step()
                _loss+=loss.item()   #item() 将一个Tensor变量转换为python标量,常用于用于深度学习训练时，将loss值转换为标量并加
                #一次训练100个
                if i%100==0:
                    # %%表示输出一个%
                    print('[epoch %d，%.2f%%] loss:%.3f' %(e+1,(i+1)*100/len(train_datas),_loss/100))
                    _loss=0
        print('train finish!')

    #测试
    def evaluate(self,test_datas):
        print('****Evaluating....****')
        corrent=0
        total=0
        with torch.no_grad():
            for data in test_datas:
                imgs,labels=data
                outputs=self.net(imgs)
                #torch.argmax 返回最大值的索引
                predict=torch.argmax(outputs,1)
                total+=labels.size(0)
                corrent += (predict == labels).sum().item()

        print(total,corrent)
        print('this test''s accuracy is: %d%%' %(100*corrent/total))

#导入数据集
#torchvision.transforms.Compose---将多个步骤整合
#torchvision.transforms.ToTensor  把0-255 变换到0-1，将HWC变为CHW
#torchvision.transforms.Normalize  #数据标准化 即均值为0  标准差为1，可使模型更收敛
def mnist_load_data():
     transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize([0,], [1,])])
     #读取数据集并下载到本地并转换
     #参考文档：https://blog.csdn.net/qq_38406029/article/details/121672906
     trainset=torchvision.datasets.MNIST(root='F://data',train=True,download=True,transform=transform)
     #torch.utils.data.DataLoader--该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tenso
     #参考文档：https://blog.csdn.net/sazass/article/details/116641511
     train_data=torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=True,num_workers=2)
     testset=torchvision.datasets.MNIST(root='F://data',train=True,download=True,transform=transform)
     test_data=torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=True,num_workers=2)
     return train_data,test_data

if __name__=='__main__':
    net=Mnistnet()
    model=Model(net,'Cross_EntropyLoss','RMSP')
    train_datas,test_datas=mnist_load_data()
    model.train(train_datas)
    model.evaluate(test_datas)






