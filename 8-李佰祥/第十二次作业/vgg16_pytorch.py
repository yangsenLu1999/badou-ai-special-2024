import os.path

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
class VGG16_net(torch.nn.Module):
    def __init__(self):
        super(VGG16_net,self).__init__()
        self.net = torch.nn.Sequential(
            #由224*224*3 卷积后到224，224，64
            nn.Conv2d(3,64,3,1,1),
            nn.ReLU(inplace=True),#当inplace参数设置为True时，ReLU函数将会直接在输入张量上修改其值，所有小于0的元素会被原地替换为0。这种方式可以节省内存，因为它避免了创建额外的输出张量。然而，这也意味着输入张量将被永久改变，之后不能用于需要原始值的其他操作
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True),
            #max_pool后到112,112,64
            nn.MaxPool2d(2,2),

            #由112,112,64到112，112，128
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            #max_pool由112，112，128，下降到56，56，128
            nn.MaxPool2d(2,2),

            #由56，56，128 到56，56，256
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            #由56，56，256 到 28，28，256
            nn.MaxPool2d(2,2),

            #由28，28，256到28，28，512
            nn.Conv2d(256,512,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            #由28，28，512到14，14，512
            nn.MaxPool2d(2,2),


            #由14，14，512到 14，14，512
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            #由 14，14，512 到7，7，512
            nn.MaxPool2d(2,2)
         )
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))

        self.classifier = nn.Sequential(
            nn.Linear(7*7*512,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,2)
        )

    def forward(self,x):
        x = self.net(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x


class customImageDataset(Dataset):
    def __init__(self,image_root, transform=None,target_transform=None):
        self.image_labels = self.read_txt()
        self.img_dir =image_root
        self.transform = transform
        self.target_transform = target_transform
    def read_txt(self):
        with open('../../VGG16-tensorflow-master/dataset.txt','r') as f:
            lines = f.readlines()
            np.random.shuffle(lines)
            annotation =[line.strip().split(';') for line in lines]
            return annotation

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path =os.path.join(self.img_dir,self.image_labels[idx][0])
        img = Image.open(img_path).convert('RGB')
        label = self.image_labels[idx][1]
        label = int(label)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        else:
            label= torch.tensor(label,dtype=torch.long)
        return img,label


class Model():
    def __init__(self,vgg16_net,loss,optimizer):
        self.vgg16_net = vgg16_net
        self.loss = loss
        self.optimizer = optimizer

    def train(self,dataloader, epochs=3):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.vgg16_net(inputs)
                loss_value = self.loss(outputs, labels)
                loss_value.backward()
                self.optimizer.step()

                running_loss += loss_value.item()

                print('[epoch %d, loss: %.3f]' %
                      (epoch, loss_value.item()))
        print('train结束')

    def eval(self,test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.vgg16_net(inputs)
                p = torch.argmax(outputs,dim=1)
                total +=labels.size(0)
                correct += (p==labels).sum().item()
                print('total: ',total)
                print('correct: ', correct)









if __name__ == '__main__':
    transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    vgg16_net = VGG16_net()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vgg16_net.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg16_net.parameters(), lr=0.001)

    dataset = customImageDataset('../../VGG16-tensorflow-master/train2/',transforms)
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=60,shuffle=True,num_workers=3)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True, num_workers=2)

    model = Model(vgg16_net,loss,optimizer)
    model.train(data_loader)
    model.eval(test_loader)














#















