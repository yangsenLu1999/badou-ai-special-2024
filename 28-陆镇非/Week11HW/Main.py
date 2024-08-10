import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import time
import os

class PytorchClassificationModel(nn.Module):
    def __init__(self, inputDim, outputDim, device):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpooling = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(64*8*8, 128)
        self.linear2 = nn.Linear(128, outputDim)
        self.loss = nn.CrossEntropyLoss()
        self.device = device
        if(torch.cuda.is_available()):
            print("model is in GPU")
            self.to(self.device)
        else:
            print("model is in CPU")

    def forward(self, x):
        x = self.maxpooling(torch.relu(self.conv1(x)))
        x = self.maxpooling(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # dim=1, [batchSize,64*7*7]
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def calLoss(self, y_predict, y):
        return self.loss(y_predict, y)

    # def data_iter(self, batch_size, features, labels):
    #     num_examples = len(features)
    #     indices = list(range(num_examples))
    #     random.shuffle(indices)
    #     for i in range(0, num_examples, batch_size):
    #         batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
    #         yield features[batch_indices], labels[batch_indices]

    def fit(self, trainingDataSet, epoch, learning_rate, batch_size):
        N = len(trainingDataSet)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        metric = dict()
        metric["avg_loss"] = list()
        metric["accuracy"] = list()
        self.train()
        for i in range(1, epoch+1):
            avg_loss = 0
            acc = 0
            if(torch.cuda.is_available()):
                for X, Y in DataLoader(trainingDataSet, batch_size=batch_size, shuffle=True,
                                       num_workers=int(os.cpu_count()/2),
                                       pin_memory=True):
                    X, Y = X.to(self.device), Y.to(self.device)
                    optimizer.zero_grad()
                    Y_predict = self.forward(X)
                    loss = self.calLoss(Y_predict, Y)
                    loss.backward()
                    optimizer.step()
                    avg_loss = avg_loss + loss
                    acc = acc + self.calAccuracy(Y_predict, Y)
            else:
                for X, Y in DataLoader(trainingDataSet, batch_size=batch_size, shuffle=True):
                    optimizer.zero_grad()
                    Y_predict = self.forward(X)
                    loss = self.calLoss(Y_predict, Y)
                    loss.backward()
                    optimizer.step()
                    avg_loss = avg_loss + loss
                    acc = acc + self.calAccuracy(Y_predict, Y)
            avg_loss = avg_loss/(int(N/batch_size))
            acc = acc/N
            metric["avg_loss"].append(avg_loss.item())
            metric["accuracy"].append(acc.item())
            if i % 10 == 0 or i == epoch:
                print(f"Epoch: {i}, AvgLoss: {avg_loss}, Accuracy: {acc}")
        return metric

    def calAccuracy(self, y_predict, y, showEachResult:bool=False):
        acc = 0
        correct = 0
        N = y.shape[0]
        self.eval()
        with torch.no_grad():  # not calculate auto-grad when doing fwd
            y_predict_index = (y_predict.argmax(dim=1, keepdims=True)).squeeze(1)
            # print(y_predict_index)
            equal_elements = torch.eq(y_predict_index, y)
            # print(equal_elements)
            count = torch.sum(equal_elements)
            # print(count)
            correct = count
            if showEachResult:
                for i in range(0, N):
                    if(y_predict_index[i] == y[i]):
                        print(f"y truth: {y[i]}, y predict: {y_predict_index[i]}, result: {'Correct'}")
                    else:
                        print(f"y truth: {y[i]}, y predict: {y_predict_index[i]}, result: {'Wrong'}")
        acc = correct
        return acc

    def plotMetric(self, metric, showImmediately):
        plt.figure()
        i = 1
        N = len(metric)
        for key, value in metric.items():
            plt.subplot(1, N, i)
            plt.plot(np.linspace(1,len(value),len(value)), value)
            plt.title(key)
            plt.xlabel('epoch')
            # plt.ylabel('value')
            i = i + 1
        if (showImmediately):
            plt.show()

    def saveModelWeights(self, localPath):
        torch.save(self.state_dict(), localPath)  # touch static method

    def loadModelWeights(self, localPath):
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(localPath))  # load weights  .path, ***model structure must be same
        else:
            self.load_state_dict(torch.load(localPath, map_location=torch.device('cpu')))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("CPU cores: " + str(os.cpu_count()))  # cpu kernels

    # normalizer
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # mean, std-var
    ])

    # download or read MNIST data
    mnist_train = datasets.CIFAR10('./data/', train=True, download=True, transform=transform)
    mnist_test = datasets.CIFAR10('./data/', train=False, download=True, transform=transform)
    # print(mnist_train[0][0].shape)  # shape (3,32,32)  channel first
    # print(mnist_train[0][1])  # index label (int)

    # train model
    model = PytorchClassificationModel(inputDim=3*32*32, outputDim=10, device=device)
    start_time = time.time()
    hist = model.fit(mnist_train, epoch=100, learning_rate=0.001, batch_size=512)
    end_time = time.time()
    print("model training excuted time cost：", end_time - start_time, "seconds")
    # save model weights
    model_path = "myClassifyModelWeight.pth"
    model.saveModelWeights(model_path)

    # load saved model
    loadedModel = PytorchClassificationModel(inputDim=3 * 32 * 32, outputDim=10, device=device)
    loadedModel.loadModelWeights(localPath="./myClassifyModelWeight.pth")
    x = torch.tensor(np.array([mnist_test[i][0].numpy() for i in range(len(mnist_test))]))
    print(x.shape)
    y = torch.LongTensor(np.array([mnist_test[i][1] for i in range(len(mnist_test))]))
    N = y.shape[0]
    print(y.shape)
    if torch.cuda.is_available():
        x, y = x.to(device), y.to(device)
    y_predict = loadedModel(x)
    acc = loadedModel.calAccuracy(y_predict, y, showEachResult=True)
    print(f"accuracy: {(acc/N).item()}")

    model.plotMetric(hist, showImmediately=True)


