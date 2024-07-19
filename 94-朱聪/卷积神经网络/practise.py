# ======================= PyTorch =======================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(nn.Module):
    """
    定义一个神经网络
    """
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512) # 全连接层
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x)) # 一般不涉及参数更改的层使用F
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x


net = Net()

def train(train_loader, epoches=3):
    # train_loader = DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])), batch_size=64, shuffle=True)

    # momentum是动量参数，动量可以加速训练过程，特别是处理高曲率、小但一致的梯度以及噪声，默认为0
    optimizer = optim.RMSprop(net.parameters(), lr=0.001, momentum=0.5)

    for epoch in range(epoches):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            output = net(inputs)
            loss = nn.CrossEntropyLoss(output, labels)
            loss.backward()
            optimizer.step()

    # torch.save(net.state_dict(), 'mnist_cnn.pt')
    # print('Train Done!')


def test(test_loader):
    correct = 0
    total = 0
    with torch.no_grad(): # 使用 torch.no_grad() 来关闭梯度，因为测试的时候不需要计算梯度
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0) # 累加当前批次中样本的总数 .size(0) 返回的是该张量的第一个维度的大小
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


# ======================= TensorFlow =======================

import tensorflow as tf

# 按部就班，一层层往下走

x = tf.placeholder(tf.float32, [None, 1]) # 参数：type, shape, name
y = tf.placeholder(tf.float32, [None, 1]) # 定义了一个列向量

Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.placeholder(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(weights, x) + bias
L1 = tf.nn.tanh(Wx_plus_b_L1)

Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))  # 加入偏置项
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2 # 得到了(?, 1)的列向量
prediction = tf.nn.tanh(Wx_plus_b_L2)


loss = tf.reduce_mean(tf.square(y - prediction))
# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})


# ======================= Keras =======================

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,))) # 神经元个数，激活函数，输入
network.add(layers.Dense(10, activation='softmax'))

# 编译模型，优化器为 RMSprop,损失函数为分类交叉熵,评估指标为准确率
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28)) # 变成一个28*28的一位数组 要区别开 reshape((60000, 1, 28 * 28))
train_images = train_images.astype('float32') / 255 # 60000行数据，每行都是28*28归一化的一位数组内容

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels) # 返回(60000,10)的数据，10是类别数。每一行数据都是一个10bit位内容
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)

res = network.predict(test_images)

for i in range(10000): # 对所有图片送入模型预测
    for j in range(res[i].shape[0]):
        if (res[i][j] == 1):
            if test_labels[i] != j:
                print('incorrect')
            print("the picture correct number is:", test_labels[i])
            print("the number for the picture is : ", j)