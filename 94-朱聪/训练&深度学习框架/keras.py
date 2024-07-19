'''
[1]
将训练数据和检测数据加载到内存中(第一次运行需要下载数据，会比较慢):
train_images是用于训练系统的手写数字图片;
train_labels是用于标注图片的信息;
test_images是用于检测系统训练效果的图片；
test_labels是test_images图片对应的数字标签。
'''
# from tensorflow.keras.datasets import mnist
import numpy as np

def load_data():
    with np.load('mnist.npz') as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)

(train_images, train_labels), (test_images, test_labels) = mload_data()
print('train_images.shape = ', train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)
'''
1.train_images.shape打印结果表明，train_images是一个含有60000个元素的数组.
数组中的元素是一个二维数组，二维数组的行和列都是28.
也就是说，一个数字图片的大小是28*28. 60000张28*28的图片
2.train_lables打印结果表明，第一张手写数字图片的内容是数字5，第二种图片是数字0，以此类推.
3.test_images.shape的打印结果表示，用于检验训练效果的图片有10000张。
4.test_labels输出结果表明，用于检测的第一张图片内容是数字7，第二张是数字2，依次类推。
'''


'''
[2]
把用于测试的第一张图片打印出来看看
'''
digit = test_images[0]
import matplotlib.pyplot as plt

plt.imshow(digit, cmap=plt.cm.binary) # plt.cm.binary是预定义的颜色映射，低值-黑色，高值-白色，中间值-灰色
plt.show()


'''
[3]
使用tensorflow.Keras搭建一个有效识别图案的神经网络，在Tensorflow中，推荐使用 Keras 来构建模型
1.layers:表示神经网络中的一个数据处理层。(dense:全连接层)
2.models.Sequential():表示把每一个数据处理层串联起来. 允许我们按顺序堆叠神经网络层
3.layers.Dense(…):构造一个数据处理层。
4.input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系.
'''

'''
models 模块提供了 Sequential 类和 Model 类，用于定义和组织神经网络模型
使用 Sequential 类创建的模型是线性堆叠的神经网络，适合于简单的前馈网络结构

Model 类更加灵活，允许我们定义包含多输入和多输出的复杂神经网络结构
与 Sequential 不同，Model 类需要显式地指定输入张量和输出张量，以定义模型的输入输出关系
ex：
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 定义输入层
input_layer = Input(shape=(28 * 28,))

# 定义中间层
hidden_layer = Dense(512, activation='relu')(input_layer)

# 定义输出层
output_layer = Dense(10, activation='softmax')(hidden_layer)

# 创建模型
model = Model(inputs=input_layer, outputs=output_layer)    
'''

from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,))) # 神经元个数，激活函数，输入
network.add(layers.Dense(10, activation='softmax'))

# 编译模型，优化器为 RMSprop,损失函数为分类交叉熵,评估指标为准确率
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


'''
[4]
在把数据输入到网络模型之前，把数据做归一化处理:
1.reshape(60000, 28*28）:train_images数组原来含有60000个元素，每个元素是一个28行，28列的二维数组，
现在把每个二维数组转变为一个含有28*28个元素的一维数组.
2.由于数字图案是一个灰度图，图片中每个像素点值的大小范围在0到255之间.
3.train_images.astype(“float32”)/255 把每个像素点的值从范围0-255转变为范围在0-1之间的浮点值。
'''
train_images = train_images.reshape((60000, 28 * 28)) # 变成一个28*28的一位数组 要区别开 reshape((60000, 1, 28 * 28))
train_images = train_images.astype('float32') / 255 # 60000行数据，每行都是28*28归一化的一位数组内容

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

'''
把图片对应的标记也做一个更改：
目前所有图片的数字图案对应的是0到9。
例如test_images[0]对应的是数字7的手写图案，那么其对应的标记test_labels[0]的值就是7。
我们需要把数值7变成一个含有10个元素的数组，然后在第8个元素设置为1，其他元素设置为0。
例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0] ---one hot


to_categorical  独热编码转换，适用于分类问题中的标签处理。
它将每个类别表示为一个向量，其中只有一个元素为1，其余都为0。例如，对于一个有3个类别的问题，可能的独热编码如下：
类别1: [1, 0, 0]
类别2: [0, 1, 0]
类别3: [0, 0, 1]

'''
from tensorflow.keras.utils import to_categorical

print("before change:", test_labels[0])
train_labels = to_categorical(train_labels) # 返回(60000,10)的数据，10是类别数。每一行数据都是一个10bit位内容
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])


'''
[5]
把数据输入网络进行训练：
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算的循环是五次
'''
network.fit(train_images, train_labels, epochs=5, batch_size=128)


'''
[6]
测试数据输入，检验网络学习后的图片识别效果.
识别效果与硬件有关（CPU/GPU）.
'''
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1) # 损失值和准确率
print(test_loss)
print('test_acc', test_acc)


'''
[7]
输入一张手写数字图片到网络中，看看它的识别效果
'''
(train_images, train_labels), (test_images, test_labels) = load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28 * 28))
res = network.predict(test_images) # 对整个测试集送入模型进行预测

for i in range(10000): # 对所有图片送入模型预测
    for j in range(res[i].shape[0]):
        if (res[i][j] == 1):
            if test_labels[i] != j:
                print('incorrect')
            print("the picture correct number is:", test_labels[i])
            print("the number for the picture is : ", j)
