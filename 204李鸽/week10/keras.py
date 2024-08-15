from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
'''tensorflow和keras有版本依赖性
也可以直接import keras
将训练数据和检测数据加载到内存中(第一次运行需要下载数据，会比较慢):
train_images是用于训练系统的手写数字图片;
train_labels是用于标注图片的信息;
test_images是用于检测系统训练效果的图片；
test_labels是test_images图片对应的数字标签。'''
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
'''
使用tensorflow.Keras搭建一个有效识别图案的神经网络，
1.layers:表示神经网络中的一个数据处理层。(dense:全连接层)
2.models.Sequential():表示把每一个数据处理层串联起来.
3.layers.Dense(…):构造一个数据处理层。
4.input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系.
'''
from tensorflow.keras import models   # 搭建模型，一开始是个空的
from tensorflow.keras import layers   # 在模型内加层

network = models.Sequential()     # 创建一个串行序列的模型，赋值给变量
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
# 连接方式是全连接，输入图像的尺寸是28*28，即输入层的节点个数是28*28个，定义隐藏层的节点个数是512个
network.add(layers.Dense(10, activation='softmax'))    # softmax也符合激活函数，他是把输出层输出的读不懂的数转换成概率
# 连接方式是全连接，定义输出层的节点个数是512个

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])
# 模型结构做好了，编译成计算机语言，保存一些优化项目，损失函数类别，这里用的是交叉熵，判断正确率的方式
# 网络搭好了，下面处理数据

train_images = train_images.reshape((60000, 28*28))   # 每张图片的二维数组转成一维
train_images = train_images.astype('float32') / 255   # 优化项，归一化

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# 处理输出和标签的对应关系
'''
把图片对应的标记也做一个更改：
目前所有图片的数字图案对应的是0到9。
例如test_images[0]对应的是数字7的手写图案，那么其对应的标记test_labels[0]的值就是7。
我们需要把数值7变成一个含有10个元素的数组，然后在第8个元素设置为1，其他元素设置为0。
例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0] ---one hot
哪个输出的概率值最大，哪个变为1
'''
from tensorflow.keras.utils import to_categorical    # 调用一个转换函数
print("before change:" ,test_labels[0])              # 7
train_labels = to_categorical(train_labels)          # train_labels和test_labels都转换成one hot
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])              # [0,0,0,0,0,0,0,1,0,0]

# 开始训练  .fit一行即可
'''
把数据输入网络进行训练：
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算的循环是五次，每一代都是把60000张图片全做完，看打印结果，结果五代的正确率逐步提高
'''
network.fit(train_images, train_labels, epochs=5, batch_size=128)

'''
测试  .evaluate数据输入，检验网络学习后的图片识别效果.
识别效果与硬件有关（CPU/GPU）.
'''
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)   # verbose=1是否要打印，是
print(test_loss)                # 打印验证集误差
print('test_acc', test_acc)     # 打印验证集准确率

# 准确率OK，投产推理  .predict去了
'''
输入一张手写数字图片到网络中，看看它的识别效果
'''
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()   # 没做现场数据集
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))    # 推理10000张图
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break















