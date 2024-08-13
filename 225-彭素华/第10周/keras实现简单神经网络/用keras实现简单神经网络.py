from tensorflow.keras.datasets import mnist

''' [1] '''
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()  # 加载训练和测试数据。
print('train_images.shape = ',train_images.shape)
print('train_labels = ',train_labels)
print('test_images.shape = ',test_images.shape)
print('test_labels = ', test_labels)

''' [2] '''
digit = test_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()

''' [3] '''
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential() # 表示把每一个数据处理层都串联起来。
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

''' [4] '''
'''

在把数据输入网络模型之前，数据做归一化处理：
1.reshape(60000, 28*28）:train_images数组原来含有60000个元素，每个元素是一个28行，28列的二维数组，
现在把每个二维数组转变为一个含有28*28个元素的一维数组.
2.由于数字图案是一个灰度图，图片中每个像素点值的大小范围在0到255之间.
3.train_images.astype(“float32”)/255 把每个像素点的值从范围0-255转变为范围在0-1之间的浮点值。

'''
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255  ###

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255   ###


'''

one-hot

'''
from tensorflow.keras.utils import to_categorical

print("before change:",test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ",test_labels[0])


''' [5] '''
''' 开始训练 '''
network.fit(train_images,train_labels,epochs=5,batch_size=128)


''' [6] '''
''' 测试 '''
test_loss,test_acc = network.evaluate(test_images,test_labels,verbose=1)
print(test_loss)
print('test_acc',test_acc)

''' [7] '''
''' 预测 '''
''' 输入一张手写数字图片到网络中，看看它的识别效果。 '''
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
digit = test_images[2]
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()

test_images = test_images.reshape((10000,28*28))
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):      ## ???
        print("the number for the picture is : ",i)
        break