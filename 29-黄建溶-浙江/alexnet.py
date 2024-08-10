from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam

def AlexNet(input_shape=(224,224,3),output_shape=2):
    model=Sequential()
#给模型添加卷积层，并设置参数
    model.add(
       Conv2D(filters=48,
              kernel_size=(11,11),
              strides=4,
              padding='valid',
              input_shape=input_shape,
              activation='relu'
        )
    )
    model.add(BatchNormalization())   #优化项，提高训练速度
#给模型添加池化，设置池化核大小3x3,步长2X2
    model.add(
       MaxPooling2D(
              pool_size=(3,3),
              strides=(2,2),
              padding='valid'
        )
    )
#添加卷积层，卷积核大小5X5，步长1X1
    model.add(
       Conv2D(filters=128,
              kernel_size=(5,5),
              strides=(1,1),
              padding='same',
              activation='relu'
        )
    )
    model.add(BatchNormalization())
#添加池化层
    model.add(
       MaxPooling2D(
              pool_size=(3,3),
              strides=(2,2),
              padding="valid"
        )
    )
#添加卷积层
    model.add(
       Conv2D(
              filters=192,
              kernel_size=(3,3),
              strides=(1,1),
              padding="same",
              activation="relu"
        )
    )
#添加卷积层
    model.add(
       Conv2D(
              filters=192,
              kernel_size=(3,3),
              strides=(1,1),
              padding="same",
              activation="relu"
       )
    )
#添加卷积层
    model.add(
       Conv2D(
               filters=128,
               kernel_size=(3,3),
               strides=(1,1),
               padding="same",
               activation='relu'
       )
    )
#添加池化层
    model.add(
       MaxPooling2D(
               pool_size=(3,3),
               strides=(2,2),
               padding="valid"
       )
    )

#添加两个全连接层，首先进行数据维度转换，再设置全连接层的节点数、激活函数等
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.25))

#对输出结果用softmax函数分类
    model.add(Dense(output_shape,activation='softmax'))
    return model