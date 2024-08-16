'''
keras写alexnet网络结构，增加BN
'''
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from keras.models import Sequential


def Alexnet(input_shape=(224, 224, 3), output_shape=2):  # keras所写第一层均需要inputshape， pytorch,vgg需要输入input而不是shape
    model = Sequential()
    # 224*224的3通道图像，处理后变为（224-11）/4+1 = 54.25向下取整，变为54*54的48通道图像，卷积滑动过程中最后剩余的就丢弃不管了
    model.add(Conv2D(filters=48, kernel_size=[11, 11], strides=[4, 4], padding='valid',
                     input_shape=input_shape, activation='relu'))  # keras所写的网络第一层都要输入inputshape,VGG需要input而不只是形状
    model.add(BatchNormalization())  # 标准化（x-mu)/sigma使得进入激活函数后的反应敏感，一般应放在relu之前，但这里影响不大
    # 最大池化不改变通道数，只改图像大小，减少数据，保留最明显特征
    model.add(MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding='valid'))  # 输出的shape为(27,27,48)
    '''以下为卷积、正则、池化多层循环，及CONV + BN + POOL'''
    model.add(Conv2D(filters=128, kernel_size=[5, 5], strides=[1, 1], padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding='valid'))  # 输出的shape为(13,13,128)

    model.add(Conv2D(filters=192, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu'))
    model.add(Conv2D(filters=192, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu'))  # 输出的shape为(13,13,192)
    model.add(MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding='valid'))  # 输出的shape为(6,6,128)
    '''以下进入全连接层'''
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))  # 0.25的比例进行神经元随机失活，防止全连接层的过拟合，卷积层主要是提取特征，不易发生过拟合，因此卷积部分没有dropout

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_shape, activation='softmax'))  # 最后一层分类用softmax求概率
    return model
