from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import layers, optimizers

from keras.layers import Input, Lambda
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import Activation,BatchNormalization,Flatten
from keras.models import Model

from keras.preprocessing import image
import keras.backend as K
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=[32,32,3],classes=10):

    img_input = Input(shape=input_shape)
    # x = ZeroPadding2D((3, 3))(img_input)
    x = img_input
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(x)  # batchsize,16,16,channel
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)  # batchsize,8,8,channel

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1)) # batchsize,8,8,channel
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b') # batchsize,8,8,channel
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c') # batchsize,8,8,channel

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a') # batchsize,4,4,channel
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b') # batchsize,4,4,channel
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c') # batchsize,4,4,channel
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d') # batchsize,4,4,channel

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a') # batchsize,2,2,channel
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b') # batchsize,2,2,channel
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c') # batchsize,2,2,channel
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d') # batchsize,2,2,channel
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e') # batchsize,2,2,channel
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f') # batchsize,2,2,channel

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')  # batchsize,1,1,channel
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b') # batchsize,1,1,channel
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c') # batchsize,1,1,channel

    x = AveragePooling2D((1, 1), name='avg_pool')(x)  # batchsize,4,4,channel

    x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid', name='conv_fullSize')(x)  # batchsize,1,1,channel
    x = Conv2D(classes, (1, 1), strides=(1, 1), padding='valid', name='conv_1x1')(x)   # batchsize,1,1,classes
    x = Lambda(lambda x: tf.squeeze(x, axis=[1, 2]))(x)  # batchsize,classes

    # x = Flatten()(x)
    # x = Dense(classes, activation=None, name='fc1000')(x)
    # x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')

    return model


# 加载cifar-10数据集
def dataLoad():
    (train_data, train_label), (test_data, test_label) = tf.keras.datasets.cifar10.load_data()
    # 数据集进行归一化
    train_data = train_data / 255
    test_data = test_data / 255
    # # 将标签数据集从数组类型array修改成整形类型int
    # train_label.astype(np.int)
    # test_label.astype(np.int)
    # train_data = tf.constant(train_data, dtype=tf.float64)
    # train_label = tf.constant(train_label, dtype=tf.int32)
    # test_data = tf.constant(test_data, dtype=tf.float64)
    # test_label = tf.constant(test_label, dtype=tf.int32)
    return train_data, train_label, test_data, test_label

def lossFunc(y, y_predict):
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # automaticly transfer index to one-hot, and it has already had softmax
    loss = loss_func(y, y_predict)
    return loss


if __name__ == '__main__':
    model = ResNet50()
    model.summary()
    train_data, train_label, test_data, test_label = dataLoad()
    batch_size = 128
    model.compile(loss=lossFunc,
                  optimizer=optimizers.Adam(lr=1e-3),
                  metrics=['accuracy'])
    model.fit(train_data, train_label, epochs=100, batch_size=batch_size)

    model.save_weights('last_Resnetmodel.h5')
    model.load_weights("last_Resnetmodel.h5")
    model.evaluate(test_data, test_label)

