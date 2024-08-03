import keras.layers as layers
from keras import backend
from keras.layers import Conv2D, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, TimeDistributed, \
    AveragePooling2D, Input
from keras.models import Model
import numpy as np


def identity_block(inputs, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = "res" + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(inputs)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, inputs])
    x = Activation('relu')(x)
    return x


def identity_block_td(inputs, kernel_size, filters, stage, block, trainable=True):
    filters1, filters2, filters3 = filters
    if backend.image_data_format == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(filters1, (1, 1), trainable=trainable, kernel_initializer='normal'),
                        name=conv_name_base + '2a')(inputs)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(filters2, kernel_size, trainable=trainable, kernel_initializer='normal', padding='same'),
                        name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(filters3, (1, 1), trainable=trainable, kernel_initializer='normal'),
                        name=conv_name_base + '2c')(x)
    x = TimeDistributed(BatchNormalization(axis=-bn_axis), name=bn_name_base + '2c')(x)

    x = layers.add([x, inputs])
    return Activation('relu')(x)


def conv_block(inputs, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = "res" + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(inputs)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(inputs)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def conv_block_td(inputs, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):
    filters1, filters2, filters3 = filters
    if backend.image_data_format == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = "res" + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(filters1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
                        input_shape=input_shape, name=conv_name_base + '2a')(inputs)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(filters2, kernel_size, padding='same', trainable=trainable, kernel_initializer='normal'),
                        name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(filters3, (1, 1), kernel_initializer='normal'), trainable=trainable,
                        name=conv_name_base + '2c')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(
        Conv2D(filters3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
        name=conv_name_base + '1')(inputs)
    shortcut = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    return Activation('relu')(x)


def create_model(inputs):
    img_input = inputs
    x = ZeroPadding2D((3, 3))(img_input)

    # 230*230*3--->112*112*64
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    return x


def classifier_layers(x, input_shape, training=False):
    x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2),
                      trainable=training)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=training)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=training)

    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)
    return x

