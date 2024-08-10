from __future__ import print_function


from keras import layers

from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model


import keras.backend as K
from keras.utils.data_utils import get_file



def identity_block(input_tensor, kernel_size, filters, stage, block):
    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    left = Conv2D(filter1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    left = BatchNormalization(name=bn_name_base + '2a')(left)
    left = Activation('relu')(left)

    left = Conv2D(filter2, kernel_size, padding='same', name=conv_name_base + '2b')(left)
    left = BatchNormalization(name=bn_name_base + '2b')(left)
    left = Activation('relu')(left)

    left = Conv2D(filter3, (1, 1), name=conv_name_base + '2c')(left)
    left = BatchNormalization(name=bn_name_base + '2c')(left)

    left = layers.add([left, input_tensor])
    left = Activation('relu')(left)
    return left

def conv_block(input_tensor, kernel_size, filters, stack, block, strides=(2, 2)):
    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stack) + block + '_branch'
    bn_name_base = 'bn' + str(stack) + block + '_branch'

    x = Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    # 右边的
    right = Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    right = BatchNormalization(name=bn_name_base + '2')(right)
    # 作加法
    x = layers.add([x, right])
    x = Activation('relu')(x)
    return x


def my_resnet_50(input_shape=[224, 224, 3], classes=1000):
    input_image = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(input_image)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], 2, 'a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], 2, 'b')
    x = identity_block(x, 3, [64, 64, 256], 2, 'c')

    x = conv_block(x, 3, [128, 128, 512], 3, 'a')
    x = identity_block(x, 3, [128, 128, 512], 3, 'b')
    x = identity_block(x, 3, [128, 128, 512], 3, 'c')
    x = identity_block(x, 3, [128, 128, 512], 3, 'd')

    x = conv_block(x, 3, [256, 256, 1024], 4, 'a')
    x = identity_block(x, 3, [256, 256, 1024], 4, 'b')
    x = identity_block(x, 3, [256, 256, 1024], 4, 'c')
    x = identity_block(x, 3, [256, 256, 1024], 4, 'd')
    x = identity_block(x, 3, [256, 256, 1024], 4, 'e')
    x = identity_block(x, 3, [256, 256, 1024], 4, 'f')

    x = conv_block(x, 3, [512, 512, 2048], 5, 'a')
    x = identity_block(x, 3, [512, 512, 2048], 5, 'b')
    x = identity_block(x, 3, [512, 512, 2048], 5, 'c')

    x = AveragePooling2D(pool_size=(7, 7), name='avg_pool')(x)
    x = Flatten()(x)

    x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(input_image, x, name='resnet50')
    model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    return model
