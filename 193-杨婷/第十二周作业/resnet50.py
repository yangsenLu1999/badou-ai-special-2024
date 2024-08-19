# -------------------------------------------------------------#
#   ResNet50的网络部分
# -------------------------------------------------------------#
from __future__ import print_function  # 使得Python 2.x的代码能够使用Python 3.x风格的print函数
import numpy as np
from keras import layers

from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model

from keras.preprocessing import image

from keras.applications.imagenet_utils import decode_predictions  # 用于将模型预测的原始输出（通常是类别的索引或概率）转换为可读的标签或类别名称
from keras.applications.imagenet_utils import preprocess_input  # 用于对输入图像进行预处理


def conv_block(input_tensor, kernal_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters  # 解包列表
    conv_name_base = 'res' + str(stage) + block + '_branch'  # 输出格式例：res3a_branch
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base+'2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernal_size, padding='same',  # 默认padding为valid
               name=conv_name_base+'2b')(x)
    x = BatchNormalization(name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base+'2c')(x)  # strides默认值(1,1)
    x = BatchNormalization(name=bn_name_base+'2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base+'1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base+'1')(shortcut)

    x = layers.add([x, shortcut])  # 输入张量列表
    x = Activation('relu')(x)
    return x


def identity_block(input_tensor, kernal_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base+'2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernal_size, padding='same',
               name=conv_name_base+'2b')(x)
    x = BatchNormalization(name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base+'2c')(x)
    x = BatchNormalization(name=bn_name_base+'2c')(x)

    x = layers.add([x, input_tensor])  # 可以考虑变成concat
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=[224, 224, 3], classes=1000):
    img_input = Input(shape=input_shape)  # 用来实例化一个输入层
    # stage0
    x = ZeroPadding2D((3, 3))(img_input)  # 每个空间维度（高度和宽度）的两侧各添加3个零填充
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # stage1
    x = conv_block(x, 3, [64, 64, 256], stage=1, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=1, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=1, block='c')

    # stage2
    x = conv_block(x, 3, [128, 128, 512], stage=2, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=2, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=2, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=2, block='d')

    # stage3
    x = conv_block(x, 3, [256, 256, 1024], stage=3, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='f')

    # stage4
    x = conv_block(x, 3, [512, 512, 2048], stage=4, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=4, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=4, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc_1000')(x)

    # 以下两句可以放在main函数model_resnet50后面写
    model = Model(img_input, x, name='resnet_50')
    model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5')  # 加载训练好的权重

    return model


if __name__ == '__main__':
    model_resnet50 = ResNet50()
    model_resnet50.summary()  # 打印出模型的概述信息,包括每一层的名称、类型、输出形状以及参数数量等
    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))  # 读入图片
    img = image.img_to_array(img)  # 将图片转换为NumPy数组
    img = np.expand_dims(img, axis=0)  # 现在x的形状是(1, 高度, 宽度, 通道数)
    img = preprocess_input(img)  # 一些预处理，如归一化

    print('Input image shape:', img.shape)
    preds = model_resnet50.predict(img)
    print('Predicted:', decode_predictions(preds))  # 默认打印top5




