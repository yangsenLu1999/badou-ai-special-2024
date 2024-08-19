#-------------------------------------------------------------#
#   ResNet50的网络部分
#-------------------------------------------------------------#
from __future__ import print_function

import numpy as np
from keras import layers

from keras.layers import Input
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import Activation,BatchNormalization,Flatten
from keras.models import Model

from keras.preprocessing import image
import keras.backend as K
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


def identity_block(input_tensor, kernel_size, filters, stage, block):
# filters: 一个包含三个元素的列表，分别代表三个卷积层的过滤器数量。
# stage: 当前块的阶段编号，通常用于区分层次结构中的不同阶段。
# block: 当前块的名称，通常为字母（如 'A', 'B'），用于区分同一阶段的不同块

    filters1, filters2, filters3 = filters # 从 filters 列表中解包出三个过滤器的数量。

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'   # 为每个卷积层和批归一化层创建基础名称，以便于后续命名
# stage: 当前块的阶段编号，通常用于标识网络的不同层次（例如，ResNet 的每个阶段）。
# block: 当前块的标识符（例如，'A', 'B'）。通常用一个字母表示同一阶段中的不同残差块。
# 例如，如果 stage 是 2，block 是 'A'，那么：
# conv_name_base 将变为 'res2A_branch'
# bn_name_base 将变为 'bn2A_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
# 创建一个 1x1 卷积层，输出通道数为 filters1，将其应用于输入张量 input_tensor。这个卷积一般用于调整维度
    x = BatchNormalization(name=bn_name_base + '2a')(x)
# name: 为这一层指定一个名称，这样可以帮助识别此层。这里名称的构造是 bn_name_base + '2a'，
# 最终形成的名称是类似 'bn2A_branch2a' 的格式，表示这是在阶段 2 和块 A 的第二分支上的批量归一化操作。
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
# padding='same' 表示输出大小与输入大小相同
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
# 实现了残差连接（skip connection），将输入张量 input_tensor 和卷积输出 x 逐元素相加。
# 相加的结果是一个新的张量，它的形状与 x 和 input_tensor 相同
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
    '''通过 1x1 卷积来调整输入张量 input_tensor 的通道数，使其与后续操作的输出张量 x 的形状一致
filters3: 目标输出的通道数，通常与模块中的其他层（如卷积层的输出）相匹配。
(1, 1): 卷积核的大小，这里使用 1x1 的卷积核主要用于改变通道数，而不改变特征图的空间维度（高度和宽度）。
strides=strides: 步幅，通常在网络架构中的某些时刻可能需要调整特征图的空间维度（如下采样），这取决于整体网络设计。
name=conv_name_base + '1': 给这一层命名，便于模型管理和调试。这里的名称构建是通过 conv_name_base + '1'，表示该卷积层的名称以 '1' 结尾，区分其他卷积层。
input_tensor: 输入张量，经过 1x1 卷积后会生成新的 shortcut 张量'''
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
    '''目的: 对通过卷积层处理后的 shortcut 张量应用批量归一化，以加快训练速度并提高模型稳定性。
name=bn_name_base + '1': 为批量归一化层命名，使用 bn_name_base + '1'，与对应的卷积层保持一致。
shortcut: 这一张量是通过 1x1 卷积操作后生成的结果。通过批量归一化处理，有助于减少内部协变量偏移，加速训练，并可能提高模型性能'''

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=[224,224,3],classes=1000):

    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(img_input) # ZeroPadding2D((3, 3)): 这里表示在每个边缘（上、下、左、右）各添加 3 个像素的零填充。

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    '''64: 输出的特征图通道数（即卷积后形成的过滤器数量），这里设置为 64。意味着该层会生成 64 个不同的特征图。
    name='conv1': 给这一层卷积命名为 'conv1'，便于后续的网络层管理和调试。'''
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    # 3: 卷积块的卷积层数，通常这里的 3 表示使用 3 个卷积层进行特征提取。
    # [64, 64, 256]: 这是一个列表，表示每个卷积层的输出通道数量。
    # stage = 2: 表示这是网络的第二个阶段
    # block='a': 表示这是第二个阶段中的第一个卷积块（Block A）

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

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)   # CV领域一般用最大池化

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')

    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")  # 加载预训练的权重文件

    return model

if __name__ == '__main__':
    model = ResNet50()
    model.summary()  # 打印模型的摘要信息，包括每一层的名称、输出形状、参数数量等，这有助于理解模型的结构
    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) # 将加载的图像从 PIL 格式转换为 NumPy 数组格式。这样可以将图像传入 Keras 模型进行预测
    x = np.expand_dims(x, axis=0)
    # 在数组 x 的最外层增加一个维度，这样可以将其转换为批大小（batch size）的格式。因为在模型中，每次输入的都是一个批处理，虽然这里我们只处理一张图像
    x = preprocess_input(x)
    # 对输入图像数据进行预处理，符合模型在训练时的数据标准化要求。preprocess_input 通常包括像素值缩放等步骤

    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
    # 将概率转换为人类可读的标签，通常会返回类名和相应的概率。decode_predictions 函数会根据模型的分类映射返回可读的预测结果
