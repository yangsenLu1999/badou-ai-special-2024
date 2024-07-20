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

'''
- `input_tensor`：input_tensor 是 identity_block 的输入 不是原始的输入
- `kernel_size`：卷积核的大小。
- `filters`：卷积核的数量，即输出的通道数。
- `stage`：表示当前是在网络的哪个阶段。
- `block`：表示当前是在阶段中的哪个块。
'''
#
def identity_block(input_tensor, kernel_size, filters, stage, block):

    """
    将变量 `filters` 中的值分别赋值给变量 `filters1`、`filters2` 和 `filters3`。
    这样的赋值操作常用于将一个数组或列表中的元素分别提取出来，以便在后续的代码中进行单独处理或使用。
    """
    filters1, filters2, filters3 = filters
    '''
    `'res' + str(stage) + block + '_branch'` 的意思是将字符串 `'res'`、变量 `stage` 转换为字符串后的值、变量 `block` 的值以及字符串 `'_branch'` 连接起来，形成一个新的字符串。
    用于给深度学习模型中的不同部分或层命名，以便在代码中更清晰地引用和操作它们。
    '''
    conv_name_base = 'res' + str(stage) + block + '_branch'
    '''
    `'bn' + str(stage) + block + '_branch'` 的意思是将字符串 `'bn'`、变量 `stage` 转换为字符串后的值、变量 `block` 的值以及字符串 `'_branch'` 连接起来，形成一个新的字符串。
    '''
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    '''
    定义一个卷积层、一个批量归一化层和一个激活函数，并将它们应用到输入张量 `input_tensor` 上。
    1. 使用 `Conv2D` 函数创建一个卷积层，该卷积层的卷积核大小为 `(1, 1)`，输出通道数为 `filters1`，并将其命名为 `conv_name_base + '2a'`。
    2. 使用 `BatchNormalization` 函数创建一个批量归一化层，并将其命名为 `bn_name_base + '2a'`。
    3. 使用 `Activation` 函数创建一个激活函数，该激活函数的类型为 `'relu'`，并将其应用到卷积层的输出上。
    这样，就完成了对输入张量的卷积、批量归一化和激活操作。
    '''
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


'''
定义一个卷积块（conv_block）
- `input_tensor`：输入张量，通常是前一层的输出。
- `kernel_size`：卷积核的大小。
- `filters`：卷积核的数量，即输出的通道数。
- `stage`：表示当前是在网络的哪个阶段。
- `block`：表示当前是在阶段中的哪个块。
- `strides`：卷积的步长。
在函数内部，可能会进行一系列的卷积操作、批量归一化操作和激活函数操作，以实现对输入张量的特征提取和转换。
卷积块的作用是通过卷积操作对输入特征进行降维或特征提取，同时通过批量归一化和激活函数操作增加模型的非线性表达能力。
'''
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters  # 将滤波器数量分解为三个部分

    conv_name_base = 'res' + str(stage) + block + '_branch'  # 创建一个卷积名称的基础字符串
    bn_name_base = 'bn' + str(stage) + block + '_branch'  # 创建一个批量归一化名称的基础字符串

    # 分支1

    x = Conv2D(filters1, (1, 1), strides=strides,  name=conv_name_base + '2a')(input_tensor)  # 使用 1x1 卷积核进行降维，并设置步长。
    x = BatchNormalization(name=bn_name_base + '2a')(x)  # 进行批量归一化
    x = Activation('relu')(x)  # 使用 ReLU 激活函数

    x = Conv2D(filters2, kernel_size, padding='same',name=conv_name_base + '2b')(x)  # 使用指定的卷积核大小进行卷积，并使用相同的填充。
    x = BatchNormalization(name=bn_name_base + '2b')(x)  # 进行批量归一化
    x = Activation('relu')(x)  # 使用 ReLU 激活函数  # 进行批量归一化

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)  # 使用 1x1 卷积核进行升维
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 分支2
    shortcut = Conv2D(filters3, (1, 1), strides=strides,name=conv_name_base + '1')(input_tensor)  # 使用 1x1 卷积核进行降维，并设置步长。
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)  # 进行批量归一化

    # 分支1 + 分支2  （add: 相同矩阵才能相加  concat: 拼接 h,w,c  在某个维度拼接其余两个维度需要相同才可以拼接）
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)  # 使用 ReLU 激活函数
    return x


def ResNet50(input_shape=[224,224,3],classes=1000):

    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(img_input)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Conv2d Block 并联(分支1+分支2)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    # Identity Block 串联
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

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')

    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model

if __name__ == '__main__':
    # 声明网络结构
    model = ResNet50()  # 构建 ResNet50 网络模型
    '''
    使用 `model.summary()` 来获取模型的结构和参数信息的摘要。
        1. 模型的层结构：显示模型中包含的各个层，如卷积层、全连接层等。
        2. 每层的参数数量：告诉你每个层的参数数量，例如卷积核的数量、权重的数量等。
        3. 模型的总参数数量：给出整个模型的参数数量。
        4. 输入和输出的形状：描述模型的输入和输出张量的形状。
    通过查看模型的摘要，你可以了解模型的架构、参数数量以及输入输出的要求。这对于理解模型的复杂性、检查模型的配置是否正确以及进行模型的调试和优化都非常有帮助。    
    '''
    model.summary()  # 使用 model.summary() 来获取模型的结构和参数信息的摘要
    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    '''
    将图像 `img` 转换为一个 NumPy 数组 `x`:
    `image.img_to_array` 是 `keras.preprocessing.image` 模块中的一个函数，它的作用是将图像数据转换为适合深度学习模型处理的数组形式。
    转换后的数组 `x` 通常具有以下特点：
    1. 数组的维度：`x` 是一个多维数组，其维度通常与图像的尺寸和颜色通道数相关。例如，如果图像是彩色的，可能有三个颜色通道（红、绿、蓝），那么 `x` 的维度可能是 `(height, width, 3)`。
    2. 数值范围：图像的像素值通常在 0 到 255 之间（对于 8 位图像），转换后的数组 `x` 的数值范围也在 0 到 255 之间。
    3. 数据类型：`x` 的数据类型通常是 `numpy.uint8` 或类似的整数类型。
    在后续的代码中，你可以对转换后的数组 `x` 进行各种操作，例如数据预处理、特征提取、模型训练等。
    '''
    x = image.img_to_array(img)
    '''
    使用 NumPy 库的 `expand_dims` 函数在数组 `x` 的指定轴（axis）上添加一个新的维度。
    在这个例子中，`axis=0` 表示在数组的第一个维度（通常是行维度）上添加一个新的维度。
    执行这行代码后，数组 `x` 的维度将增加 1，例如，如果 `x` 原本是一个二维数组，那么现在它将变成一个三维数组。
    添加新的维度通常是为了满足某些操作或算法的要求，例如在深度学习中，模型可能需要输入具有特定维度的张量。通过使用 `expand_dims` 函数，我们可以方便地在数组上添加或删除维度，以适应不同的计算需求。
    '''
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
