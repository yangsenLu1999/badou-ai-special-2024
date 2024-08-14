from keras.layers import Input, Conv2D, BatchNormalization, ReLU, concatenate
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,Dense
from keras.models import Model
from modules.global_params import IMG_SHAPE, NUM_CLASS

def _basic_conv(x,filters, kernel_size, strides=1, padding="valid", name=None):
    '''基础卷积块: Conv2d+BN+ReLU'''
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(scale=False)(x)
    x = ReLU(name=name)(x)
    return x


def _inception_a(x, pool_features, name=None):
    '''inception a: 得到输入大小不变，通道数为224+pool_features的特征图'''

    # branch1: 经过64个1 * 1的卷积核
    branch1 = _basic_conv(x, filters=64, kernel_size=(1,1))

    # branch2: 先经过48个1*1的卷积核, 再经过64个5*5大小且填充为2的卷积核
    branch2 = _basic_conv(x, filters=48, kernel_size=(1,1))
    branch2 = _basic_conv(branch2, filters=64, kernel_size=(5,5), padding="same")

    # branch3: 先经过64个1*1的卷积核，再经过2次96个3*3大小且填充为1的卷积核
    branch3 = _basic_conv(x, filters=64, kernel_size=(1,1))
    branch3 = _basic_conv(branch3, filters=96, kernel_size=(3,3), padding="same")
    branch3 = _basic_conv(branch3, filters=96, kernel_size=(3, 3), padding="same")

    # branch4: 先经过avg_pool2d，其中池化核3*3，步长为1，填充为1;再经过有pool_features个的1*1卷积
    branch4 = AveragePooling2D(pool_size=(3,3),strides=(1,1), padding="same")(x)
    branch4 = _basic_conv(branch4, filters=pool_features, kernel_size=(1,1))

    return concatenate([branch1, branch2, branch3, branch4], axis=3, name=name)

def _inception_b(x, name=None):
    '''inception b: 得到输入大小减半，通道数为480的特征图'''

    # branch1: 经过384个3*3大小且步长2的卷积核
    branch1 = _basic_conv(x, filters=384, kernel_size=(3,3), strides=2) # 图片大小减半

    # branch2: 先经过64个1*1的卷积核，再经过96个3*3填充为1的卷积核，最后96个3*3大小步长2的卷积核
    branch2 = _basic_conv(x, filters=64, kernel_size=(1,1))
    branch2 = _basic_conv(branch2, filters=96, kernel_size=(3,3), padding="same")
    branch2 = _basic_conv(branch2, filters=96, kernel_size=(3,3), strides=2) # 图片大小减半

    # branch3: 经过max_pool2d，池化核大小3*3，步长为2
    branch3 = MaxPooling2D(pool_size=(3,3), strides=2)(x) # 图片大小减半

    return concatenate([branch1, branch2, branch3], axis=3, name=name)

def _inception_c(x, channels7x7, name=None):
    '''inception c: 得到输入大小不变，通道数为768的特征图'''

    # branch1: 经过192个1*1的卷积核
    branch1 = _basic_conv(x, filters=192, kernel_size=(1,1))

    # branch2: 先经过channels7x7个1*1的卷积核, 再用1*7和7*1卷积核代替7*7卷积核
    branch2 = _basic_conv(x, filters=channels7x7, kernel_size=(1,1))
    branch2 = _basic_conv(branch2, filters=channels7x7, kernel_size=(1,7), padding="same")
    branch2 = _basic_conv(branch2, filters=192, kernel_size=(7,1), padding="same")

    # branch3: 先经过channels7x7个1*1的卷积核, 再经过2次用7*1和1*7卷积核代替7*7卷积核
    branch3 = _basic_conv(x, filters=channels7x7, kernel_size=(1,1))
    branch3 = _basic_conv(branch3, filters=channels7x7, kernel_size=(7,1), padding="same")
    branch3 = _basic_conv(branch3, filters=channels7x7, kernel_size=(1, 7), padding="same")
    branch3 = _basic_conv(branch3, filters=channels7x7, kernel_size=(7,1), padding="same")
    branch3 = _basic_conv(branch3, filters=192, kernel_size=(1, 7), padding="same")

    # branch4: 先经过avg_pool2d，其中池化核3*3，步长为1，填充为1；再经过192个的1*1卷积
    branch4 = AveragePooling2D(pool_size=(3,3), strides=1, padding="same")(x)
    branch4 = _basic_conv(branch4, filters=192, kernel_size=(1,1))

    return concatenate([branch1, branch2, branch3, branch4], axis=3, name=name)


def _inception_d(x, name=None):
    '''inception d: 得到输入大小减半，通道数512的特征图'''

    # branch1: 先经过192个1*1的卷积核，再经过320个3*3大小步长为2的卷积核
    branch1 = _basic_conv(x, filters=192, kernel_size=(1,1))
    branch1 = _basic_conv(branch1, filters=320, kernel_size=(3,3), strides=2) # 图片大小减半

    # branch2: 先经过192个1*1的卷积核, 再192个用1*7和7*1代替7*7, 最后经过192个3*3大小步长为2的卷积核
    branch2 = _basic_conv(x, filters=192, kernel_size=(1,1))
    branch2 = _basic_conv(branch2, filters=192, kernel_size=(1,7),padding="same")
    branch2 = _basic_conv(branch2, filters=192, kernel_size=(7,1),padding="same")
    branch2 = _basic_conv(branch2, filters=192, kernel_size=(3,3), strides=2) # 图片大小减半

    # branch3: 经过max_pool2d，池化核大小3*3，步长为2
    branch3 = MaxPooling2D(pool_size=(3,3), strides=2)(x) # 图片大小减半

    return concatenate([branch1, branch2, branch3], axis=3, name=name)

def _inception_e(x, name=None):
    '''inception e: 得到输入大小不变，通道数为2048的特征图'''

    # branch1: 经过320个1*1的卷积核
    branch1 = _basic_conv(x, filters=320, kernel_size=(1,1))

    # branch2: 先经过384个1*1的卷积核；再分别经过384个1*3和3*1卷积核
    branch2 = _basic_conv(x, filters=384, kernel_size=(1,1))
    branch2_1 = _basic_conv(branch2, filters=384, kernel_size=(1,3), padding="same")
    branch2_2 = _basic_conv(branch2, filters=384, kernel_size=(3,1), padding="same")
    branch2 = concatenate([branch2_1, branch2_2], axis=3) # 合并branch2_1, branch2_2

    # branch3: 先经过448个1*1的卷积核，再经过384个3*3大小且填充为1的卷积核,再分别经过384个1*3和3*1卷积核
    branch3 = _basic_conv(x, filters=448, kernel_size=(1,1))
    branch3 = _basic_conv(branch3, filters=384, kernel_size=(3,3), padding="same")
    branch3_1 = _basic_conv(branch3, filters=384, kernel_size=(1, 3), padding="same")
    branch3_2 = _basic_conv(branch3, filters=384, kernel_size=(3, 1), padding="same")
    branch3 = concatenate([branch3_1, branch3_2], axis=3)  # 合并branch3_1, branch3_2

    # branch4: 先经过avg_pool2d，其中池化核3*3，步长为1，填充为1；再经过192个的1*1卷积核
    branch4 = AveragePooling2D(pool_size=(3,3), strides=1, padding="same")(x)
    branch4 = _basic_conv(branch4, filters=192, kernel_size=(1,1))

    return concatenate([branch1, branch2, branch3, branch4], axis=3, name=name)

def InceptionV3(input_shape=IMG_SHAPE, output_dim=NUM_CLASS):
    # 创建inputs用于存放输入数据
    inputs = Input(shape=input_shape)

    # stage 1
    x = _basic_conv(inputs, filters=32, kernel_size=(3,3), strides=2)
    x = _basic_conv(x, filters=32, kernel_size=(3, 3))
    x = _basic_conv(x, filters=64, kernel_size=(3, 3), padding="same")
    x = MaxPooling2D(pool_size=(3,3), strides=2)(x)

    # stage 2
    x = _basic_conv(x,filters=80, kernel_size=(1,1))
    x = _basic_conv(x, filters=192, kernel_size=(3,3))
    x = MaxPooling2D(pool_size=(3,3), strides=2)(x)

    # stage 3: 3个inception_a
    x = _inception_a(x, pool_features=32)
    x = _inception_a(x, pool_features=64)
    x = _inception_a(x, pool_features=64)

    # stage 4: 1个inception_b + 4个inception_c
    x = _inception_b(x)
    x = _inception_c(x, channels7x7=128)
    x = _inception_c(x, channels7x7=160)
    x = _inception_c(x, channels7x7=160)
    x = _inception_c(x, channels7x7=192)

    # stage 5: 1个inception_d + 2个inception_e
    x = _inception_d(x)
    x = _inception_e(x)
    x = _inception_e(x)

    # stage 6
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=output_dim, activation="softmax")(x)

    model = Model(inputs, x, name="inception_v3")

    return model

