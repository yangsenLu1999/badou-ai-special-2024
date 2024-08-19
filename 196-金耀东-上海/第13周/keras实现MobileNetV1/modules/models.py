from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Softmax
from keras.layers import GlobalAveragePooling2D, Reshape, Dropout, DepthwiseConv2D
from keras.models import Model
from modules.global_params import IMG_SHAPE, NUM_CLASS

def _basic_conv2d(x, filters, kernel_size, strides=1, padding="same"):
    '''conv2d + bn + relu6'''
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization()(x)
    return ReLU(max_value=6)(x)

def _separable_conv2d(x, filters, kernel_size, strides=1, padding="same", depth_multiplier=1):
    '''conv2d_dw + bn + relu6 + conv2d_pw'''
    x = DepthwiseConv2D(kernel_size, strides, padding, depth_multiplier, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    return _basic_conv2d(x, filters, kernel_size=(1,1))


def MobileNetV1(input_shape=IMG_SHAPE, output_dim=NUM_CLASS, dropout_rate=1e-3):
    inputs = Input(shape=input_shape)

    # 普通卷积层
    x = _basic_conv2d(inputs, filters=32, kernel_size=(3,3), strides=2)

    # 深度可分离卷积层
    x = _separable_conv2d(x, filters=64, kernel_size=(3, 3))
    x = _separable_conv2d(x, filters=128, kernel_size=(3, 3), strides=2)
    x = _separable_conv2d(x, filters=128, kernel_size=(3, 3))
    x = _separable_conv2d(x, filters=256, kernel_size=(3, 3), strides=2)
    x = _separable_conv2d(x, filters=256, kernel_size=(3, 3))
    x = _separable_conv2d(x, filters=512, kernel_size=(3, 3), strides=2)
    for i in range(5):
        x = _separable_conv2d(x, filters=512, kernel_size=(3,3))
    x = _separable_conv2d(x, filters=1024, kernel_size=(3,3), strides=2)
    x = _separable_conv2d(x, filters=1024, kernel_size=(3, 3), strides=2)

    # 分类层
    x = GlobalAveragePooling2D()(x)
    x = Reshape(target_shape=(1,1,-1))(x)
    x = Dropout(rate=dropout_rate)(x)
    x = Conv2D(filters=output_dim, kernel_size=(1,1))(x)
    x = Softmax()(x)
    x = Reshape(target_shape=(output_dim,))(x)

    return Model(inputs, x, name="mobilenet_v1")