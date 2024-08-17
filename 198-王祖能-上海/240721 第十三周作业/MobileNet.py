'''
模型轻量化：采用深度可分离‘卷积’， depthwise separable convolution = depthwise separable filters + point separable filters
'''
import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Input, Conv2D, BatchNormalization, Activation, AveragePooling2D, MaxPooling2D, Dense
from keras.layers import DepthwiseConv2D, Dropout, Reshape, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras_applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras import backend as K


def relu6(x):  # 调整激活函数敏感范围，越小越能提升收敛速度，精确度会有所下降
    return K.relu(x, max_value=6)


def _conv_block(inputs, filters, size=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel_size=size, strides=strides, padding='same', use_bias=False, name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation(relu6, name='conv1_ac')(x)
    return x


def _depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, size=(3, 3), strides=(1, 1), padding='same', block_id=1):
    x = DepthwiseConv2D(size, strides, padding, depth_multiplier=depth_multiplier,
                        use_bias=False, name='conv_dw_%d' % block_id)(inputs)
    '''
    DepthwiseConv2D同一个卷积核在不同的channel上卷积输出，所以输出维度始终是与输入维度一样，不用指定filter大小，都一样的。(如果使用了depth_multiplier参数，参数数量乘以depth_multiplier)
    '''
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_ac' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1), strides=(1, 1), padding=padding, use_bias=False, name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_pw_%d_ac' % block_id)(x)
    return x


def MobileNet(input_shape=(224, 224, 3), depth_multiplier=1, dropout=1e-3, class_num=1000):
    img_input = Input(shape=input_shape)
    # ----------conv1 (224, 224, 3) ->(112, 112, 32)---------- #
    x = _conv_block(img_input, 32, (3, 3), strides=(2, 2))  # (224+2-3)/2+1=112.5 -> 112
    # ----------block1 (112, 12, 32) ->(112, 112, 64)---------- #
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)  # 根据pointwise通道数64确定
    # ----------block2 (112, 112, 64) ->(56, 56, 128)---------- #
    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)  # (112+2-3)/2+1=56.5 -> 56
    # ----------block3 (56, 56, 128) ->(56, 56, 128)---------- #
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)
    # ----------block4 (56, 56, 128) ->(28, 28, 256)---------- #
    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)  # (56+2-3)/2+1=28.5 -> 28
    # ----------block5 (28, 28, 256) ->(28, 28, 256)---------- #
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)
    # ----------block6 (28, 28, 256) ->(14, 14, 512)---------- #
    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)  # (28+2-3)/2+1=14.5 -> 14
    # ----------block7~11 (14, 14, 512) ->(14, 14, 512)---------- #
    for i in range(5):
        x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=(i+7))
    # ----------block12~13 (14, 14, 512) ->(7, 7, 1024)---------- #
    x = _depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)  # (14+2-3)/2+1=7.5 -> 7
    x = _depthwise_conv_block(x, 1024, depth_multiplier, strides=(1, 1), block_id=13)
    # ----------block14 (7, 7, 1024) ->(1, 1, 1024)---------- #
    x = GlobalAveragePooling2D(name='average_pool')(x)  # flatten + pooling
    x = Reshape(target_shape=(1, 1, 1024), name='reshape1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(class_num, (1, 1), padding='same', name='conv_prediction1')(x)  # inception用DENSE，这里1*1卷积代替全连接Dense
    x = Activation('softmax', name='softmax1')(x)
    x = Reshape((class_num,), name='reshape2')(x)
    print(x.shape)
    inputs = img_input
    model = Model(inputs, x, name='MobileNet')
    model.load_weights('mobilenet_1_0_224_tf.h5')
    return model


def preprocess(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = MobileNet(input_shape=(224, 224, 3))
    path = 'elephant.jpg'
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess(x)
    pred = model.predict(x)
    print(pred)
    print(np.argmax(pred))  # 预测结果不是默认从大到小排序
    print('Prediction:', decode_predictions(pred, 5))  # 默认top5, 只显示top1
