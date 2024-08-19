from keras.layers import Input, Conv2D, MaxPooling2D, Deconvolution2D, Add, ZeroPadding2D
from keras.models import Model


def get_model(n_channels, n_classes):
    """
    :param n_channels:
    :param n_classes:
    :return:
    """

    inputs = Input(shape=[512, 512, n_channels], name='input_image')
    # 下采样部分
    # 512x512x3 -> 254x254x64
    x = Conv2D(64, kernel_size=3, activation='relu')(inputs)
    step1 = x = Conv2D(64, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    # 254x254x64 -> 125x125x128
    x = Conv2D(128, kernel_size=3, activation='relu')(x)
    step2 = x = Conv2D(128, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    # 125x125x128 -> 61x61x256
    x = Conv2D(256, kernel_size=3, activation='relu')(x)
    step3 = x = Conv2D(256, kernel_size=3, activation='relu')(x)
    # 池化向上取整
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    # 61x61x256 -> 29x29x512
    x = Conv2D(512, kernel_size=3, activation='relu')(x)
    step4 = x = Conv2D(512, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    # 中间部分
    # 29x29x512 -> 25x25x1024
    x = Conv2D(1024, kernel_size=3, activation='relu')(x)
    x = Conv2D(1024, kernel_size=3, activation='relu')(x)

    # 上采样部分
    # 51x51x512
    x = Deconvolution2D(filters=512, kernel_size=3, strides=2)(x)
    # 特征合并
    x = ZeroPadding2D(padding=3)(x)
    x = Add()([x, step4])
    x = Conv2D(512, kernel_size=3, activation='relu')(x)
    x = Conv2D(512, kernel_size=3, activation='relu')(x)

    x = Deconvolution2D(filters=256, kernel_size=3, strides=2)(x)
    # 特征合并
    x = ZeroPadding2D(padding=7)(x)
    x = Add()([x, step3])
    x = Conv2D(256, kernel_size=3, activation='relu')(x)
    x = Conv2D(256, kernel_size=3, activation='relu')(x)

    x = Deconvolution2D(filters=128, kernel_size=3, strides=2, padding='same')(x)
    # 特征合并
    x = ZeroPadding2D(padding=8)(x)
    x = Add()([x, step2])
    x = Conv2D(128, kernel_size=3, activation='relu')(x)
    x = Conv2D(128, kernel_size=3, activation='relu')(x)

    x = Deconvolution2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
    # 特征合并
    x = ZeroPadding2D(padding=8)(x)
    x = Add()([x, step1])
    x = Conv2D(64, kernel_size=3, activation='relu')(x)
    x = Conv2D(64, kernel_size=3, activation='relu')(x)

    x = Conv2D(n_classes, kernel_size=1)(x)
    x = ZeroPadding2D(padding=4)(x)

    net = Model([inputs], [x])
    return net
