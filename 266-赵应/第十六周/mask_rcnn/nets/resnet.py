from keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, Add
from keras.models import Model


def conv_block(inputs, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    filter1, filter2, filter3 = filters
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base + "2a", use_bias=use_bias)(inputs)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)
    x = Activation('relu')(x)

    shortcut = Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)(inputs)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = Add()([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + '_out')(x)
    return x


def identity_block(inputs, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    filter1, filter2, filter3 = filters
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(filter1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias)(inputs)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

    x = Add()([x, inputs])
    x = Activation('relu', name="res" + str(stage) + block + "_out")(x)
    return x


def get_resnet(input_image, stage5=False, train_bn=True):
    """
    使用resnet网络抽取公共特征并返回4种抽象程度不同的特征。默认按照输入1024x1024计算
    :param input_image:
    :param stage5: 是否启用第五层网络结构
    :param train_bn:
    :return:
    """
    # stage1
    # 1024x1024 -> 1030x1030
    x = ZeroPadding2D((3, 3))(input_image)
    # 1030x1030 -> 512x512
    x = Conv2D(64, (7, 7), strides=2, use_bias=True, name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x, training=train_bn)
    x = Activation('relu')(x)

    # 512x512 -> 256x256
    C1 = x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # stage2
    # 256x256 -> 256x256
    x = conv_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)

    # stage 3
    # 256x256 -> 128x128
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)

    # stage4
    # 64x64
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = 22
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    if stage5:
        # 64x64 -> 32x32
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)

    else:
        C5 = None
    return [C1, C2, C3, C4, C5]
