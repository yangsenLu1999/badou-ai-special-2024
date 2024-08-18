# -------------------------------------------------------------#
#   ResNet50的网络部分
# -------------------------------------------------------------#
from __future__ import print_function

import numpy as np
from keras import layers

from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model

from keras.preprocessing import image
import keras.backend as K
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


# def identity_block(input_tensor, kernel_size, filters, stage, block):
#
#     filters1, filters2, filters3 = filters
#
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
#     x = BatchNormalization(name=bn_name_base + '2a')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
#
#     x = BatchNormalization(name=bn_name_base + '2b')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
#     x = BatchNormalization(name=bn_name_base + '2c')(x)
#
#     x = layers.add([x, input_tensor])
#     x = Activation('relu')(x)
#     return x
#
#
# def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
#
#     filters1, filters2, filters3 = filters
#
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = Conv2D(filters1, (1, 1), strides=strides,
#                name=conv_name_base + '2a')(input_tensor)
#     x = BatchNormalization(name=bn_name_base + '2a')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters2, kernel_size, padding='same',
#                name=conv_name_base + '2b')(x)
#     x = BatchNormalization(name=bn_name_base + '2b')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
#     x = BatchNormalization(name=bn_name_base + '2c')(x)
#
#     shortcut = Conv2D(filters3, (1, 1), strides=strides,
#                       name=conv_name_base + '1')(input_tensor)
#     shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
#
#     x = layers.add([x, shortcut])
#     x = Activation('relu')(x)
#     return x
#
#
# def ResNet50(input_shape=[224,224,3],classes=1000):
#
#     img_input = Input(shape=input_shape)
#     x = ZeroPadding2D((3, 3))(img_input)
#
#     x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
#     x = BatchNormalization(name='bn_conv1')(x)
#     x = Activation('relu')(x)
#     x = MaxPooling2D((3, 3), strides=(2, 2))(x)
#
#     x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
#     x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
#     x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
#
#     x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
#
#     x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
#
#     x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
#
#     x = AveragePooling2D((7, 7), name='avg_pool')(x)
#
#     x = Flatten()(x)
#     x = Dense(classes, activation='softmax', name='fc1000')(x)
#
#     model = Model(img_input, x, name='resnet50')
#
#     model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
#
#     return model
#
# if __name__ == '__main__':
#     model = ResNet50()
#     model.summary()
#     img_path = 'elephant.jpg'
#     # img_path = 'bike.jpg'
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#
#     print('Input image shape:', x.shape)
#     preds = model.predict(x)
#     print('Predicted:', decode_predictions(preds))

#
# # 第一次
# def ConvBlock(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
#     filters1, filters2, filters3 = filters
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = Conv2D(filters=filters1, kernel_size=(1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
#     x = BatchNormalization(name=bn_name_base + '2a')(x)
#     x = Activation(activation='relu')(x)
#
#     x = Conv2D(filters=filters2, kernel_size=kernel_size, padding='same', name=conv_name_base + '2b')(x)
#     x = BatchNormalization(name=bn_name_base + '2b')(x)
#     x = Activation(activation='relu')(x)
#
#     x = Conv2D(filters=filters3, kernel_size=(1, 1), name=conv_name_base + '2c')(x)
#     x = BatchNormalization(name=bn_name_base + '2c')(x)
#
#     shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
#     shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
#
#     x = layers.add([x, shortcut])
#     x = Activation(activation='relu')(x)
#
#     return x
#
#     pass
#
#
# def IdentityBlock(input_tensor, kernel_size, filters, stage, block):
#     filters1, filters2, filters3 = filters
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = Conv2D(filters=filters1, kernel_size=(1, 1), name=conv_name_base + '2a')(input_tensor)
#     x = BatchNormalization(name=bn_name_base + '2a')(x)
#     x = Activation(activation='relu')(x)
#
#     x = Conv2D(filters=filters2, kernel_size=kernel_size, padding='same', name=conv_name_base + '2b')(x)
#     x = BatchNormalization(name=bn_name_base + '2b')(x)
#     x = Activation(activation='relu')(x)
#
#     x = Conv2D(filters=filters3, kernel_size=(1, 1), name=conv_name_base + '2c')(x)
#     x = BatchNormalization(name=bn_name_base + '2c')(x)
#
#     # 记得add([x,input_tensor])
#     x = layers.add([x, input_tensor])
#     x = Activation(activation='relu')(x)
#     return x
#
#     pass
#
#
# # resnet网络
# def ResNet50(input_shape=[224, 224, 3], classes=1000):
#     img_input = Input(shape=input_shape)
#     x = ZeroPadding2D(padding=(3, 3))(img_input)
#     x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1')(x)
#     x = BatchNormalization(name='bn_conv1')(x)
#     x = Activation(activation='relu')(x)
#     x = MaxPooling2D(pool_size=[3, 3], strides=[2, 2])(x)
#
#     x = ConvBlock(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
#     x = IdentityBlock(x, 3, [64, 64, 256], stage=2, block='b')
#     x = IdentityBlock(x, 3, [64, 64, 256], stage=2, block='c')
#
#     x = ConvBlock(x, 3, [128, 128, 512], stage=3, block='a')
#     x = IdentityBlock(x, 3, [128, 128, 512], stage=3, block='b')
#     x = IdentityBlock(x, 3, [128, 128, 512], stage=3, block='c')
#     x = IdentityBlock(x, 3, [128, 128, 512], stage=3, block='d')
#
#     x = ConvBlock(x, 3, [256, 256, 1024], stage=4, block='a')
#     x = IdentityBlock(x, 3, [256, 256, 1024], stage=4, block='b')
#     x = IdentityBlock(x, 3, [256, 256, 1024], stage=4, block='c')
#     x = IdentityBlock(x, 3, [256, 256, 1024], stage=4, block='d')
#     x = IdentityBlock(x, 3, [256, 256, 1024], stage=4, block='e')
#     x = IdentityBlock(x, 3, [256, 256, 1024], stage=4, block='f')
#
#     x = ConvBlock(x, 3, [512, 512, 2048], stage=5, block='a')
#     x = IdentityBlock(x, 3, [512, 512, 2048], stage=5, block='b')
#     x = IdentityBlock(x, 3, [512, 512, 2048], stage=5, block='c')
#
#     x = AveragePooling2D(pool_size=[7, 7], name='avg_pool')(x)
#     x = Flatten()(x)
#     # 记得接收x变量
#     x = Dense(classes, activation='softmax', name='fc1000')(x)
#     model = Model(img_input, x, name='resnet50')
#
#     model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5')
#     return model

# 第二次
def IdentityBlock(inputs, kernel_size, filters, stage, block):
    filter1, filter2, filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, (1, 1), strides=(1, 1),
               padding='SAME', name=conv_name_base + '2a')(inputs)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(filter2, kernel_size, strides=(1, 1), padding='SAME',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(filter3, kernel_size=(1, 1), strides=(1, 1),
               padding='SAME', name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    output = layers.add([x, inputs])
    output = Activation(activation='relu')(output)
    return output


def ConvBlock(inputs, kernel_size, filters, stage, block, strides=(2, 2)):
    filter1, filter2, filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, kernel_size=(1, 1), strides=strides, name=conv_name_base + '2a')(inputs)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(filter2, kernel_size=kernel_size, strides=(1, 1), padding='SAME',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(filter3, kernel_size=(1, 1), strides=(1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    short_cut = Conv2D(filter3, kernel_size=(1, 1), strides=strides,
                       name=conv_name_base + '1')(inputs)
    short_cut = BatchNormalization(name=bn_name_base + '1')(short_cut)

    output = layers.add([x, short_cut])
    output = Activation(activation='relu')(output)
    return output


def ResNet50(input_shape=[224, 224, 3], classes=1000):
    inputs = Input(input_shape)
    net = ZeroPadding2D(padding=(3, 3))(inputs)
    net = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), name='conv1')(net)
    net = BatchNormalization(name='bn_conv1')(net)
    net = Activation(activation='relu')(net)
    net = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(net)
    print(net.shape)

    net = ConvBlock(net, kernel_size=(3, 3), filters=[64, 64, 256], stage=2, block='a', strides=(1, 1))
    print(net.shape)
    net = IdentityBlock(net, kernel_size=(3, 3), filters=[64, 64, 256], stage=2, block='b')
    net = IdentityBlock(net, kernel_size=(3, 3), filters=[64, 64, 256], stage=2, block='c')
    print(net.shape)

    net = ConvBlock(net, kernel_size=(3, 3), filters=[128, 128, 512], stage=3, block='a')
    net = IdentityBlock(net, kernel_size=(3, 3), filters=[128, 128, 512], stage=3, block='b')
    net = IdentityBlock(net, kernel_size=(3, 3), filters=[128, 128, 512], stage=3, block='c')
    net = IdentityBlock(net, kernel_size=(3, 3), filters=[128, 128, 512], stage=3, block='d')

    net = ConvBlock(net, kernel_size=(3, 3), filters=[256, 256, 1024], stage=4, block='a')
    net = IdentityBlock(net, kernel_size=(3, 3), filters=[256, 256, 1024], stage=4, block='b')
    net = IdentityBlock(net, kernel_size=(3, 3), filters=[256, 256, 1024], stage=4, block='c')
    net = IdentityBlock(net, kernel_size=(3, 3), filters=[256, 256, 1024], stage=4, block='d')
    net = IdentityBlock(net, kernel_size=(3, 3), filters=[256, 256, 1024], stage=4, block='e')
    net = IdentityBlock(net, kernel_size=(3, 3), filters=[256, 256, 1024], stage=4, block='f')

    net = ConvBlock(net, kernel_size=(3, 3), filters=[512, 512, 2048], stage=5, block='a')
    net = IdentityBlock(net, kernel_size=(3, 3), filters=[512, 512, 2048], stage=5, block='b')
    net = IdentityBlock(net, kernel_size=(3, 3), filters=[512, 512, 2048], stage=5, block='c')

    net = AveragePooling2D(pool_size=(7, 7), name='avg_pool')(net)
    net = Flatten()(net)
    net = Dense(classes, activation='softmax', name='fc1000')(net)

    model = Model(inputs, net, name='resnet50')
    model.load_weights('./resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    return model


if __name__ == '__main__':
    # model = ResNet50()
    # model.summary()
    # img = image.load_img(path='./elephant.jpg', target_size=(224, 224))
    # img_array = image.img_to_array(img)
    # img_array = preprocess_input(img_array)
    # img_array = np.expand_dims(img_array, axis=0)
    # p = model.predict(img_array)
    # print(decode_predictions(p, 1))
    model=ResNet50()
    model.summary()
    img=image.load_img(path='./elephant.jpg',target_size=(224,224))
    img=image.img_to_array(img)
    img=preprocess_input(img)
    img=np.expand_dims(img,0)
    p=model.predict(img)
    print(decode_predictions(p,1))

#
# if __name__ == '__main__':
#     # 构建模型
#     model = ResNet50()
#     model.summary()
#     img = image.load_img(path='./elephant.jpg', target_size=[224, 224])
#     img_array = image.img_to_array(img)
#     img_array = preprocess_input(img_array)
#     img_array = np.expand_dims(img_array, axis=0)
#     p = model.predict(img_array)
#     print(decode_predictions(p))
#     pass

# if __name__ == "__main__":
#     model = ResNet50()
#     # 打印一些模型参数信息
#     model.summary()
#     img_path='elephant.jpg'
#     img=image.load_img(path=img_path,target_size=(224,224))
#     img=image.img_to_array(img)
#     # 对输入图像进行预处理，通常是进行归一化等操作，使其符合模型的输入要求
#     img=preprocess_input(img)
#     img=np.expand_dims(img,axis=0)
#     prids=model.predict(img)
#     print('Predicted:',decode_predictions(prids))


# if __name__=='__main__':
#     model=ResNet50()
#     model.summary()
#     img_path='elephant.jpg'
#     img=image.load_img(img_path,target_size=(224,224))
#     x=image.img_to_array(img)
#     x=np.expand_dims(x,axis=0)
#     x=preprocess_input(x)
#     preds=model.predict(x)
#     print('Predicted:',decode_predictions(preds))

# if __name__ == '__main__':
#     model = ResNet50()
#     model.summary()
#     img_path = 'elephant.jpg'
#     # img_path = 'bike.jpg'
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#
#     print('Input image shape:', x.shape)
#     preds = model.predict(x)
#     print('Predicted:', decode_predictions(preds))
