from __future__ import print_function
import numpy as np
from keras import layers
from keras.layers import Input
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import Activation,BatchNormalization,Flatten
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 添加卷积和归一化层 卷积核大小1*1 缩减维度 归一化 使用relu激活
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # 添加卷积和归一化层 使用补零 归一化 使用relu激活
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 添加卷积和归一化层 卷积核大小1*1 归一化 使用relu激活
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 添加卷积和归一化层 卷积核大小1*1 归一化 使用relu激活
    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # 添加卷积和归一化层 卷积核大小1*1 归一化 使用relu激活
    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 添加卷积和归一化层 归一化
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 与主路径的输出相加 卷积核1*1 归一化
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=[224,224,3],classes=1000):

    # 补零 3*3
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(img_input)

    # 卷积64个核 7*7 步长2*2 归一层 激活层 最大池化3*3 步长2*2 阶段1
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 卷积64*64*256个核 3*3 阶段2 使用a块
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    # 残差64*64*256个核 3*3 阶段2 使用b块
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    # 残差64*64*256个核 3*3 阶段2 使用c块
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # 卷积128*128*512个核 3*3 阶段3 使用a块
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    # 残差128*128*512个核 3*3 阶段3 使用b块
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    # 残差128*128*512个核 3*3 阶段3 使用c块
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    # 残差128*128*512个核 3*3 阶段3 使用d块
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # 卷积256*256*1024个核 3*3 阶段4 使用a块
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    # 残差256*256*1024个核 3*3 阶段4 使用b块
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    # 残差256*256*1024个核 3*3 阶段4 使用c块
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    # 残差256*256*1024个核 3*3 阶段4 使用d块
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    # 残差256*256*1024个核 3*3 阶段4 使用e块
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    # 残差256*256*1024个核 3*3 阶段4 使用f块
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # 卷积512*512*2048个核 3*3 阶段5 使用a块
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    # 残差512*512*2048个核 3*3 阶段5 使用b块
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    # 残差512*512*2048个核 3*3 阶段5 使用c块
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    # 平均池化大小7*7
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    # 拉直
    x = Flatten()(x)
    # 全连接使用softmax得到概率值
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')
    # 加载模型权重
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model

if __name__ == '__main__':
    # 调用网络并总结参数
    model = ResNet50()
    model.summary()
    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    # 限制大小为224*224读取图片 并转为数组添加维度预处理进行计算
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('Input image shape:', x.shape)

    # 调用预测函数
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
