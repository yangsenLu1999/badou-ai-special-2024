'''
residual net残差结构50层，增加网络层数的同时，减少信息损失过拟合，相比于VGG16/19层数上明显提高
网络结构理论上可根据结果自己调整，这里由于直接调用了训练好的.h权重参数，网络结构必须一致，否则权重不匹配，不能推理。
'''
import numpy as np
from keras import layers
from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model

from keras.preprocessing import image  # 按路径读图，并调整成指定大小，灰度转换
from keras.applications.imagenet_utils import preprocess_input  # 归一化标准化
from keras.applications.imagenet_utils import decode_predictions  # 预测的顶级类别和概率从数字表示为人可读的标签


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filter1, filter2, filter3 = filters  # filters是一个列表，1~3分别对应[0],[1],[2]
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)  # filter1表示卷积后维度, 默认padding='valid'
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
    # conv 2a层和shortcut层strides=(2, 2), conv 2b/2c层strides=(1, 1)，保证了shortcut和conv图像大小一致

    x = layers.add([x, shortcut])  # conv 2a层和shortcut层都用了filter3, 保证结果的(h, w, c)一致，可以add
    x = Activation('relu')(x)
    return x
    pass


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = Conv2D(filter1, (1, 1), name=conv_name_base + '2a')(input_tensor)  # 默认缺省参数strides=(1, 1), padding='valid'
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Conv2D(filter2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])  # 由于identityblock的shortcut没有处理，所以conv 2c层的filter3要和输入input的通道数相同
    x = Activation('relu')(x)
    return x
    pass


def resnet50(input_shape=(224, 224, 3), num_classes=1000):
    img_input = Input(shape=input_shape)  # 通常定义一个新的网络模型时，需要先定义一个Input层指定数据输入格式。
    x = ZeroPadding2D((3, 3))(img_input)  # 在img_input图形上下左右，分别增加3个0，因为马上做7x7卷积

    x = Conv2D(64, [7, 7], strides=(2, 2), name='conv1')(x)  # 计算param时，有w和bias，所以是64+7*7*3*64=9472
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)  # conv2d和maxpooling2d步长都是2，也在进行降维
    # conv_block是正常降维提取特征，高宽通道数都会变，identity是增加网络结构深度的，不会改变高宽通道数，所以strides不一样
    '''
    为何与conv_block里定义的strides不同？得分情况，不需要降维的时候用(1,1)，需要就用默认的(2,2)呗，都是实验出来的
    '''
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
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

    x = AveragePooling2D(pool_size=(7, 7), name='avg_pool')(x)  # strides if None，默认为pool_size=(7, 7)
    x = Flatten()(x)  # 注意格式
    x = Dense(num_classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')
    model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5')  # keras训练权重结果是.h5
    return model
    pass


if __name__ == '__main__':
    model = resnet50()
    model.summary()  # 显示构建模型的层级关系，统计各种参数量，bias应该就是不训练的参数
    img1_path = 'elephant.jpg'
    img2_path = 'bike.jpg'
    img = image.load_img(img2_path, target_size=(224, 224))  # 从指定路径加载图像，并将其调整为指定的大小(224, 224)
    print(type(img), img.size)  # 此时还是图片，没有shape, <class 'PIL.Image.Image'> (224, 224)
    x = image.img_to_array(img)  # [224, 224, 3]
    print(x.shape)  # 图片转化成数组，且是浮点型数据
    x = np.expand_dims(x, axis=0)  # [1, 224, 224, 3]
    print(x.shape)
    x = preprocess_input(x)  # 图像标准化归一化，缩小差异提高准确度和收敛速度

    print('Input image shape:', x.shape)
    pred = model.predict(x)
    print('Predicted:', decode_predictions(pred))
