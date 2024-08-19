'''
googlenet升级V3最具性价比，带并联网络分支，通过1x1卷积减少权重参数，扩大到7x7的卷积核，可以识别不同尺度
h5权重只对应当前网络结构，结构一旦随意修改，就应当重新测试得到新的网络结构下的权重参数。不然导致参数量不匹配
'''
import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, AveragePooling2D, MaxPooling2D, Input
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D  # POOL + FLATTEN
from keras.applications.imagenet_utils import decode_predictions  # 解码模型的预测结果
from keras.preprocessing import image  # 载入图片并转化成数组

# from keras.engine.topology import get_source_inputs
# from keras.utils.layer_utils import convert_all_kernels_in_model
# from keras.utils.data_utils import get_file
# from keras import backend
# import warnings  # Python 中红色警告部分太多时可选择用代码忽略这些警告


def conv_bn(input, filters, size=(3, 3), strides=(1, 1), padding='same', name=None):
    # 整合了conv, bn, activation，不同层的卷积处理时只有尺寸不同，但C+B+A的步骤相同。默认是same, 步长1，也就是不改图像尺寸
    if name is not None:
        conv_name, bn_name,  ac_name = name + '_conv', name + '_bn', name + '_acti'
    else:
        conv_name, bn_name, ac_name = None, None, None
    x = Conv2D(filters, kernel_size=size, strides=strides, padding=padding, use_bias=False, name=conv_name)(input)
    # filters表示输出通道数，即卷积核个数；strides和padding调用函数时未指定则按默认值，activation和use_bias表示是否使用激活函数和偏置
    x = BatchNormalization(scale=False, name=bn_name)(x)
    # gamma 是个可学习的缩放因子（初始化为1），设置 scale=False 禁用缩放; beta 是个科学性的平移变量（初始化为0），设置 center=False 禁用
    x = Activation('relu', name=ac_name)(x)
    return x


def InceptionV3(input_shape=(299, 299, 3), class_num=1000):
    img_input = Input(shape=input_shape)  # 符合调用函数时的增加维度，变(n, h, w, c)
    '''
    shape和batch_shape只能使用一个。
    shape：会自动在最前面补充一个维度，比如a = Input(shape=[3,None])，那么此时a的具体形状为[None,3,None]
    batch_shape：就不会补充维度，比如a = Input(shape=[3,None])，那么此时a的具体形状为[3,None]
    '''
    x = conv_bn(img_input, 32, (3, 3), strides=(2, 2), padding='valid')  # [299,299,3] -> [149,149,32] 卷积核[3, 3]步长2，不做填充：（299-3）/2 +1=149 如遇小数向下取整
    x = conv_bn(x, 32, (3, 3), strides=(1, 1), padding='valid')  # [149,149,32] -> [147,147,32] 卷积核[3, 3]步长1，不做填充：（149-3）/1 +1=147
    x = conv_bn(x, 64, (3, 3), strides=(1, 1), padding='same')  # [147,147,32] -> [147,147,64] 卷积核[3, 3]步长1，做填充：尺寸不变
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)  # [147,147,64] -> [73,73,64] 最大池化[3, 3]步长2：（147-3）/2 +1=73
    x = conv_bn(x, 80, (1, 1), padding='valid')  # [73,73,64] -> [73,73,80]课件卷积核3步长2，老师代码卷积核[1, 1]步长1：（73-1）/1 +1=73
    x = conv_bn(x, 192, (3, 3), padding='valid')  # [73,73,80] -> [71,71,192]课件卷积核3步长2，老师代码步长1：（73-3）/1 +1=71
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)  # [71,71,192] -> [35,35,192] 最大池化3步长2：（71-3）/2 +1=35

    # 上述为：常规卷积*3 + 最大池化*1 + 常规卷积*2 + 最大池化*1，[229, 229, 3] --> [35, 35, 192], 进入Inception
    # --------------------- #
    # Block1 35*35
    # ---------------------part1  1x1 + 5x5 + 3x3double + pool  [35,35,192] -> [35,35,256] # 默认CONV2DBN函数是same, (1,1)不改图像形状
    branch1x1 = conv_bn(x, 64, (1, 1))  # [35,35,192] -> [35,35,64] 卷积核[1, 1]步长1，做填充：尺寸不变
    branch5x5 = conv_bn(x, 48, (1, 1))
    branch5x5 = conv_bn(branch5x5, 64, (5, 5))  # [35,35,192] -> [35,35,64] 卷积核[5, 5]步长1，做填充：尺寸不变
    branch3x3db = conv_bn(x, 64, (1, 1))
    branch3x3db = conv_bn(branch3x3db, 96, (3, 3))  # double3x3相当于5x5
    branch3x3db = conv_bn(branch3x3db, 96, (3, 3))  # [35,35,192] -> [35,35,96] 卷积核[3, 3]步长1，做填充：尺寸不变
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)  # 平均池化后面接C+B+A
    branch_pool = conv_bn(branch_pool, 32, (1, 1))  # [35,35,192] -> [35,35,32] 平均池化[1, 1]步长1，做填充：尺寸不变
    x = layers.concatenate([branch1x1, branch5x5, branch3x3db, branch_pool], axis=3, name='mixed1-1')  # axis沿着第3轴拼接，要求前两轴尺寸相同即[17, 17]，第三轴为64+64+96+32=256
    # ---------------------part2  1x1 + 5x5 + 3x3double + pool  [35,35,256] -> [35,35,288]#
    branch1x1 = conv_bn(x, 64, (1, 1))
    branch5x5 = conv_bn(x, 48, (1, 1))
    branch5x5 = conv_bn(branch5x5, 64, (5, 5))
    branch3x3db = conv_bn(x, 64, (1, 1))
    branch3x3db = conv_bn(branch3x3db, 96, (3, 3))
    branch3x3db = conv_bn(branch3x3db, 96, (3, 3))
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_bn(branch_pool, 64, (1, 1))
    x = layers.concatenate([branch1x1, branch5x5, branch3x3db, branch_pool], axis=3, name='mixed1-2')  # 第三轴为64+64+96+64=288
    # ---------------------part3  1x1 + 5x5 + 3x3double + pool  [35,35,288] -> [35,35,288]#
    branch1x1 = conv_bn(x, 64, (1, 1))
    branch5x5 = conv_bn(x, 48, (1, 1))
    branch5x5 = conv_bn(branch5x5, 64, (5, 5))
    branch3x3db = conv_bn(x, 64, (1, 1))
    branch3x3db = conv_bn(branch3x3db, 96, (3, 3))
    branch3x3db = conv_bn(branch3x3db, 96, (3, 3))
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_bn(branch_pool, 64, (1, 1))
    x = layers.concatenate([branch1x1, branch5x5, branch3x3db, branch_pool], axis=3, name='mixed1-3')  # 第3轴为64+64+96+64=288

    # --------------------- #
    # Block2 17*17
    # ---------------------part1  3x3 + 3x3double + pool  [35,35,288] -> [17,17,768]#
    branch3x3 = conv_bn(x, 384, (3, 3), strides=(2, 2), padding='valid')  # [35,35,288] -> [17,17,384] 卷积核[3, 3]步长2，不做填充：（35-3）/1 +1=17
    branch3x3db = conv_bn(x, 64, (1, 1))
    branch3x3db = conv_bn(branch3x3db, 96, (3, 3))
    branch3x3db = conv_bn(branch3x3db, 96, (3, 3), strides=(2, 2), padding='valid')  # [35,35,96] -> [17,17,64] 卷积核[3, 3]步长2，不做填充：（35-3）/1 +1=17
    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)  # [35,35,288] -> [17,17,64] 最大池化[3, 3]步长2，不做填充：（35-3）/1 +1=17
    x = layers.concatenate([branch3x3, branch3x3db, branch_pool], axis=3, name='mixed2-1')  # 第3轴为384+96+288=768
    # ---------------------part2  1x1 + 7x7 + 7x7double + pool  [17,17,768] -> [17,17,768]#
    branch1x1 = conv_bn(x, 192, (1, 1))  # [17,17,768] -> [17,17,192] 卷积核[1, 1]步长1，做填充：尺寸不变
    branch7x7 = conv_bn(x, 128, (1, 1))
    branch7x7 = conv_bn(branch7x7, 128, (1, 7))
    branch7x7 = conv_bn(branch7x7, 192, (7, 1))  # [17,17,768] -> [17,17,192] 卷积核[1, 1]步长1，做填充：尺寸不变
    branch7x7db = conv_bn(x, 128, (1, 1))
    branch7x7db = conv_bn(branch7x7db, 128, (7, 1))
    branch7x7db = conv_bn(branch7x7db, 128, (1, 7))
    branch7x7db = conv_bn(branch7x7db, 128, (7, 1))
    branch7x7db = conv_bn(branch7x7db, 192, (1, 7))  # [17,17,768] -> [17,17,192] 卷积核[1, 1]步长1，做填充：尺寸不变
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_bn(branch_pool, 192, (1, 1))  # [17,17,768] -> [17,17,192] 平均池化[1, 1]步长1，做填充：尺寸不变
    x = layers.concatenate([branch1x1, branch7x7, branch7x7db, branch_pool], axis=3, name='mixed2-2')  # 第3轴为192+192+192+192=768

    # ---------------------part3-4  1x1 + 7x7 + 7x7double + pool  [17,17,768] -> [17,17,768] -> [17,17,768]#
    for i in range(2):
        branch1x1 = conv_bn(x, 192, (1, 1))  # [17,17,768] -> [17,17,192] 卷积核[1, 1]步长1，做填充：尺寸不变
        branch7x7 = conv_bn(x, 160, (1, 1))
        branch7x7 = conv_bn(branch7x7, 160, (1, 7))
        branch7x7 = conv_bn(branch7x7, 192, (7, 1))  # [17,17,768] -> [17,17,192] 卷积核[1, 1]步长1，做填充：尺寸不变
        branch7x7db = conv_bn(x, 160, (1, 1))
        branch7x7db = conv_bn(branch7x7db, 160, (7, 1))
        branch7x7db = conv_bn(branch7x7db, 160, (1, 7))
        branch7x7db = conv_bn(branch7x7db, 160, (7, 1))
        branch7x7db = conv_bn(branch7x7db, 192, (1, 7))  # [17,17,768] -> [17,17,192] 卷积核[1, 1]步长1，做填充：尺寸不变
        branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv_bn(branch_pool, 192, (1, 1))  # [17,17,768] -> [17,17,192] 平均池化[1, 1]步长1，做填充：尺寸不变
        x = layers.concatenate([branch1x1, branch7x7, branch7x7db, branch_pool], axis=3, name='mixed2-' + str(i+3))  # 第3轴为192+192+192+192=768
    # ---------------------part5  1x1 + 7x7 + 7x7double + pool  [17,17,768] -> [17,17,768]#
    branch1x1 = conv_bn(x, 192, (1, 1))  # [17,17,768] -> [17,17,192] 卷积核[1, 1]步长1，做填充：尺寸不变
    branch7x7 = conv_bn(x, 192, (1, 1))
    branch7x7 = conv_bn(branch7x7, 192, (1, 7))
    branch7x7 = conv_bn(branch7x7, 192, (7, 1))  # [17,17,768] -> [17,17,192] 卷积核[1, 1]步长1，做填充：尺寸不变
    branch7x7db = conv_bn(x, 192, (1, 1))
    branch7x7db = conv_bn(branch7x7db, 192, (7, 1))
    branch7x7db = conv_bn(branch7x7db, 192, (1, 7))
    branch7x7db = conv_bn(branch7x7db, 192, (7, 1))
    branch7x7db = conv_bn(branch7x7db, 192, (1, 7))  # [17,17,768] -> [17,17,192] 卷积核[1, 1]步长1，做填充：尺寸不变
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)  # [17,17,288] -> [17,17,288] 平均池化[1, 1]步长1，做填充：尺寸不变
    branch_pool = conv_bn(branch_pool, 192, (1, 1))
    x = layers.concatenate([branch1x1, branch7x7, branch7x7db, branch_pool], axis=3, name='mixed2-5')  # 第3轴为192+192+192+192=768

    # --------------------- #
    # Block3 7*7
    # ---------------------part1  3x3 + 7x7x3 + pool  [17,17,768] -> [8,8,1280]#
    branch3x3 = conv_bn(x, 192, (1, 1))
    branch3x3 = conv_bn(branch3x3, 320, (3, 3), strides=(2, 2), padding='valid')  # [17,17,192] -> [8,8,320] 卷积核[3, 3]步长2，不做填充：（17-3）/2 +1=8
    branch7x7x3 = conv_bn(x, 192, (1, 1))
    branch7x7x3 = conv_bn(branch7x7x3, 192, (1, 7))
    branch7x7x3 = conv_bn(branch7x7x3, 192, (7, 1))
    branch7x7x3 = conv_bn(branch7x7x3, 192, (3, 3), strides=(2, 2), padding='valid')  # [17,17,192] -> [8,8,192] 卷积核[3, 3]步长2，不做填充：（17-3）/2 +1=8
    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)  # [17,17,768] -> [8,8,768] 最大池化[3, 3]步长2，不做填充：（17-3）/2 +1=8
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed3-1')  # 第3轴为320+192+768=1280
    # ---------------------part2-3  1x1 + 3x3 + 3x3double + pool  [8,8,1280] -> [8,8,2048] -> [8,8,2048]#
    for i in range(2):
        branch1x1 = conv_bn(x, 320, (1, 1))  # [8,8,1280] -> [8,8,320] 卷积核[1, 1]步长1，做填充：尺寸不变
        branch3x3 = conv_bn(x, 384, (1, 1))
        branch3x3_1 = conv_bn(branch3x3, 384, (1, 3))
        branch3x3_2 = conv_bn(branch3x3, 384, (3, 1))
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3, name=None)  # 第3轴为384+384=768
        branch3x3db = conv_bn(x, 448, (1, 1))
        branch3x3db = conv_bn(branch3x3db, 384, (3, 3))
        branch3x3db_1 = conv_bn(branch3x3db, 384, (1, 3))
        branch3x3db_2 = conv_bn(branch3x3db, 384, (3, 1))
        branch3x3db = layers.concatenate([branch3x3db_1, branch3x3db_2], axis=3, name=None)  # 第3轴为384+384=768
        branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv_bn(branch_pool, 192, (1, 1))  # [8,8,1280] -> [8,8,192] 平均池化[1, 1]步长1，做填充：尺寸不变
        x = layers.concatenate([branch1x1, branch3x3, branch3x3db, branch_pool], axis=3, name='mixed3-' + str(i+2))  # 第3轴为320+768+768+192=2048
    # --------------------- #
    x = GlobalAveragePooling2D(name='average_pool')(x)  # 即flatten + pooling
    x = Dense(class_num, activation='softmax', name='prediction')(x)
    inputs = img_input
    model = Model(inputs, x, name='Inception_V3')
    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x  # 图片元素归一化到[-1,1]


if __name__ == '__main__':
    model = InceptionV3()
    model.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
    path = 'elephant1.jpg'
    img = image.load_img(path, target_size=(299, 299))
    x = image.img_to_array(img)  # 将图片转化成数组，元素类型是整型转换后是浮点型。
    x = np.expand_dims(x, axis=0)  # 相当于在最前增加维度变为(n, h, w, c), n表述图片个数，这里是1
    x = preprocess_input(x)  # 图片预处理，归一化
    pred = model.predict(x)
    print('Predict:', decode_predictions(pred))  # 解码模型的预测结果，两参数:[preds:Numpy数组编码的批预测; top:整数，返回多少个概率大的值。默认为5。]
    '''
    AttributeError: ‘str‘ object has no attribute ‘decode‘原始h5py版本3.7.0需要降级到2.1.0
    '''
