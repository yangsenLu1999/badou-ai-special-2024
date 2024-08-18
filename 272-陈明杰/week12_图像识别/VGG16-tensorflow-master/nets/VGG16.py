#-------------------------------------------------------------#
#   vgg16的网络部分
#-------------------------------------------------------------#
import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization
# from tensorflow.keras.models import Model


# # 创建slim对象
# slim = tf.contrib.slim
#
# def vgg_16(inputs,
#            num_classes=1000,
#            is_training=True,
#            dropout_keep_prob=0.5,
#            spatial_squeeze=True,
#            scope='vgg_16'):
#
#     with tf.variable_scope(scope, 'vgg_16', [inputs]):
#         # 建立vgg_16的网络
#
#         # conv1两次[3,3]卷积网络，输出的特征层为64，输出为(224,224,64)
#         net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
#         # 2X2最大池化，输出net为(112,112,64)
#         net = slim.max_pool2d(net, [2, 2], scope='pool1')
#
#         # conv2两次[3,3]卷积网络，输出的特征层为128，输出net为(112,112,128)
#         net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
#         # 2X2最大池化，输出net为(56,56,128)
#         net = slim.max_pool2d(net, [2, 2], scope='pool2')
#
#         # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(56,56,256)
#         net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
#         # 2X2最大池化，输出net为(28,28,256)
#         net = slim.max_pool2d(net, [2, 2], scope='pool3')
#
#         # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(28,28,512)
#         net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
#         # 2X2最大池化，输出net为(14,14,512)
#         net = slim.max_pool2d(net, [2, 2], scope='pool4')
#
#         # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(14,14,512)
#         net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
#         # 2X2最大池化，输出net为(7,7,512)
#         net = slim.max_pool2d(net, [2, 2], scope='pool5')
#
#         # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
#         net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
#         net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
#                             scope='dropout6')
#         # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
#         net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
#         net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
#                             scope='dropout7')
#         # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1000)
#         net = slim.conv2d(net, num_classes, [1, 1],
#                         activation_fn=None,
#                         normalizer_fn=None,
#                         scope='fc8')
#
#         # 由于用卷积的方式模拟全连接层，所以输出需要平铺
#         if spatial_squeeze:
#             net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
#         return net
#


# # 第一次
# slim=tf.contrib.slim
#
# # inputs是一个(1,224,224,3)结构的张量
# def vgg_16(inputs,
#            num_classes=1000,
#            is_training=True,
#            dropout_keep_prob=0.5,
#            spatial_squeeze=True,
#            scope='vgg_16'
#            ):
#     with tf.variable_scope(scope,'vgg_16',[inputs]):
#         # inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1'
#         # 输入，层数，卷积，输出通道，卷积核大小，操作的名称
#         net=slim.repeat(inputs,2,slim.conv2d,64,[3,3],scope='conv1')
#         # net, [2, 2], scope = 'pool1'
#         # 输入，池化的大小，操作的名称
#         net=slim.max_pool2d(net,[2,2],scope='pool1')
#
#         net=slim.repeat(net,2,slim.conv2d,128,[3,3],scope='conv2')
#         net=slim.max_pool2d(net,[2,2],scope='pool2')
#
#         net=slim.repeat(net,3,slim.conv2d,256,[3,3],scope='conv3')
#         net=slim.max_pool2d(net,[2,2],scope='pool3')
#
#         net=slim.repeat(net,3,slim.conv2d,512,[3,3],scope='conv4')
#         net=slim.max_pool2d(net,[2,2],scope='pool4')
#
#         net=slim.repeat(net,3,slim.conv2d,512,[3,3],scope='conv5')
#         net=slim.max_pool2d(net,[2,2],scope='pool5')
#
#         # net, 4096, [7, 7], padding = 'VALID', scope = 'fc6'
#         # 输入，输出通道数，卷积核大小，方式，操作名称
#         net=slim.conv2d(net,4096,[7,7],padding='VALID',scope='fc6')
#         # input_tensor（或类似的命名）：这是输入到dropout层的张量（tensor）。
#         # keep_prob：这是一个浮点数，表示在训练过程中保留神经元的比例。例如，如果keep_prob设置为0.5，那么在每个训练步骤中，大约一半的神经元会被随机“关闭”或“丢弃”。
#         # is_training：这是一个布尔值，用于指示当前是否处于训练模式。当is_training为True时，dropout会被应用；当为False时，dropout不会被应用，所有的神经元都会被保留（即相当于keep_prob=1.0）。
#         # scope（可选）：这是操作的可选作用域名称。
#         # seed（可选）：用于随机数生成的种子。在固定种子的情况下，每次运行都会得到相同的dropout模式，这有助于实验的可重复性。
#         # noise_shape（可选）：一个张量，用于指定每个元素的dropout概率。如果未指定，则假定所有元素都有相同的dropout概率。
#         net=slim.dropout(net,dropout_keep_prob,is_training=is_training,scope='dropout6')
#         net=slim.conv2d(net,4096,[1,1],padding='VALID',scope='fc7')
#         net=slim.dropout(net,dropout_keep_prob,is_training=is_training,scope='dropout7')
#         net=slim.conv2d(net,num_classes,[1,1],padding='VALID',scope='fc8',
#                         activation_fn=None,
#                         normalizer_fn=None,
#                         )
#
#         if spatial_squeeze:
#             net=tf.squeeze(net,axis=[1,2],name='fc8/squeezed')
#         print(net)
#         return net
#
#
#






# # 第一次
# from keras.layers import Input
# def vgg_16(inputs,
#            num_classes=1000,
#            is_training=True,
#            dropout_keep_prob=0.5,
#            spatial_squeeze=True,
#            ):
#     print(inputs)
#
#     # 两个卷积层
#     net=Conv2D(filters=64,kernel_size=(3,3),padding='same',
#                activation='relu',name='conv_1')(inputs)
#     net=BatchNormalization(axis=-1)(net)
#     net=Conv2D(filters=64,kernel_size=(3,3),padding='same',
#                activation='relu',name='conv_2')(net)
#     net=BatchNormalization(axis=-1)(net)
#     print(net.shape)
#     # 一个最大池化层
#     net=MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid',name='pool_1')(net)
#     print(net.shape)
#
#     # 两个卷积层
#     net=Conv2D(filters=128,kernel_size=(3,3),padding='same',
#                activation='relu',name='conv_3')(net)
#     net=BatchNormalization(axis=-1)(net)
#     net=Conv2D(filters=128,kernel_size=(3,3),padding='same',
#                activation='relu',name='conv_4')(net)
#     net=BatchNormalization(axis=-1)(net)
#
#     # 一个最大池化层
#     net=MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid',name='pool_2')(net)
#     print(net.shape)
#
#
#     # 三个卷积层
#     net=Conv2D(filters=256,kernel_size=(3,3),padding='same',
#                activation='relu',name='conv_5')(net)
#     net=BatchNormalization(axis=-1)(net)
#     net=Conv2D(filters=256,kernel_size=(3,3),padding='same',
#                activation='relu',name='conv_6')(net)
#     net=BatchNormalization(axis=-1)(net)
#     net=Conv2D(filters=256,kernel_size=(3,3),padding='same',
#                activation='relu',name='conv_7')(net)
#     net=BatchNormalization(axis=-1)(net)
#
#     # 最大池化层
#     net=MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid',name='pool_3')(net)
#     print(net.shape)
#
#
#     # 三个卷积层
#     net=Conv2D(filters=512,kernel_size=(3,3),padding='same',
#                activation='relu',name='conv_8')(net)
#     net=BatchNormalization(axis=-1)(net)
#     net=Conv2D(filters=512,kernel_size=(3,3),padding='same',
#                activation='relu',name='conv_9')(net)
#     net=BatchNormalization(axis=-1)(net)
#     net=Conv2D(filters=512,kernel_size=(3,3),padding='same',
#                activation='relu',name='conv_10')(net)
#     net=BatchNormalization(axis=-1)(net)
#
#     # 最大池化层
#     net=MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid',name='pool_4')(net)
#     print(net.shape)
#
#
#     # 三个卷积层
#     net=Conv2D(filters=512,kernel_size=(3,3),padding='same',
#                activation='relu',name='conv_11')(net)
#     net=BatchNormalization(axis=-1)(net)
#     net=Conv2D(filters=512,kernel_size=(3,3),padding='same',
#                activation='relu',name='conv_12')(net)
#     net=BatchNormalization(axis=-1)(net)
#     net=Conv2D(filters=512,kernel_size=(3,3),padding='same',
#                activation='relu',name='conv_13')(net)
#     net=BatchNormalization(axis=-1)(net)
#
#     # 最大池化层
#     net=MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid',name='pool_5')(net)
#     print(net.shape)
#
#
#     # 两次卷积层
#     net=Conv2D(filters=4096,kernel_size=(7,7),padding='valid',
#                activation='relu',name='conv_14')(net)
#     net=BatchNormalization(axis=-1)(net)
#     print(net.shape)
#     net=Dropout(dropout_keep_prob)(net)
#     net=Conv2D(filters=4096,kernel_size=(1,1),padding='valid',
#                activation='relu',name='conv_15')(net)
#     net=BatchNormalization(axis=-1)(net)
#     print(net.shape)
#     net=Dropout(dropout_keep_prob)(net)
#     # 一次卷积层
#     net=Conv2D(filters=1000,kernel_size=(1,1),padding='valid',
#                activation='relu',name='conv_16')(net)
#     net=BatchNormalization(axis=-1)(net)
#     print(net.shape)
#
#     if spatial_squeeze:
#         net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
#     print(net.shape)
#
#     return net
#

# 第二次
slim=tf.contrib.slim

def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):

    # repeat(输入, 重复次数, 操作, 输出通道数, 卷积核大小, scope='')
    with tf.variable_scope(scope,'vgg_16',[inputs]):

        # 卷积两层，池化
        net=slim.repeat(inputs,2,slim.conv2d,64,[3,3],scope='conv1')
        # net=slim.max_pool2d(输入,池化核大小，scope='')
        net=slim.max_pool2d(net,[2,2],scope='pool1')

        # 卷积两层，池化
        net=slim.repeat(net,2,slim.conv2d,128,[3,3],scope='conv2')
        net=slim.max_pool2d(net,[2,2],scope='pool2')

        # 卷积三层，池化
        net=slim.repeat(net,3,slim.conv2d,256,[3,3],scope='conv3')
        net=slim.max_pool2d(net,[2,2],scope='pool3')

        # 卷积三层，池化
        net=slim.repeat(net,3,slim.conv2d,512,[3,3],scope='conv4')
        net=slim.max_pool2d(net,[2,2],scope='pool4')

        # 卷积三层，池化
        net=slim.repeat(net,3,slim.conv2d,512,[3,3],scope='conv5')
        net=slim.max_pool2d(net,[2,2],scope='pool5')

        # [7,7]卷积
        net=slim.conv2d(net,4096,[7,7],stride=[1,1],padding='VALID',scope='fc6')
        net=slim.dropout(net,dropout_keep_prob,is_training=is_training,scope='dropout6')
        net=slim.conv2d(net,4096,[1,1],stride=[1,1],padding='VALID',scope='fc7')
        net = slim.dropout(net, dropout_keep_prob,is_training=is_training,scope='dropout7')
        net=slim.conv2d(net,num_classes,[1,1],stride=[1,1],padding='VALID',
                        activation_fn=None,normalizer_fn=None,scope='fc8')
        if spatial_squeeze:
            net=tf.squeeze(net,[1,2],name='fc8/squeezed')

        return net
