import tensorflow.compat.v1 as tf


tf.disable_control_flow_v2()

slim = tf.contrib.slim


def vgg_16(inputs,
           num_classes=1000,
           is_training = True,
           dropout_keep_prob =0.5,
           spatial_squeeze = True,
           scope = 'vgg_16'
           ):

    '''

    :param inputs:  输入张量
    :param num_classes:  分类任务类别
    :param is_training:    区分是否需要训练
    :param dropout_keep_prob:  droupt率
    :param spatial_squeeze:   该操作用是否减少卷积神经网络（CNN）中特征图的维数
    :return:

    scope用于定义变量作用域的上下文管理器
    '''



    # 以下运行在tensflow 1.X版本

    with tf.variable_scope(scope, 'vgg_16', [inputs]):
       # 建立vgg_16的网络

       # 两次[3,3]卷积 输出224,224,64
       net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
       # 112,112,64
       net = slim.max_pool2d(net, [2, 2], scope='pool1')

       # 112,112,128
       net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
       # 56,56,128
       net = slim.max_pool2d(net, [2, 2], scope='pool2')

       # 56,56,256
       net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
       #  28,28,256
       net = slim.max_pool2d(net, [2, 2], scope='pool3')

       # 28,28,512
       net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
       # 14,14,512
       net = slim.max_pool2d(net, [2, 2], scope='pool4')

       # 14,14,512
       net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
       # 7,7,512
       net = slim.max_pool2d(net, [2, 2], scope='pool5')

       # 1,1,4096
       net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
       net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                          scope='dropout6')
       # 卷积模拟全连接层 1,1,4096
       net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
       net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                          scope='dropout7')
       # 1,1,1000
       net = slim.conv2d(net, num_classes, [1, 1],
                         activation_fn=None,
                         normalizer_fn=None,
                         scope='fc8')

       # 由于用卷积的方式模拟全连接层，所以输出需要平铺
       if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
       return net


