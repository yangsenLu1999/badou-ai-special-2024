'''用tf实现vgg16'''

import tensorflow as tf
# 创建slim对象
# 创建该对象会报导包失败
slim = tf.contrib.slim


def vgg_16(inputs, num_class=1000, is_training=True,
           dropout_keep_prob=0.5, spatial_squeeze=True,
           scope='vgg_16'):

    # 创建一个变量空间，用于构建vgg_16的网络
    with tf.variable_scope(scope, 'vgg_16', [inputs]):

        # conv1两次[3, 3]的卷积，输出64层，输出为(224， 224， 64)
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        # [2, 2]的池化，
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # conv2两次[3，3]卷积，输出128，
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # conv3三次[3，3]卷积，输出256
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # conv4三次[3, 3]卷积，输出512
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        # conv5三次[3, 3]卷积，输出512
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        # 此时输出[7, 7, 512]

        # 利用卷积的方式代替全链接，效果等同，输出net（1， 1， 4096）
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')

        # 利用卷积的方式代替全链接，效果等同，输出net（1， 1， 4096）
        net = slim.conv2d(net, 4096, [1, 1], padding='VALID', scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')

        # # 利用卷积的方式代替全链接，效果等同，输出net（1， 1， num_class）
        net = slim.conv2d(net, num_class, [1, 1],
                          activation_fn=None, normalizer_fn=None,
                          scope='fc8')

        # 由于用卷积的方式模拟全连接层，所以输出需要平铺
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net
