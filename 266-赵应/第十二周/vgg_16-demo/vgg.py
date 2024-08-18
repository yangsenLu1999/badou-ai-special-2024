import tensorflow as tf


class Vgg16:
    def __init__(self, input_data):
        self.input_data = input_data

    def vgg_16(self,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True, scope='vgg_16'):
        """使用tensorflow构建vgg_16"""
        with tf.variable_scope(scope, 'vgg_16', [self.input_data]):
            # conv1两次[3,3]卷积网络，输出的特征层为64，输出为(224,224,64)
            with tf.variable_scope(scope, 'conv1'):
                conv_output = conv2d(self.input_data, ksize=[3, 3, 3, 64], strides=[1, 1, 1, 1], padding='SAME')
                conv_output = conv2d(conv_output, ksize=[3, 3, 64, 64], strides=[1, 1, 1, 1], padding='SAME')
                # 2X2最大池化，输出net为(112,112,64)
                pool_output = tf.nn.max_pool2d(conv_output, ksize=[2, 2], strides=[1, 2, 2, 1], padding='VALID')

            # conv2两次[3,3]卷积网络，输出的特征层为128，输出net为(112,112,128)
            conv_output = conv2d(pool_output, ksize=[3, 3, 64, 128], strides=[1, 1, 1, 1], padding='SAME')
            conv_output = conv2d(conv_output, ksize=[3, 3, 128, 128], strides=[1, 1, 1, 1], padding='SAME')
            # 2X2最大池化，输出net为(56,56,128)
            pool_output = tf.nn.max_pool2d(conv_output, [2, 2], strides=[1, 2, 2, 1], padding='VALID')

            # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(56,56,256)
            conv_output = conv2d(pool_output, ksize=[3, 3, 128, 256], strides=[1, 1, 1, 1], padding='SAME')
            conv_output = conv2d(conv_output, ksize=[3, 3, 256, 256], strides=[1, 1, 1, 1], padding='SAME')
            conv_output = conv2d(conv_output, ksize=[3, 3, 256, 256], strides=[1, 1, 1, 1], padding='SAME')
            # 2X2最大池化，输出net为(28,28,256)
            pool_output = tf.nn.max_pool2d(conv_output, [2, 2], strides=[1, 2, 2, 1], padding='VALID')

            # conv3三次[3,3]卷积网络，输出的特征层为512，输出net为(28,28,512)
            conv_output = conv2d(pool_output, ksize=[3, 3, 256, 512], strides=[1, 1, 1, 1], padding='SAME')
            conv_output = conv2d(conv_output, ksize=[3, 3, 512, 512], strides=[1, 1, 1, 1], padding='SAME')
            conv_output = conv2d(conv_output, ksize=[3, 3, 512, 512], strides=[1, 1, 1, 1], padding='SAME')
            # 2X2最大池化，输出net为(14,14,512)
            pool_output = tf.nn.max_pool2d(conv_output, [2, 2], strides=[1, 2, 2, 1], padding='VALID')

            # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(14,14,512)
            conv_output = conv2d(pool_output, ksize=[3, 3, 512, 512], strides=[1, 1, 1, 1], padding='SAME')
            conv_output = conv2d(conv_output, ksize=[3, 3, 512, 512], strides=[1, 1, 1, 1], padding='SAME')
            conv_output = conv2d(conv_output, ksize=[3, 3, 512, 512], strides=[1, 1, 1, 1], padding='SAME')
            # 2X2最大池化，输出net为(7,7,512)
            pool_output = tf.nn.max_pool2d(conv_output, [2, 2], strides=[1, 2, 2, 1], padding='VALID')

            # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
            conv_output = conv2d(pool_output, ksize=[7, 7, 512, 4096], strides=[1, 1, 1, 1], padding='VALID')
            if is_training is not None:
                dropout_output = tf.nn.dropout(conv_output, dropout_keep_prob)
            # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
            conv_output = conv2d(dropout_output, ksize=[1, 1, 4096, 4096], strides=[1, 1, 1, 1], padding='VALID')
            if is_training is not None:
                dropout_output = tf.nn.dropout(conv_output, dropout_keep_prob)
            # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1000)
            net = conv2d(dropout_output, ksize=[1, 1, 4096, num_classes], strides=[1, 1, 1, 1], padding='VALID')

            # 由于用卷积的方式模拟全连接层，所以输出需要平铺
            if spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
            return net


def conv2d(input_data, ksize=None, strides=None, padding=None):
    weight = weight_variable(ksize, .1)
    bias = bias_variable([ksize[-1]])
    conv = tf.nn.conv2d(input_data, weight, strides=strides, padding=padding)
    return tf.nn.relu(tf.nn.bias_add(conv, bias))


def weight_variable(shape, stddev):
    # 根据随机分布参数stddev,卷积核shape生成卷积核
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
