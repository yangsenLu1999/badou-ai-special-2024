'''
定义VGG16模型网络结构
'''
import tensorflow as tf

slim = tf.contrib.slim  # tf一些训练等重复代码的封装
# dropout根据训练使用，推理不使用确定；scope是变量空间名称vgg16，spatial_squeeze等效flatten展平操作


def vgg_16(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.5,
           spatial_squeeze=True, scope='vgg_16'):

    with tf.variable_scope(scope, 'vgg_16', [inputs]):  # name_or_scope为None时，用‘vgg_16'
        # conv1为2次3x3卷积，输出特征层为 64，图片格式的[224, 224, 3]变为[224, 224, 64]，最大池化[2, 2]后变为[112, 112, 64]
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')  # 生成’conv1/conv1_1‘
        net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 降维，过滤区域内冗余的不重要的特征
        # conv2为2次3x3的卷积，输出特征层128，变为[112, 112, 128]，最大池化后变[56， 56, 128]
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # conv3为3次3x3的卷积，输出特征层为256，变为[56, 56, 256], [2, 2]最大池化后变为[28, 28, 256]
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # conv4为3次3x3的卷积，输出特征层为512，变为[28, 28, 512], [2, 2]最大池化后变为[14, 14, 512]
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # conv5为3次3x3的卷积， 输出特征层为512， 变为[14, 14, 512], [2, 2]最大池化后变为[7, 7, 512]
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # 以上共13层，接下来进入全连接层，而用卷积的方式模拟，即卷积核大小和输入一致，输出[1, 1, 4096]
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='drop6')
        # 一定概率丢弃神经元，防止过拟合。slim.dropout只有两个必传参数，is_training是可选参数，要传值必须要is_training = 形式
        net = slim.conv2d(net, 4096, [1, 1], padding='VALID', scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='drop7')  # 只有训练过程才进行dropout,推理是不考虑的
        net = slim.conv2d(net, num_classes, [1, 1], padding='VALID', activation_fn=None, normalizer_fn=None, scope='fc8')

        # 输出数据是多维张量形式，需要展平平铺开
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8_squeeze')
        return net
