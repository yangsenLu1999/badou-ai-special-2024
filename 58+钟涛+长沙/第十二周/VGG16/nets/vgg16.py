import tensorflow as tf

slim = tf.contrib.slim

def vgg16(inputs,dropout_keep_prob = 0.5, is_training=True, num_classes = 1000, scope='vgg_16',spatial_squeeze = True):

    with tf.variable_scope(scope,'vgg_16',[inputs]):

        # 2次卷积，输出特征为64，输出为 [224,224,64]
        net = slim.repeat(inputs,2 , slim.conv2d,64,[3,3],scope='conv1')
        #[2X2]池化，输出为[112,112,64]
        net = slim.max_pool2d(net,[2,2],scope='pool1')

        #2次卷积，输出为[112,112,128]
        net = slim.repeat(net, 2, slim.conv2d,128,[3,3],scope='conv2')
        #[2X2] 池化，输出为[56,56,128]
        net = slim.max_pool2d(net,[2,2], scope='pool2')

        # 2次卷积，输出为[56,56,256]
        net = slim.repeat(net, 3, slim.conv2d,256,[3,3],scope='conv3')
        #池化，输出为[28,28,256]
        net = slim.max_pool2d(net,[2,2], scope='pool3')

        #卷积3次，输出为[28,28,512]
        net = slim.repeat(net,3,slim.conv2d, 512,[3,3],scope='conv4')
        #池化，输出为[14,14,512]
        net = slim.max_pool2d(net,[2,2],scope='pool4')

        #3次卷积，输出为[14,14,512]
        net = slim.repeat(net,3,slim.conv2d, 512,[3,3],scope='conv5')
        # 池化，输出为[7,7,512]
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net,4096,[7,7],padding='VALID',scope='fc6')
        # dropout
        net= slim.dropout(net, dropout_keep_prob,scope='dropout6',is_training=is_training)

        #利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net,4096,[1,1],scope='fc7')
        # dropout
        net = slim.dropout(net, dropout_keep_prob, scope='dropout7', is_training=is_training)

        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1000)
        net=slim.conv2d(net, num_classes,[1, 1], activation_fn=None,normalizer_fn=None,scope='fc8')

        # 卷积模拟全连接，所以，输出需要平铺
        if spatial_squeeze:
            net = tf.squeeze(net, [1,2], name='fc8/squeeze')
        return net





