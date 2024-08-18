import tensorflow as tf

# create slim model
# tf.contrib.slim is a lightweight library for defining, training and evaluating complex models in TensorFlow.
slim = tf.contrib.slim 

def vgg16(inputs, 
          num_classes=1000, 
          is_training=True, 
          dropout_rate=0.5, 
          spatial_squeeze=True, 
          scope='vgg_16'):
    with tf.variable_scope(scope, 'vgg_16', [inputs]):

        # conv1 twice [3, 3] 64
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # conv2 twice [3, 3] 128
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # conv3 three times [3, 3] 256
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool(net, [2, 2], scope='pool3')

        # conv4 three times [3, 3] 512
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool(net, [2, 2], scope='pool4')

        # conv5 three times [3, 3] 512
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool(net, [2, 2], scope='pool5')

        # using conv2d instead of fully_connected layers
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_rate, is_training=is_training, scope='dropout6')

        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_rate, is_training=is_training, scope='dropout7')

        net = slim.conv2d(net, num_classes, [1, 1], 
                          activation_fn=None, normalizer_fn=None, scope='fc8')

        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net
        
