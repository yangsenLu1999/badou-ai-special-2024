import tensorflow as tf

slim=tf.contrib.slim
def vgg_16(inputs,
           num_class=1000,
           is_training =True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'
):
    with tf.variable_scope(scope,'vgg_16',[inputs]):
        #创建两个卷积层，输出为64通道，3x3的卷积核，命名为卷积层1
        net=slim.repeat(inputs,2,slim.conv2d,64,[3,3],scope="conv1")
        #创建一个池化层，2x2最大池化，命名为池化层1
        net=slim.max_pool2d(net,[2,2],scope='pool1')

        ##创建两个卷积层，输出为128通道，3x3的卷积核，命名为卷积层2
        net=slim.repeat(net,2,slim.conv2d,128,[3,3],scope='conv2')
        ##创建一个池化层，2X2最大池化，命名为池化层2
        net=slim.max_pool2d(net,[2,2],scope="pool2")

        #创建3个卷积层，输出为256通道，3x3的卷积核，命名为卷积层3
        net=slim.repeat(net,3,slim.conv2d,256,[3,3],scope="conv3")
        ##创建一个池化层，2X2最大池化，命名为池化层3
        net=slim.max_pool2d(net,[2,2],scope="pool3")

        #创建3个卷积层，输出为512通道，3x3的卷积核，命名为卷积层4
        net=slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope="conv4")
        ##创建一个池化层，2X2最大池化，命名为池化层4
        net=slim.max_pool2d(net,[2,2],scope="pool4")

        #创建3个卷积层，输出为512通道，3x3的卷积核，命名为卷积层5
        net=slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope="conv5")
        ##创建一个池化层，2X2最大池化，命名为池化层5
        net=slim.max_pool2d(net,[2,2],scope="pool5")

        #将全连接层改为卷积层，输出为（1，1，4096），7X7的卷积核
        net=slim.conv2d(net,4096,[7,7],padding="VALID",scope="fc6")
        net=slim.dropout(net,dropout_keep_prob,is_training=is_training,scope="dropout6")

        # 将全连接层改为卷积层，输出为（1，1，4096），1X1的卷积核
        net=slim.conv2d(net,4096,[1,1],padding="VALID",scope='fc7')
        net=slim.dropout(net,dropout_keep_prob,is_training=is_training,scope="dropout7")
        #输出为（1，1，1000），1X1的卷积核
        net=slim.conv2d(net,num_class,[1,1],activation_fn=None,normalizer_fn=None, scope="fc8")

        #需要将卷积结果进行平铺
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net
