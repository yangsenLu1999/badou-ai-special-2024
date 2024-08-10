import tensorflow as tf
import numpy as np
import math

# 创建slim对象
slim = tf.contrib.slim

def vgg_16(inputs,
           num_classes=10,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):
    with tf.variable_scope(scope, 'vgg_16', [inputs]):
        # 建立vgg_16的网络

        # conv1两次[3,3]卷积网络，输出的特征层为64，输出为(224,224,64)
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        # 2X2最大池化，输出net为(112,112,64)
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # conv2两次[3,3]卷积网络，输出的特征层为128，输出net为(112,112,128)
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        # 2X2最大池化，输出net为(56,56,128)
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(56,56,256)
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        # 2X2最大池化，输出net为(28,28,256)
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(28,28,512)
        # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        # # 2X2最大池化，输出net为(14,14,512)
        # net = slim.max_pool2d(net, [2, 2], scope='pool4')
        #
        # # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(14,14,512)
        # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        # # 2X2最大池化，输出net为(7,7,512)
        # net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net, 4096, [4, 4], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1000)
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')

        # 由于用卷积的方式模拟全连接层，所以输出需要平铺
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net

# 加载cifar-10数据集
def dataLoad(batch_size=64):
    (train_data, train_label), (test_data, test_label) = tf.keras.datasets.cifar10.load_data()
    # 数据集进行归一化
    train_data = train_data / 255
    test_data = test_data / 255
    # 将标签数据集从数组类型array修改成整形类型int
    train_label.astype(np.int)
    test_label.astype(np.int)
    train_data = tf.constant(train_data)
    train_label = tf.constant(train_label.reshape(-1))
    test_data = tf.constant(test_data)
    test_label = tf.constant(test_label.reshape(-1))

    training_dataset = tf.train.slice_input_producer([train_data, train_label])
    training_x = training_dataset[0]
    training_y = training_dataset[1]
    # training_x.set_shape([32, 32, 3])
    # training_y.set_shape([1])

    test_dataset = tf.train.slice_input_producer([test_data, test_label])
    test_x = test_dataset[0]
    test_y = test_dataset[1]
    # test_x.set_shape([32, 32, 3])
    # test_y.set_shape([1])

    images_train, labels_train = tf.train.shuffle_batch([training_x, training_y],
                                                batch_size=batch_size, num_threads=16,
                                                capacity=batch_size * 4, min_after_dequeue=batch_size * 2)
    images_test, labels_test = tf.train.batch([test_x, test_y],
                                              batch_size=batch_size, num_threads=16,
                                              capacity=batch_size*4)
    # # 将标签重塑为一维张量
    #     # labels_train = tf.reshape(labels_train,[batch_size])
    #     # labels_test = tf.reshape(labels_test,[batch_size])

    return images_train, labels_train, images_test, labels_test

if __name__ == "__main__":
    print('Tensorflow version = ', tf.__version__)  # have to be Tensorflow 2.6.0

    num_epochs = 100
    batch_size = 128

    images_train, labels_train, images_test, labels_test = dataLoad(batch_size)

    num_train = images_train.shape[0]
    num_val = images_test.shape[0]

    # 输入数据
    inputs = tf.placeholder(tf.float32, [batch_size, 32, 32, 3], name='inputs')
    labels = tf.placeholder(tf.int32, [batch_size], name='labels')

    model_output = vgg_16(inputs)

    # 2. 定义损失函数
    loss = tf.losses.sparse_softmax_cross_entropy(logits=model_output, labels=labels)

    # 3. 定义优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss)

    num_batch = int(num_train//batch_size)  # math.ceil()函数用于求整
    print(num_batch)
    saver = tf.train.Saver()
    # 4. 训练循环
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners() # 使用tf.train.shuffle_batch会有队列，在CPU上异步多线程进行训练，快很多

        for epoch in range(num_epochs):
            for _ in range(num_batch):
                image_batch, label_batch = sess.run([images_train, labels_train])
                _, loss_value = sess.run([train_op, loss], feed_dict={inputs: image_batch, labels: label_batch})
            print(f'Epoch {epoch}, Loss: {loss_value}')

        save_path = saver.save(sess, "VGGmodel.ckpt")
        print(f"Model saved in path: {save_path}")
        saver.restore(sess, "VGGmodel.ckpt")