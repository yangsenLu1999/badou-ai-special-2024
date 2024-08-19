import tensorflow as tf
import numpy
import time
import math
import numpy as np
import Cifar10_data  # 不能数字开头，不能包含空格，引入编写的数据处理py

data_dir = 'cifar_data/cifar-10-batches-bin'  # 读数据与’data_batch_%d.bin‘连接
batch_size = 100
max_step = 2000
examples_for_eval = 10000

img_train, label_train = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)  # 训练数据是需要图像增强，包络更多样本，进入if语句
img_test, label_test = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=False)  # 推理数据不需要，进入else语句
# 用于训练或推理提供输入数据和对应标签。注意，由于定义全连接网络的时候用到了batch_size，所以第一个参数不是None，而是batch_size
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])  # 因为数据是按照batch_size取出来的，tensor_size = [batch_size, 24, 24, 3]
y = tf.placeholder(tf.int32, [batch_size])  # 标签的tensor_size = [batch_size]

'''
#   variable_with_weight_loss()函数作用是：
#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss.
'''


def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))  # 就是权重矩阵
    # 因为权重w 和偏差 b 都是可训练参数，所以需要用tf.Variable()定义训练
    # 函数是一种“截断”方式生成正态分布随机值，“截断”意思指生成的随机数值与均值的差不能大于两倍中误差，否则会重新生成。
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weights_loss')
        tf.add_to_collection('losses', weights_loss)
    return var
    pass


# 定义第一个卷积层， shape=[kh, kw, ci, co]卷积的高、宽、输入维度、输出维度
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding='SAME')  # 步长表示上下左右四个方向都为1，一般一样。padding有valid, same, full三种
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 定义第二个卷积层
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 128], stddev=0.05, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.0, shape=[128]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 全连接层前，将结果转换为一维向量
reshape = tf.reshape(pool2, shape=[batch_size, -1])  # 应该有batch_size？？？-1代表将pool2的三维结构拉直为一维结构
dim = reshape.get_shape()[1].value  # get_shape()[1].value表示获取reshape之后的第二个维度的值

# 建立第一个全连接层
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.02, w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)
# 建立第二个全连接层
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.02, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
fc2 = tf.nn.relu(tf.matmul(fc1, weight2) + fc_bias2)
# 建立第三个全连接层
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=0.002, w1=0.0)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
result = tf.matmul(fc2, weight3) + fc_bias3

# 计算损失包括权重参数的正则化损失和交叉熵损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y, tf.int64))  # 多分类交叉熵计算函数。它适用于每个类别相互独立且排斥的情况，例如一幅图只能属于一类，而不能同时包含一条狗和一头大象。

weights_with_l2_loss = tf.add_n(tf.get_collection('losses'))
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss
# tf.add_n实现一个列表的元素的相加
# tf.get_collection获取key集合中所有元素，返回列表。顺序依变量放入集合的先后定。scope可选，表示名称空间（名称域），指定就返回名称域中‘key’的变量列表，不指定则返回所有变量
train_op = tf.train.AdamOptimizer(0.002).minimize(loss)

top1_op = tf.nn.in_top_k(result, y, 1)
# predictions：预测的结果，预测矩阵大小为样本数×标注的label类的个数的二维矩阵。
# targets：实际的标签，大小为样本数。
# k：每个样本的预测结果的前k个最大的数里面是否包含targets预测中的标签，一般都是取1，即取预测最大概率的索引与标签对比。

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners()  # data处理图像增强时使用了train.shuffle_batch函数通过参数num_threads配置16个线程用于batch

    for step in range(max_step):  # 每隔100step计算显示当前的loss,训练的样本数量，以及训练一个batch的时间
        start_time = time.time()
        img_batch, label_batch = sess.run([img_train, label_train])  # fetch获取多个数据需要, 每次只是取出来batch_size个数据
        _, loss_value = sess.run([train_op, loss], feed_dict={x: img_batch, y: label_batch})
        batch_time = time.time() - start_time

        if (step+1) % 100 == 0:
            time_per_batch = float(batch_time)
            examples_per_sec = batch_size / batch_time
            print('step:%d, loss:%.2f (%.1f examples/sec, %.3f secs/batch' % ((step+1), loss_value, examples_per_sec, time_per_batch))
    # 下面进行测试或推理，判断正确率
    num_batches = int(math.ceil(examples_for_eval / batch_size))
    true_count = 0
    total_examples = num_batches * batch_size  # 由于每次取出来batchsize个数据，total表示推理总数据量
    for j in range(num_batches):
        img_batch, label_batch = sess.run([img_test, label_test])
        predictions = sess.run([top1_op], feed_dict={x: img_batch, y: label_batch})
        true_count += np.sum(predictions)
    print('accuracy = %.3f%%' % ((true_count / total_examples) * 100))
