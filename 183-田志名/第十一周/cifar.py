import tensorflow.compat.v1 as tf
import numpy as np
import time
import math
import cifar_data
import test
tf.disable_v2_behavior()

max_steps=4000               #最大更新次数，和epoch不同
batch_size=100
num_examples_for_eval=10000
data_dir="Cifar_data/cifar-10-batches-bin"

#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，
#在深度学习中，通常需要在模型初始化时给参数赋予一些小的随机值，这是为了打破完全对称性，确保网络层在训练开始时不是完全一样的，帮助模型更好地学习数据特征。
def variable_with_weight_loss(shape,stddev,w1):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))#tf.truncated_normal是一个TensorFlow函数，用于从截断的正态分布中生成随机数。
    if w1 is not None:
        weights_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weights_loss")  #tf.multiply函数允许用户对两个张量进行逐元素的乘法操作,name: 操作的名称，是一个可选参数，用于在计算图中标记这一操作。tf.nn.l2_loss(var),计算 var 的 L2 范数
        tf.add_to_collection("losses",weights_loss)   #用于将指定的张量（在这里是 weights_loss）添加到名为 "losses" 的集合中。tf.get_collection 获取该集合中的所有损失值
    return var

images_train,labels_train=cifar_data.inputs(data_dir=data_dir,batch_size=batch_size,flag=True)
images_test,labels_test=cifar_data.inputs(data_dir=data_dir,batch_size=batch_size,flag=None)

x=tf.placeholder(tf.float32,[batch_size,24,24,3])
y=tf.placeholder(tf.int32,[batch_size])

#搭建模型
#创建第一个卷积层
kernel1=variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)  #w1为0，表明不希望对变量施加任何L2正则化惩罚。
conv1=tf.nn.conv2d(x,kernel1,[1,1,1,1],padding="SAME")   #使用零填充，使得输出张量的尺寸与输入张量相同，[1,1,1,1]只需要关注中间两个即可
bias1=tf.Variable(tf.constant(0.0,shape=[64]))
relu1=tf.nn.relu(tf.nn.bias_add(conv1,bias1))            #如果偏置张量的形状与输入张量的最后一个维度不匹配，将会引发错误。
pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")   #(100, 12, 12, 64)

'''
pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
relu1: 输入张量，通常是一个经过激活函数（如ReLU）处理后的四维张量，形状为[batch_size, height, width, channels]。
ksize=[1,3,3,1]: 池化窗口的大小，这里表示在高度和宽度方向上分别使用大小为3x3的窗口进行池化操作。第一个和最后一个维度是1，表示在批次和通道维度上不进行池化。
strides=[1,2,2,1]: 池化窗口在每个维度上的步长，这里表示在高度和宽度方向上每次移动2个单位进行池化操作。同样，第一个和最后一个维度是1，表示在批次和通道维度上不进行移动。
padding="SAME": 填充方式，表示在进行池化操作时，使用全0填充
'''
#第二个卷积层
kernel2=variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)     #(100, 12, 12, 64)
conv2=tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding='SAME')
bias2=tf.Variable(tf.constant(0.0,shape=[64]))
relu2=tf.nn.relu(tf.nn.bias_add(conv2,bias2))
pool2=tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")   #(100, 6, 6, 64)

#全连接、
reshape=tf.reshape(pool2,[batch_size,-1])
dim=reshape.get_shape()[1].value

#建立第一个全连接层
weight1=variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)
fc_bias1=tf.Variable(tf.constant(0.1,shape=[384]))
fc_1=tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)

#建立第二个全连接层
weight2=variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
fc_bias2=tf.Variable(tf.constant(0.1,shape=[192]))
local4=tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)

#建立第三个全连接层
weight3=variable_with_weight_loss(shape=[192,10],stddev=1 / 192.0,w1=0.0)
fc_bias3=tf.Variable(tf.constant(0.1,shape=[10]))
result=tf.add(tf.matmul(local4,weight3),fc_bias3)

#计算损失，包括权重参数的正则化损失和交叉熵损失
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y,tf.int64))
weights_with_l2_loss=tf.add_n(tf.get_collection("losses"))   #它接受一个张量列表作为输入，并将它们逐个相加，返回一个新的张量。
loss=tf.reduce_mean(cross_entropy)+weights_with_l2_loss      #tf.reduce_mean函数用于计算张量的各个元素的平均值，最后结果是一个标量
train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)         #学习率为1e-3（即0.001），并将损失函数作为输入来最小化。

#函数tf.nn.in_top_k()计算预测值是否在前两个最高概率中，如果在，返回true
top_k_op=tf.nn.in_top_k(result,y,k=1)

init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    #启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
    tf.train.start_queue_runners()
    # 每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y: label_batch})
        duration = time.time() - start_time

        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)" % (
            step, loss_value, examples_per_sec, sec_per_batch)) #记录每个样本花费的时间，以及一个batch的时间
    # 计算最终的正确率
    num_batch = int(math.ceil(num_examples_for_eval / batch_size))  # math.ceil()函数用于求整
    true_count = 0
    total_sample_count = num_batch * batch_size

    # 在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={x: image_batch, y: label_batch})
        true_count += np.sum(predictions)
    #打印正确率信息
    print("accuracy = %.3f%%"%((true_count/total_sample_count) * 100))