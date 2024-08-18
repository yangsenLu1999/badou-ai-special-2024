import tensorflow as tf
import numpy as np
import math
import cifar10_loaddata
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

steps=4000
batch_size=100
epoch_num_for_eva=10000
data_path='F:/历史工作文档/学习资料/python/八斗2024精品班/cifar10/cifar_data/cifar-10-batches-bin'

#创建一个variable_with_weight_loss()函数，该函数的作用是：
#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，
def variable_with_weight_loss(shape,stddev,w1): #shape代表--[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
        var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))  #tf.truncated_normal--产生一个符合正态分布（标准差为sttdnv）的,尺寸为shape的张量
        if w1 is not None:
            weights_loss=tf.multiply(tf.nn.l2_loss(var),w1)
            tf.add_to_collection('losses',weights_loss)
        return var

#使用上一个文件里面已经定义好的文件序列读取函数读取训练数据文件和测试数据从文件.
#其中训练数据文件进行数据增强处理，测试数据文件不进行数据增强处理
imgs_train,labels_train=cifar10_loaddata.input_datas(data_path=data_path,batch_size=batch_size,is_distored=True)
imgs_test,labels_test=cifar10_loaddata.input_datas(data_path=data_path,batch_size=batch_size,is_distored=None)

#创建x和y_两个placeholder，用于在训练或评估时提供输入的数据和对应的标签值。
#要注意的是，由于以后定义全连接网络的时候用到了batch_size，所以x中，第一个参数不应该是None，而应该是batch_size
x=tf.placeholder(tf.float32,[batch_size,24,24,3])
y=tf.placeholder(tf.int32,[batch_size])

#创建第一个卷积层
kennel1=variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-3,w1=0.0)
conv1=tf.nn.conv2d(x,kennel1,[1,1,1,1],padding='SAME')
b1=tf.Variable(tf.constant(0.0,shape=[64]))   #tf.constant--定义一个常量
relu1=tf.nn.relu(tf.nn.bias_add(conv1,b1))
pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

#第二个卷积层
kennel2=variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-3,w1=0.0)
conv2=tf.nn.conv2d(x,kennel1,[1,1,1,1],padding='SAME')
b2=tf.Variable(tf.constant(0.0,shape=[64]))   #tf.constant--定义一个常量
relu2=tf.nn.relu(tf.nn.bias_add(conv2,b2))
pool2=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

#全连接的输入必须是一维的  因此reshape成一维
reshape=tf.reshape(pool2,[batch_size,-1])
dim=reshape.get_shape()[1].value   ##get_shape()[1].value表示获取reshape之后的第二个维度的值

#第一个全连接层
weight1=variable_with_weight_loss(shape=[dim,384],stddev=0.4,w1=0.004)
fc_b1=tf.Variable(tf.constant(0.1,shape=[384]))
fc1=tf.nn.relu(tf.nn.bias_add(tf.matmul(reshape,weight1),fc_b1))


#第二个全连接层
weight2=variable_with_weight_loss(shape=[384,192],stddev=0.4,w1=0.004)
fc_b2=tf.Variable(tf.constant(0.1,shape=[192]))
fc2=tf.nn.relu(tf.nn.bias_add(tf.matmul(fc1,weight2),fc_b2))

#第三个全连接层
weight3=variable_with_weight_loss(shape=[192,10],stddev=0.4,w1=0.004)
fc_b3=tf.Variable(tf.constant(0.1,shape=[10]))
fc3=tf.nn.relu(tf.nn.bias_add(tf.matmul(fc2,weight3),fc_b3))

#计算损失，包括权重参数的正则化损失和交叉熵损失
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc3,labels=tf.cast(y,tf.int64))
#tf.add_n 列表元素相加
weight_loss=tf.add_n(tf.get_collection('losses'))
loss=tf.reduce_mean(cross_entropy)+weight_loss

train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)  #采用Adam进行优化，学习率为 e-3
#函数tf.nn.in_top_k()用来计算输出结果中top k的准确率，函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
top_k_op=tf.nn.in_top_k(fc3,y,1)
init_op=tf.global_variables_initializer()  #tf.global_variables_initializer()能够将所有的变量一步到位的初始化
with tf.Session() as sess:
    sess.run(init_op)
    #启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
    tf.train.start_queue_runners()

#每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
    for step in range (steps):
        start_time=time.time()
        image_batch,label_batch=sess.run([imgs_train,labels_train])
        _,loss_value=sess.run([train_op,loss],feed_dict={x:image_batch,y:label_batch})
        duration=time.time() - start_time

        if step % 100 == 0:
            examples_per_sec=batch_size / duration
            sec_per_batch=float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)"%(step,loss_value,examples_per_sec,sec_per_batch))

#计算最终的正确率
    num_batch=int(math.ceil(epoch_num_for_eva/batch_size))  #math.ceil()函数用于求整
    true_count=0
    total_sample_count=num_batch * batch_size

    #在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch,label_batch=sess.run([imgs_test,labels_test])
        predictions=sess.run([top_k_op],feed_dict={x:image_batch,y:label_batch})
        true_count += np.sum(predictions)

    #打印正确率信息
    print("accuracy = %.3f%%"%((true_count/total_sample_count) * 100))

