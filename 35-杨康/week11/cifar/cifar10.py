import tensorflow as tf
import time
import numpy as np
import math
import Cifar10_data

batch_size = 100
max_steps = 4000
num_examples_for_eval = 10000
data_dir = "cifar_data/cifar-10-batches-bin"

# 1.加载数据
image_train, label_train = Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=True)
image_test, label_test = Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=None)

# 2.定义权重损失
def var_weight_loss(shape,stddev,w1):
    var = tf.Variable(tf.truncated_normal(shape=shape,stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),w1,name='weight_loss')
        tf.add_to_collection('losses',weight_loss)
    return var

# 3.创建x和y_的placeholder
x = tf.placeholder(np.float32,[batch_size,24,24,3])
y_ = tf.placeholder(np.int32,[batch_size])

# 4.创建第一个卷积层
kernel1 = var_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)
conv1 = tf.nn.conv2d(x,kernel1,[1,1,1,1],padding='SAME')
baise1 = tf.Variable(tf.constant(0.0,shape=[64]))
relu1 = tf.nn.relu(conv1 + baise1)
pool1 = tf.nn.max_pool2d(relu1,[1,3,3,1],[1,2,2,1],padding='SAME')

# 5.创建第二个卷积层
kernel2 = var_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
conv2 = tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding='SAME')
baise2 = tf.Variable(tf.constant(0.1,shape=[64]))
relu2 = tf.nn.relu(conv2 + baise2)
pool2 = tf.nn.max_pool2d(relu2,[1,3,3,1],[1,2,2,1],padding='SAME')

# 6.创建第一个全连接层,全连接之前需要将pool2数据转成一维数据
reshape = tf.reshape(pool2,shape=[batch_size,-1])  #-1代表将pool2的三维结构拉直为一维结构
dim = reshape.get_shape()[1].value #get_shape()[1].value表示获取reshape之后的第二个维度的值

weight1 = var_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)
fc_baise1 = tf.Variable(tf.constant(0.1,shape=[384]))
fc1 = tf.nn.relu(tf.matmul(reshape,weight1) + fc_baise1)  # wx + b

# 7.创建第二个全连接层
weight2 = var_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
fc_baise2 = tf.Variable(tf.constant(0.1,shape=[192]))
fc2 = tf.nn.relu(tf.matmul(fc1,weight2) + fc_baise2)  # wx + b

# 8.创建第三个全连接层
weight3 = var_weight_loss(shape=[192,10],stddev=0.04,w1=0.004)
fc_baise3 = tf.Variable(tf.constant(0.1,shape=[10]))
result = tf.matmul(fc2,weight3) + fc_baise3  # wx + b

# 9.计算损失 交叉熵损失：tf.nn.sparse_softmax_cross_entropy_with_logits() + 权重损失
entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y_,tf.int64)) #如果标签数据是one-hot格式的可以用函数tf.nn.softmax_cross_entropy_with_logits()
weight_loss = tf.add_n(tf.get_collection('losses'))
loss = tf.reduce_mean(entropy_loss) + weight_loss

# 10.设置反向传播方法
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 11.计算准确率 tf.nn.in_top_k(result,y_,1)
top_k = tf.nn.in_top_k(result,y_,1)

init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
    tf.train.start_queue_runners()

    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([image_train, label_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})
        duration = time.time()-start_time
        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)"%(step,loss_value,examples_per_sec,sec_per_batch))

#计算最终的正确率
    num_batch=int(math.ceil(num_examples_for_eval/batch_size))  #math.ceil()函数用于向上求整
    true_count=0
    total_sample_count=num_batch * batch_size

    #在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch, label_batch = sess.run([image_test, label_test])
        predictions = sess.run([top_k], feed_dict={x: image_batch, y_: label_batch})
        true_count += np.sum(predictions)

    #打印正确率信息
    print("accuracy = %.3f%%"%((true_count/total_sample_count) * 100))