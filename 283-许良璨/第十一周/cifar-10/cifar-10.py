import tensorflow as tf
import cifar10_data
import time
import math
import numpy as np


max_step=4000
num_examples_for_eval=10000
num_examples_pre_epoch_for_train=50000
batch_size=100
data_dir="Cifar_data/cifar-10-batches-bin"

def wight_and_loss(shape,stddev,w1):
    var=tf.Variable(tf.truncated_normal(shape=shape,stddev=stddev))
    if w1 != None:
        L2_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weights_loss")
        tf.add_to_collection("losses",L2_loss)
    return var

train_lable,train_img=cifar10_data.input(datadir=data_dir,batch_size=batch_size,distorted=True)
test_lable,test_img=cifar10_data.input(datadir=data_dir,batch_size=batch_size,distorted=False)

x=tf.placeholder(tf.float32,[batch_size,24,24,3])
y=tf.placeholder(tf.int32,[batch_size])

kenel1=wight_and_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)
conv1=tf.nn.conv2d(x,kenel1,[1,1,1,1],padding="SAME")
B=tf.Variable(tf.constant(0.0,shape=[64]))
relu1=tf.nn.relu(tf.nn.bias_add(conv1,B))
pool1=tf.nn.max_pool(relu1,[1,3,3,1],[1,2,2,1],padding="SAME")

kenel2=wight_and_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
conv2=tf.nn.conv2d(pool1,kenel2,[1,1,1,1],padding="SAME")
B2=tf.Variable(tf.constant(0.1,shape=[64]))
relu2=tf.nn.relu(tf.nn.bias_add(conv2,B2))
pool2=tf.nn.max_pool(relu2,[1,3,3,1],[1,2,2,1],padding="SAME")

reshaped=tf.reshape(pool2,[batch_size,-1])
dim=reshaped.get_shape()[1].value

wight1=wight_and_loss([dim,384],stddev=0.04,w1=0.004)
FC_bias1=tf.Variable(tf.constant(0.1,shape=[384]))
FC_relu1=tf.nn.relu(tf.matmul(reshaped,wight1)+FC_bias1)

wight2=wight_and_loss([384,192],stddev=0.04,w1=0.004)
FC_bias2=tf.Variable(tf.constant(0.1,shape=[192]))
FC_relu2=tf.nn.relu(tf.matmul(FC_relu1,wight2)+FC_bias2)

wight3=wight_and_loss([192,10],stddev=1 / 192.0,w1=0.0)
FC_bias3=tf.Variable(tf.constant(0.1,shape=[10]))
result=tf.add(tf.matmul(FC_relu2,wight3),FC_bias3)

cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y,tf.int64))
weight_with_l2_loss=tf.add_n(tf.get_collection("losses"))
loss=tf.reduce_mean(cross_entropy)+weight_with_l2_loss

train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)

top_K_op=tf.nn.in_top_k(result,y,1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners()

    for step in range(max_step):
        start_time=time.time()
        lable_batch, img_batch =sess.run([train_lable,train_img])
        _,loss_value=sess.run([train_op,loss],feed_dict={x:img_batch,y:lable_batch})
        duration=time.time()-start_time

        if step % 100 ==0:
            examples_per_sec=batch_size/duration
            sce_per_batch=float(duration)
            time1=sce_per_batch*(max_step-step)
            hour1=time1/360

            print("step:%d,loss:%.2f(%.1f examples/sec;%.3f sec/batch,need %.2f hours)"%(step,loss_value,examples_per_sec,sce_per_batch,hour1))

    num_batch=int(math.ceil(num_examples_for_eval/batch_size))
    true_count=0


    for i in range(num_batch):
        lable_batch, img_batch = sess.run([test_lable,test_img])
        predections=sess.run([top_K_op], feed_dict={x: img_batch, y: lable_batch})
        true_count+=np.sum(predections)

    print("accuracy = %.3f%%"%((true_count/num_examples_for_eval) * 100))
