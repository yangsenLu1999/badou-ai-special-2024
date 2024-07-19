from  model import vgg_16
import tensorflow.compat.v1 as tf
import numpy as np
import utils

tf.disable_v2_behavior()

# 读图resize 使其满足（-1，224，224，3）
img1 = utils.load_image("./test_data/dog.jpg")
inputs =tf.placeholder(tf.float32,[None,None,3])
resized_img = utils.resize_img(inputs,(224,224))

#
prediction = vgg_16(resized_img)



sess = tf.Session()
# 获取预训练模型
ckpt_filename = './vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

saver.restore(sess,ckpt_filename)

pro = tf.nn.softmax(prediction)
pre  = sess.run(pro,feed_dict={inputs:img1})


print('result:')
utils.print_prob(pre[0],'./synset.txt')


