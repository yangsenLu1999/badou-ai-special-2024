from net import vgg16
import tensorflow as tf
import numpy as np
import utils

img1 = utils.load_image(r"E:\data\dabou\dog.jpg")

inputs = tf.placeholder(tf.float32, [None,None,3])
resize_img = utils.resize_image(inputs, (224, 224))

prediction = vgg16.vgg_16(resize_img)

sess = tf.Session()
ckpt_filename = r'E:\data\dabou\vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

pro = tf.nn.softmax(prediction)
pre = sess.run(pro, feed_dict={inputs:img1})

print('result: ')
utils.print_prob(pre[0],r'E:\data\dabou\synset.txt')

