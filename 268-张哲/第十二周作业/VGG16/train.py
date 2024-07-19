from net import VGG16
import tensorflow as tf
import numpy as np
import utils

# 读取图片
img1 = utils.load_image("./test_data/dog.jpg")

# 对输入的图片进行resize，使其shape满足(-1,224,224,3)
input = tf.placeholder(tf.float32,[None,None,3])
resized_img = utils.resize_image(input,(224,224))

# 建立网络结构
prediction = VGG16.vgg_16(resized_img)

sess = tf.Session()
ckpt_filename = './model/vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

pro = tf.nn.softmax(prediction)
pre = sess.run(pro,feed_dict={input:img1})

# 打印预测结果
print("result: ")
utils.print_prob(pre[0], './synset.txt')

