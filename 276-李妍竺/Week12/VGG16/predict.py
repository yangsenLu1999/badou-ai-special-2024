# 预测
from nets import vgg16
import cv2
import tensorflow as tf
import numpy as np
import utils

# 读取待测图片
img = utils.load_image('./test_data/dog.jpg')

# 处理图片形状，使其shape满足(-1,224,224,3)
inputs = tf.placeholder(tf.float32,[None,None,3])
img_resize = utils.resize_image(inputs,(224,224))


# 建立网络结构
prediction = vgg16.vgg_16(img_resize)

# 载入模型
sess = tf.Session()

train_data = './model/vgg_16.ckpt'    #训练数据
sess.run(tf.global_variables_initializer())
'''
tf.train.Saver(): 这个函数被用于创建一个 Saver 对象，该对象可以保存和加载 TensorFlow 模型中的变量。它会默认使用会话中定义的变量，除非你在构造函数中明确地指定变量。
saver.restore(sess, train_data): 这个函数被用于在 sess 中恢复先前存储的模型。train_data 参数是模型的路径，它通常是一个检查点（checkpoint）文件，其中包含先前训练的模型参数。
'''
saver = tf.train.Saver()
saver.restore(sess,train_data)

# 进行softmax预测
pro = tf.nn.softmax(prediction)
pre = sess.run(pro,feed_dict={inputs:img})

# 预测结果
print('result:')

utils.print_prob(pre[0], './synset.txt')  # pre[0]：第一张图的意思