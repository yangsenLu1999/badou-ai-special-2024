'''推理的demo'''

import model_vgg16 as model
import tensorflow as tf
import numpy as np
import utils

# 读取图片
image = utils.load_image('./test_data/dog.jpg')

# 调整图片尺寸
inputs = tf.placeholder(tf.float32, [None, None, 3])
resized_image = utils.resize_img(inputs, [224, 224])

# 构建模型
prediction = model.vgg_16(resized_image)

# 载入模型
sess = tf.Session()
kpt_filename = 'vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, kpt_filename)

# 最后的结果预测
pro = tf.nn.softmax(prediction)
pre = sess.run(pro, feed_dict={inputs:image})

# 打印结果
print('result:')
utils.print_prob(pre[0], './synset.txt')
