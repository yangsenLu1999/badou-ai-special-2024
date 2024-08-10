from computervision.vgg import myvgg16
import tensorflow as tf
import numpy as np
import imageutils

image = imageutils.read_image('test_data/cat.4628.jpg')
# 对输入的图片进行resize，使其shape满足(-1,224,224,3)
inputs = tf.placeholder(tf.float32, [None, None, 3])
resize_image = imageutils.resize_image(inputs, (224, 224))

# 建立网络结构
net = myvgg16.my_vgg_16(resize_image)
# 载入模型
session = tf.Session()
mode_fileName = './model/vgg_16.ckpt'
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, mode_fileName)
# 最后结果进行softmax预测
result = tf.nn.softmax(net)
pre = session.run(result, feed_dict={inputs:image})
# 打印预测结果
print("result: ")
imageutils.print_prob(pre[0], './synset.txt')

