from nets import VGG16
import tensorflow as tf
import numpy as np
import utils
import cv2

# 读取图片
img1 = utils.load_image("./test_data/dog.jpg")

# 对输入的图片进行resize，使其shape满足(-1,224,224,3)
inputs = tf.placeholder(tf.float32,[None,None,3])
resized_img = utils.resize_image(inputs, (224, 224))
# cv2.imshow('img',img1)
# cv2.waitKey(0)
# print('*********************')
# print(resized_img)
# print('*********************')


# 建立网络结构
prediction = VGG16.vgg_16(resized_img)
# print(prediction)

# 载入模型
sess = tf.Session()
ckpt_filename = './model/vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

# 最后结果进行softmax预测
pro = tf.nn.softmax(prediction)
pre = sess.run(pro,feed_dict={inputs:img1})

# 打印预测结果
print("result: ")
utils.print_prob(pre[0], './synset.txt')
