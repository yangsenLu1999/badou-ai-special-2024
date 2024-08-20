from nets import vgg16
import tensorflow as tf
import numpy as np
import utils

# 读取图片
img1 = utils.load_image('./test_data/table.jpg')
# 对输入的图片进行resize，使其shape满足(-1,224,224,3)
inputs = tf.placeholder(tf.float32, [None, None, 3])
resized_image = utils.resize_image(inputs, (224, 224))

# 建立网络结构
prediction = vgg16.vgg_16(resized_image)

# 载入模型
sess = tf.Session()
ckpt_filename = './model/vgg_16.ckpt'  # Checkpoint文件是TensorFlow用来保存和恢复模型参数的一种格式。
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()  # Saver类有很多方法，如果是训练可以用saver.run分布存储模型
saver.restore(sess, ckpt_filename)  # 恢复模型到当前的会话（sess）中

# 最后结果进行softmax预测
# softmax不需要训练，所以可以写在网络结构里也可以写在外面
pro = tf.nn.softmax(prediction)
pre = sess.run(pro, feed_dict={inputs: img1})
# print(pre)
# print(pre.shape)  # -->(1,1000)
# print(pre[0])  # 这里是把推理的唯一一张图片所对应1000个类别的概率提取出来了
# 打印预测结果
print("result: ")
utils.print_prob(pre[0], './synset.txt')
