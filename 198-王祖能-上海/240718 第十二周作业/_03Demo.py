'''
tensorflow1.15，对应的tensorflow-estimator也应该是1.15，需要重装一下它，用命令conda install tensorflow-estimator=1.15
'''
import tensorflow as tf
from nets import _01Vgg16
import _02Utils

img = _02Utils.load_img('./test_data/dog.jpg')

inputs = tf.placeholder(tf.float32, shape=[None, None, 3])  # 对输入的图片进行resize，使其shape满足(-1,224,224,3)
resize_img = _02Utils.resize_img(inputs, (224, 224))  # input with rank 4， 才能传到convolution
# 建立网络结构
prediction = _01Vgg16.vgg_16(resize_img)
# 载入模型
sess = tf.Session()
ckpt_filepath = './model/vgg_16.ckpt'  # 导入已提前储存的训练好的模型，进行推理
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()  # 将训练好的模型参数保存起来，以便以后进行验证或测试，仅训练过程需要
saver.restore(sess, ckpt_filepath)

pro = tf.nn.softmax(prediction)  # vgg16返回的net是经过卷积激活全连接后展平的结果
pre = sess.run(pro, feed_dict={inputs: img})  # 要传入的是原始数据，而不是处理后的resize_img

print('result:')
_02Utils.print_prob(pre[0], './synset.txt')
