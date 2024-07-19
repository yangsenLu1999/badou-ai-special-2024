from nets import VGG_nets
import tensorflow as tf
import numpy as np
import utils

#导入推理的图片
img=utils.load_image("./test_data/dog.ipg")

#占位，设置一个输入节点，并修改推理的图片size
inputs=tf.placeholder(tf.float32,[None,None,3])
resized_image=utils.resize_image(inputs,(224,224,3))

#建立网络结构
prediction=VGG_nets.vgg_16(resized_image)

#创建会话，导入模型，初始化变量，保存模型
sess=tf.Session()
ckpt_filename="./model/vgg_16.ckpt"
sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()
saver.restore(saver,ckpt_filename)

#利用训练模型进行推理
pro=tf.nn.softmax(prediction)
pre=sess.run(pro,feed_dict={inputs:img})

#打印结果
print("result: ")
utils.print_prob(pre[0], './synset.txt')


