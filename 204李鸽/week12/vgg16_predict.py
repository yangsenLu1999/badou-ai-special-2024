# 这里是推理完整过程，训练可用上次alexnet的，改成vgg16就行
from nets import vgg16  # 导入自定义的 vgg16 模块
import tensorflow as tf
import numpy as np
import utils  # 导入自定义的 utils 模块，通常用于工具函数，可能包括图像处理、参数加载等功能

# 读取图片
img1 = utils.load_image("./test_data/dog.jpg")

# 对输入的图片进行resize，使其shape满足(-1,224,224,3)
inputs = tf.placeholder(tf.float32,[None,None,3])
'''创建一个 TensorFlow 占位符 inputs，用于接受输入图像。
参数 [None, None, 3] 表示高度和宽度可以是任意值，但图像必须是 RGB（3 通道）'''
resized_img = utils.resize_image(inputs, (224, 224))
'''调用 utils.resize_image 函数调整输入图像的大小为 224x224，以满足 VGG16 模型的输入要求'''
# 建立网络结构
prediction = vgg16.vgg_16(resized_img)
'''调用 vgg16 模块中的 vgg_16 函数，使用调整后的图像 resized_img 来建立 VGG16 网络结构。
prediction 表示模型的输出，通常是各类的 raw scores（未经过激活的输出）'''

# 载入模型
sess = tf.Session()  # sess = tf.Session(): 创建 TensorFlow 会话，用于执行图计算
ckpt_filename = './model/vgg_16.ckpt'   # 已训练模型的存储位置。
sess.run(tf.global_variables_initializer())   # 初始化所有 TensorFlow 变量，此步骤在加载模型之前必需
saver = tf.train.Saver()   # 创建 Saver 对象，这是用于保存和恢复 TensorFlow 模型的工具
saver.restore(sess, ckpt_filename)  # 从指定的检查点文件恢复模型参数

# 最后结果进行softmax预测
pro = tf.nn.softmax(prediction)  # pro 将包含每个类的预测概率。
pre = sess.run(pro,feed_dict={inputs:img1})
'''在会话中运行操作，feed_dict 将实际输入的图像 img1 提供给占位符 inputs，并计算得到预测结果 pre'''

# 打印预测结果
print("result: ")
utils.print_prob(pre[0], './synset.txt')
'''utils.print_prob 函数，以处理和输出预测结果。pre[0] 是第一个图像的预测概率向量，
'./synset.txt' 是一个文本文件，包含类别名称，通常用于显示预测结果对应的标签'''
