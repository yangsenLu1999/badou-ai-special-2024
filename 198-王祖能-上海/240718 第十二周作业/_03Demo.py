'''
tensorflow1.15，对应的tensorflow-estimator也应该是1.15，需要重装一下它，用命令conda install tensorflow-estimator=1.15
'''
import tensorflow as tf
from nets import _01Vgg16
import _02Utils

img = _02Utils.load_img('./test_data/dog.jpg')
img1 = _02Utils.load_img('test_data/table.jpg')

inputs = tf.placeholder(tf.float32, shape=[None, None, 3])  # 未知量占位符，只限制通道数，不限尺寸，读入后经过后续的resize形成【224， 224】
resize_img = _02Utils.resize_img(inputs, (224, 224))  # 对输入的图片进行resize，使其shape满足(-1,224,224,3)
# 建立网络结构
net = _01Vgg16.vgg_16(resize_img)  # 返回squeee的1000个数, vgg16返回的net是经过卷积激活全连接后展平的结果
prob = tf.nn.softmax(net)
# 载入模型
with tf.Session() as sess:
    ckpt_filepath = './model/vgg_16.ckpt'  # 导入已提前储存的训练好的模型，ckpt是怎么来的，alexnet是models.save_weights(last1.h5)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()  # 将训练好的模型参数保存起来，以便以后进行验证或测试，仅训练过程需要
    saver.restore(sess, ckpt_filepath)

    pred1 = sess.run(prob, feed_dict={inputs: img})  # 要传入的是原始数据，而不是处理后的resize_img，resie_img取决于未知量placeholder
    pred2 = sess.run(prob, feed_dict={inputs: img1})
    # print('result:', [pred1, pred2], pred1.shape, pred2.shape)  # shape 是（1， 1000）的np.array
    print(pred1.shape, pred1[0].shape, type(pred1[0]))  # shape 是（1000, ）的np.array

    print(_02Utils.answer('synset.txt', pred1[0]))  # 这里为什么用pred1[0]， pred是(1,1000)，调用pred[0]去除第一维度，即变为（1000，）的list，可以按照index调用
    print(_02Utils.answer('synset.txt', pred2[0]))
