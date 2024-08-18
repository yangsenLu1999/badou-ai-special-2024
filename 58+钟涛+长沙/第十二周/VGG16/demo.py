from nets import vgg16
import tensorflow as tf
import utils

image = utils.load_image('test_data/dog.jpg')

input = tf.placeholder(tf.float32, [None,None, 3])

resize_image = utils.resize_image(input,[224,224])

net = vgg16.vgg16(resize_image)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

#加载模型
ckpt_filename = './model/vgg_16.ckpt'
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

pre = sess.run(tf.nn.softmax(net),feed_dict={input:image})

utils.print_prob(pre[0],'synset.txt')