import vgg16_tf
import utils
import tensorflow as tf

img1 = utils.crop_image('dog.jpg')

inputs = tf.placeholder(tf.float32, [None, None, 3])
resized_img = utils.resize_image(inputs, (224, 224))

prediction = vgg16_tf.vgg16(resized_img)

sess = tf.Session()
ckpt_path = 'vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_path)

prob = tf.nn.softmax(prediction)
prob = sess.run(prob, feed_dict={inputs: img1})

utils.print_prob(prob, 'synset.txt')
