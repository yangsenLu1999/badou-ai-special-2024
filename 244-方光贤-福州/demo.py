from nets import vgg16
import tensorflow as tf
import utils

# 测试图片读取
img1 = utils.load_image("./test_data/dog.jpg")

# 重塑测试图片 匹配模型输入
inputs = tf.placeholder(tf.float32, [None, None, 3])
resized_img = utils.resize_image(inputs, (224, 224))

# 调用网络
prediction = vgg16.vgg_16(resized_img)

# 载入模型
sess = tf.Session()
ckpt_filename = './model/vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

# softmax得到概率值
pro = tf.nn.softmax(prediction)
pre = sess.run(pro, feed_dict={inputs:img1})

# 打印预测结果
print("result: ")
utils.print_prob(pre[0], './synset.txt')
