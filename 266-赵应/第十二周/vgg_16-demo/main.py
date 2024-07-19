import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import vgg


def load_image(path):
    # 读取图片，rgb
    img = mpimg.imread(path)
    # 将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img


def resize_image(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
    with tf.name_scope('resize_image'):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size,
                                       method, align_corners)
        image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))
        return image


def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    # 将概率从大到小排列的结果的序号存入pred
    pred = np.argsort(prob)[::-1]
    # 取最大的1个、5个。
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


if __name__ == '__main__':
    # 读取图片
    img1 = load_image("./test_data/dog.jpg")

    # 对输入的图片进行resize，使其shape满足(-1,224,224,3)
    inputs = tf.placeholder(tf.float32, [None, None, 3])
    resized_img = resize_image(inputs, (224, 224))

    # 建立网络结构
    vgg16 = vgg.Vgg16(resized_img)
    net = vgg16.vgg_16()

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # 载入模型
        ckpt_filename = './model/vgg_16.ckpt'
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_filename)

        # 最后结果进行softmax预测
        pro = tf.nn.softmax(net)
        pre = sess.run(pro, feed_dict={inputs: img1})

        # 打印预测结果
        print("result: ")
        print_prob(pre[0], './synset.txt')