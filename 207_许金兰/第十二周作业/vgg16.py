import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

slim = tf.contrib.slim


class VggNet:
    def __init__(self, num_classes=1000, dropout_keep_prob=0.5, is_training=True,
                 scope='vgg_16', spatial_squeeze=True):
        """
        参数初始化
        :param num_classes:分类类别数
        :param dropout_keep_prob:dropout概率
        :param is_training:是否处于训练模式，True是训练模式，False不是训练模式
        :param scope: 名称
        :param spatial_squeeze:是否需要平铺，True将数据平铺，False不处理
        """
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob
        self.is_training = is_training
        self.scope = scope
        self.spatial_squeeze = spatial_squeeze

    def read_image(self, img_path):
        """
        读取图片，并裁剪成正方形
        :param img_path: 图片路径
        :return: 返回处理后的图片
        """
        img = plt.imread(img_path)
        short_edge = min(img.shape[:2])  # 最短的边
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
        return crop_img

    def resize_image(self, img, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
        """
        将图片处理成指定大小,并增加一个维度
        :param img: 待处理的图片
        :param size: 目标h,w
        :param method:
        :param align_corners:
        :return:
        """
        with tf.name_scope('resize_image'):
            img = tf.expand_dims(img, 0)
            img = tf.image.resize_images(img, size, method, align_corners)
            return img

    def print_prob(self, prob, file_path):
        """
        打印结果
        :param prob: 模型预测结果
        :param file_path: 标签文件路径
        :return:
        """
        synset = [l.strip() for l in open(file_path).readlines()]
        pred = np.argsort(prob)[::-1]
        top1 = synset[pred[0]]
        print(("Top1: ", top1, prob[pred[0]]))
        top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
        print(("Top5: ", top5))
        return top1

    def net(self, inputs):
        """
        构建vgg16模型
        :param inputs：输入数据
        """
        with tf.variable_scope(self.scope, 'vgg_16', [inputs]):
            x = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            x = slim.max_pool2d(x, [2, 2], scope='pool1')

            x = slim.repeat(x, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            x = slim.max_pool2d(x, [2, 2], scope='pool2')

            x = slim.repeat(x, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            x = slim.max_pool2d(x, [2, 2], scope='pool3')

            x = slim.repeat(x, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            x = slim.max_pool2d(x, [2, 2], scope='pool4')

            x = slim.repeat(x, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            x = slim.max_pool2d(x, [2, 2], scope='pool5')

            x = slim.conv2d(x, 4096, [7, 7], padding='VALID', scope='fc6')
            x = slim.dropout(x, self.dropout_keep_prob, is_training=self.is_training,
                             scope='dropout6')

            x = slim.conv2d(x, 4096, [1, 1], scope='fc7')
            x = slim.dropout(x, self.dropout_keep_prob, is_training=self.is_training,
                             scope='dropout7')

            x = slim.conv2d(x, self.num_classes, [1, 1],
                            activation_fn=None,
                            normalizer_fn=None,
                            scope='fc8')
            # 是否将数据平铺
            if self.spatial_squeeze:
                x = tf.squeeze(x, [1, 2], name='fc8/squeezed')
            return x


if __name__ == '__main__':
    # 读取并处理图片
    model = VggNet()
    img = model.read_image("./test_data/dog.jpg")
    resized_img = model.resize_image(img, (224, 224))
    # 构建网络
    prediction = model.net(resized_img)
    # 加载模型
    ckpt_filename = './model/vgg_16.ckpt'
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_filename)
    # 预测结果
    pro = tf.nn.softmax(prediction)
    pre = sess.run(pro)
    # 打印结果
    model.print_prob(pre[0], './synset.txt')
