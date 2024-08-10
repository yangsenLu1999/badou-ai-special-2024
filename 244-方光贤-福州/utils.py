import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

def load_image(path):
    # 处理读取图片异常
    try:
        img = mpimg.imread(path)
        if img is None:
            raise IOError("无法读取图片")
        # 中心裁剪
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
        return crop_img
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {path} 不存在")
    except IOError as e:
        raise IOError(f"读取图片时发生错误: {e}")

def resize_image(image, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
    # 尺寸重塑
    with tf.name_scope('resize_image'):
        image = tf.expand_dims(image, 0)  # 增加维度变成四维
        image = tf.image.resize_images(image, size, method, align_corners)
        image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))  # 重塑图片大小
        return image

def print_prob(prob, file_path):
    # 异常处理
    try:
        with open(file_path, 'r') as f:
            synset = [l.strip() for l in f.readlines()]
        # 根据给定的概率和类别标签文件路径打印预测结果
        pred = np.argsort(prob)[::-1]
        top1 = synset[pred[0]]  # 输出第一个
        print(f"Top1: {top1}, {prob[pred[0]]:.4f}")
        top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]  # 输出前五个
        print(f"Top5: {top5}")
        return top1
    except FileNotFoundError:
        raise FileNotFoundError(f"类别标签文件 {file_path} 不存在")
    except Exception as e:
        print(f"打印预测结果时发生错误: {e}")
        return None



