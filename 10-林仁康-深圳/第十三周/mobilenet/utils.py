import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf

INPUT_SHAPE = (299, 299, 3)
OUTPUT_SHAPE = 1000

ANS_PATH = "./synset.txt"

# 重塑图像
def ResizeImage(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
        return images

# 读取图片并初始化
def InitInput(path):
    img = cv2.imread(path)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgNor = np.expand_dims(imgRGB / 255, axis=0)
    imgResize = ResizeImage(imgNor, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
    return imgResize

# 打印结果
def PrintAnswer(res):
    with open("./data/model/index_word.txt", "r", encoding='utf-8') as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]

    print(synset[res])
    return synset[res]
    # synset = [l.strip() for l in open(file_path).readlines()]
    # # 将概率从大到小排列的结果的序号存入pred
    # pred = np.argsort(res)[::-1]
    # # 取最大的1个、5个。
    # top1 = synset[pred[0]]
    # print(("Top1: ", top1, res[pred[0]]))
    # top5 = [(synset[pred[i]], res[pred[i]]) for i in range(5)]
    # print(("Top5: ", top5))
    return top1