import json
import numpy as np
import tensorflow as tf
from PIL import Image
from collections import defaultdict


def load_weights(var_list, weights_file):
    """
    Introduction
    ------------
        加载预训练好的darknet53权重文件
    Parameters
    ----------
        var_list: 赋值变量名
        weights_file: 权重文件
    Returns
    -------
        assign_ops: 赋值更新操作
    """
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'conv2d' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'batch_normalization' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'conv2d' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops


# ----------------------------------------------------#
#   对预测输入图像进行缩放，按照长宽比进行缩放，不足的地方进行填充
# ----------------------------------------------------#
def letterbox_image(image, size):
    """
    Introduction
    ------------
        对预测输入图像进行缩放，按照长宽比进行缩放，不足的地方进行填充
    Parameters
    ----------
        image: 输入图像     image_w : 350, image_h : 500
        size: 图像大小      (416, 416)
    Returns
    -------
        boxed_image: 缩放后的图像
    """
    image_w, image_h = image.size  # image_w : 350, image_h : 500
    w, h = size  # h ：416   w : 416
    '''
    计算缩放后的图像宽度: 
       首先，`w*1.0/image_w`表示目标宽度与原始图像宽度的比例，`h*1.0/image_h`表示目标高度与原始图像高度的比例。
       然后，`min(w*1.0/image_w, h*1.0/image_h)`取这两个比例中的较小值，即保证缩放后的图像在宽度和高度上都不会超过目标大小。
       最后，将这个较小值乘以原始图像宽度，得到缩放后的图像宽度，并将结果转换为整数类型。
    '''
    new_w = int(image_w * min(w * 1.0 / image_w, h * 1.0 / image_h))  # 416
    new_h = int(image_h * min(w * 1.0 / image_w, h * 1.0 / image_h))  # 291
    # Image.BICUBIC`是`PIL`（Python Imaging Library）库中的一个图像缩放插值方法。它使用双三次插值算法来进行图像缩放，可以在保持图像质量的同时，对图像进行较大比例的缩放操作。
    # <PIL.Image.Image image mode=RGB size=416x291 at 0x2D0500B4D68>   `
    resized_image = image.resize((new_w, new_h), Image.BICUBIC)

    '''
    创建了一个新的图像对象，用于存储缩放并填充后的图像。        
        - `Image.new('RGB', size, (128, 128, 128))`：这是`PIL`库中的`Image.new()`函数，用于创建一个具有指定模式（这里是`RGB`）、大小（由`size`参数指定）和初始颜色（这里是`(128, 128, 128)`，即灰色）的新图像。
        - `'RGB'`：表示图像的模式为`RGB`，即红、绿、蓝三原色。这是一种常见的色彩模式，用于显示彩色图像。
        - `size`：是一个元组，表示新图像的大小，即宽度和高度。
        - `(128, 128, 128)`：是一个元组，表示新图像的初始颜色。在这里，使用了灰色作为初始颜色。
        
        通过创建这个新的图像对象，我们可以将缩放后的图像粘贴到其中，并得到最终的结果图像。
    => <PIL.Image.Image image mode=RGB size=416x416 at 0x2D050176EF0>
    '''
    boxed_image = Image.new('RGB', size, (128, 128, 128))
    '''
    将缩放后的图像resized_image 粘贴到 新创建的图像对象boxed_image中:    
        - `boxed_image`：目标图像,这是之前创建的新图像对象，用于存储缩放并填充后的图像。
        - `paste()`：这是`Image`对象的方法，用于将一个图像粘贴到另一个图像上。
        - `resized_image`：要粘贴的图像,这是之前缩放后的图像。
        - `((w-new_w)//2,(h-new_h)//2)`：这是一个坐标元组，表示将缩放后的图像粘贴到新图像的位置。(这样可以将图像粘贴到目标图像的中心位置)
                                         其中，`(w-new_w)//2`计算了水平方向上的偏移量，`(h-new_h)//2`计算了垂直方向上的偏移量。
                                               (416-416)//2                        (416-291)//2
    这样可以将缩放后的图像居中粘贴到新图像中，将较小的图像按照一定的比例放大或缩小，并在放大或缩小后的图像周围进行填充，以达到指定的尺寸，从而得到最终的结果图像。
    => <PIL.Image.Image image mode=RGB size=416x416 at 0x2D050176EF0>
    '''
    boxed_image.paste(resized_image, ((w - new_w) // 2, (h - new_h) // 2))
    return boxed_image


def draw_box(image, bbox):
    """
    Introduction
    ------------
        通过tensorboard把训练数据可视化
    Parameters
    ----------
        image: 训练数据图片
        bbox: 训练数据图片中标记box坐标
    """
    xmin, ymin, xmax, ymax, label = tf.split(value=bbox, num_or_size_splits=5, axis=2)
    height = tf.cast(tf.shape(image)[1], tf.float32)
    weight = tf.cast(tf.shape(image)[2], tf.float32)
    new_bbox = tf.concat(
        [tf.cast(ymin, tf.float32) / height, tf.cast(xmin, tf.float32) / weight, tf.cast(ymax, tf.float32) / height,
         tf.cast(xmax, tf.float32) / weight], 2)
    new_image = tf.image.draw_bounding_boxes(image, new_bbox)
    tf.summary.image('input', new_image)


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre
