import os
import config
import random
import colorsys
import numpy as np
import tensorflow as tf
from model.yolo3_model import yolo


class yolo_predictor:

    def __init__(self, obj_threshold, nms_threshold, classes_file, anchors_file):
        """
        Introduction
        ------------
            初始化函数
        Parameters
        ----------
            obj_threshold: 目标检测为物体的阈值
            nms_threshold: nms阈值
        """
        self.obj_threshold = obj_threshold  # 0.5
        self.nms_threshold = nms_threshold  # 0.5
        # 预读取
        self.classes_path = classes_file  # './model_data/coco_classes.txt'
        self.anchors_path = anchors_file  # './model_data/yolo_anchors.txt'
        # 读取种类名称
        self.class_names = self._get_class()  # {list:80} ['person', 'bicycle', 'car', 'motorbike',  ... , 'toothbrush']
        # 读取先验框
        self.anchors = self._get_anchors()  # (9,2) [[ 10.  13.], [ 16.  30.],..., [373. 326.]]

        # 画框用
        # 创建一个包含色相值（Hue）的元组列表 hsv_tuples。其中，色相值是通过将类别数量除以元组的索引值再乘以一个比例因子得到的。
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]

        # 使用 map() 函数将 hsv_tuples 中的每个元组传递给 colorsys.hsv_to_rgb() 函数，将色相值转换为 RGB 颜色值。然后，将转换后的颜色值列表赋值给实例变量 self.colors。
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        # 使用 map() 函数将 self.colors 中的每个颜色值转换为整数，并将结果赋值给实例变量 self.colors
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        '''
        `random.seed(10101)` 是为了在程序中设置随机数生成器的种子
            在 Python 中，`random` 模块用于生成随机数。当使用 `random` 模块生成随机数时，每次运行程序时生成的随机数序列可能会有所不同，这是因为随机数生成器的初始状态是不确定的。
            通过设置随机数生成器的种子，可以确保在每次运行程序时生成的随机数序列都是相同的。这在需要重复实验或生成可重现的随机结果时非常有用。
            
            在上述代码中，`random.seed(10101)` 将随机数生成器的种子设置为 10101。这意味着在后续使用 `random` 模块生成随机数时，将基于这个种子生成相同的随机数序列。
            例如，如果你在代码中多次调用 `random.randint(1, 10)`，每次调用都会返回相同的随机整数，因为随机数生成器的种子是相同的。
            注意，设置随机数生成器的种子只会影响后续使用 `random` 模块生成的随机数，而不会影响其他代码中的随机行为。如果你在程序的其他部分使用了不同的随机数生成器或设置了不同的种子，它们将产生不同的随机结果。
        '''
        # 设置随机数生成器的种子为 10101，以确保每次运行代码时生成的随机数序列是相同的
        random.seed(10101)
        # 使用 random.shuffle() 函数打乱 self.colors 列表中元素的顺序，以实现随机颜色的效果
        random.shuffle(self.colors)
        # 将随机数生成器的种子设置为 None，以便在后续的代码中使用系统默认的随机数生成器。
        random.seed(None)

    def _get_class(self):
        """
        Introduction
        ------------
            读取类别名称
        """
        # 获取类别名称文件的路径。os.path.expanduser 用于将路径中的 ~ 或 $HOME 等用户目录缩写扩展为完整的用户目录路径
        classes_path = os.path.expanduser(self.classes_path)
        # 以读模式打开类别名称文件，并将文件对象赋值给变量 f。
        with open(classes_path) as f:
            # 通过调用文件对象的 readlines 方法，将文件中的每一行内容作为一个元素存储在列表 class_names 中
            class_names = f.readlines()
        # 使用列表推导式对 class_names 列表中的每个元素进行处理。strip 方法用于去除元素两端的空格和换行符。
        class_names = [c.strip() for c in class_names]
        # 函数返回处理后的类别名称列表
        return class_names

    def _get_anchors(self):
        """
        Introduction
        ------------
            读取anchors数据
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            # 读取文件中的第一行内容，并将其赋值给变量 anchors
            anchors = f.readline()
            # 使用列表推导式将 anchors 字符串按照逗号分割，并将每个分割后的部分转换为浮点数，存储在列表 anchors 中
            anchors = [float(x) for x in anchors.split(',')]
            # 将列表 anchors 转换为 numpy 数组，并使用 reshape 方法将其重塑为一个二维数组，其中每行包含两个 anchor 的值。
            # 将anchors数组重塑为一个二维数组，其中第二维的大小为 2。第一维的大小由-1表示，这意味着 numpy 会根据anchors数组的元素数量和第二维的大小来自动计算第一维的大小。
            #          例如，如果anchors数组有 6 个元素，那么reshape(-1, 2)将把它重塑为一个 3x2 的二维数组
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    # ---------------------------------------#
    #   对三个特征层解码
    #   进行排序并进行非极大抑制
    # ---------------------------------------#
    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        """
        Introduction
        ------------
            将预测出的box坐标转换为对应原图的坐标，然后计算每个box的分数
        Parameters
        ----------
            feats: yolo输出的feature map
            anchors: anchor的位置
            class_num: 类别数目
            input_shape: 输入大小
            image_shape: 图片大小
        Returns
        -------
            boxes: 物体框的位置
            boxes_scores: 物体框的分数，为置信度和类别概率的乘积
        """
        # 获得特征
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        # 寻找在原图上的位置
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])
        # 获得置信度box_confidence * box_class_probs
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores

    # 获得在原图上框的位置
    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        """
        Introduction
        ------------
            计算物体框预测坐标在原图中的位置坐标
        Parameters
        ----------
            box_xy: 物体框左上角坐标
            box_wh: 物体框的宽高
            input_shape: 输入的大小
            image_shape: 图片的大小
        Returns
        -------
            boxes: 物体框的位置
        """
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        # 416,416
        input_shape = tf.cast(input_shape, dtype=tf.float32)
        # 实际图片的大小
        image_shape = tf.cast(image_shape, dtype=tf.float32)

        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))

        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.concat([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis=-1)
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        return boxes

    # 其实是解码的过程
    def _get_feats(self, feats, anchors, num_classes, input_shape):
        """
        Introduction
        ------------
            根据yolo最后一层的输出确定bounding box
        Parameters
        ----------
            feats: yolo模型最后一层输出
            anchors: anchors的位置
            num_classes: 类别数量
            input_shape: 输入大小
        Returns
        -------
            box_xy, box_wh, box_confidence, box_class_probs
        """
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])

        # 这里构建13*13*1*2的矩阵，对应每个格子加上对应的坐标
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, tf.float32)

        # 将x,y坐标归一化，相对网格的位置
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        # 将w,h也归一化
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs

    def eval(self, yolo_outputs, image_shape, max_boxes=20):
        """
        Introduction
        ------------
            根据Yolo模型的输出进行非极大值抑制，获取最后的物体检测框和物体检测类别
        Parameters
        ----------
            yolo_outputs: yolo模型输出
            image_shape: 图片的大小
            max_boxes:  最大box数量
        Returns
        -------
            boxes_: 物体框的位置
            scores_: 物体类别的概率
            classes_: 物体类别
        """
        # 每一个特征层对应三个先验框
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        # 存储每个特征层的预测框和对应的分数
        boxes = []
        box_scores = []
        # inputshape是416x416
        # image_shape是实际图片的大小(输入图像的大小，通过 YOLO 模型的输出形状计算得到)
        input_shape = tf.shape(yolo_outputs[0])[1: 3] * 32

        # 对三个特征层的输出获取每个预测box坐标和box的分数，score = 置信度x类别概率
        # ---------------------------------------#
        #   对三个特征层解码
        #   获得分数和框的位置
        # ---------------------------------------#
        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]],
                                                        len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        # 将所有特征层的预测框和分数连接在一起, 放在一行里面便于操作
        boxes = tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)

        mask = box_scores >= self.obj_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []

        # ---------------------------------------#
        #   1、取出每一类得分大于self.obj_threshold
        #   的框和得分
        #   2、对得分进行非极大抑制
        # ---------------------------------------#
        # 对每一个类进行判断
        for c in range(len(self.class_names)):
            # 取出所有类为c的box
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            # 取出所有类为c的分数
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            # 非极大抑制
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                                     iou_threshold=self.nms_threshold)

            # 获取非极大抑制的结果
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            classes = tf.ones_like(class_box_scores, 'int32') * c

            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_, axis=0)
        scores_ = tf.concat(scores_, axis=0)
        classes_ = tf.concat(classes_, axis=0)
        return boxes_, scores_, classes_

    # ---------------------------------------#
    #   predict用于预测，分三步
    #   1、建立yolo对象
    #   2、获得预测结果
    #   3、对预测结果进行处理
    # ---------------------------------------#
    def predict(self, inputs, image_shape):
        """
        Introduction
        ------------
            构建预测模型
        Parameters
        ----------
            inputs: 处理之后的输入图片   Tensor("Placeholder_1:0", shape=(?, 416, 416, 3), dtype=float32)
            image_shape: 图像原始大小   Tensor("Placeholder:0", shape=(2,), dtype=int32)
        Returns
        -------
            boxes: 物体框坐标
            scores: 物体概率值
            classes: 物体类别
        """
        model = yolo(config.norm_epsilon, config.norm_decay, self.anchors_path, self.classes_path, pre_train=False)
        # yolo_inference用于获得网络的预测结果
        output = model.yolo_inference(inputs, config.num_anchors // 3, config.num_classes, training=False)
        boxes, scores, classes = self.eval(output, image_shape, max_boxes=20)
        return boxes, scores, classes
