# coding = utf-8

'''
    mask r-cnn实例
'''

import os
import sys
import random
import math
import numpy as np
import skimage.io
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from nets.mrcnn import get_predict_model
from utils.config import Config
from utils.anchors import get_anchors
from utils.utils import mold_inputs, unmold_detections
from utils import visualize
import keras.backend as K

class MASK_RCNN(object):
    # 设置默认属性
    _defaults = {
        'model_path': 'model_data/mask_rcnn_coco.h5',
        'classes_path': 'model_data/coco_classes.txt',
        'confidence': '0.7',
        # 使用coco数据集：
        'RPN_ANCHOR_SCALES': (32, 64, 128, 256, 512),
        'IMAGE_MIN_DIM': 1024,
        'IMAGE_MAX_DIM': 1024
        # 使用自己的数据集，显存不足要调小图片大小和anchors：
        # 'RPN_ANCHOR_SCALES': (16, 32, 64, 128, 256),
        # 'IMAGE_MIN_DIM': 512,
        # 'IMAGE_MAX_DIM': 512
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return '无此属性' + n

    def __init__(self, **kwargs):
        # 初始化 MASK_RCNN
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.sess = K.get_session()
        self.config = self._get_config()
        self.generate()


    def _get_class(self):
        # 获取所有类别
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        class_names.insert(0, 'BG')     # 在列表第一位插入背景'BG'
        return class_names

    def _get_config(self):
        # 获取配置项
        class InferenceConfig(Config):
            NUM_CLASSES = len(self.class_names)
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = self.confidence
            NAME = 'shapes'
            RPN_ANCHOR_SCALES = self.RPN_ANCHOR_SCALES
            IMAGE_MIN_DIM = self.IMAGE_MIN_DIM
            IMAGE_MAX_DIM = self.IMAGE_MAX_DIM
        config = InferenceConfig()
        config.display()
        return config

    def generate(self):
        # 生成模型
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras模型必须为.h5文件！'
        self.num_classes = len(self.class_names)    # 所有种类数
        self.model = get_predict_model(self.config)
        self.model.load_weights(self.model_path, by_name=True)  # by_name=True 参数意味着预训练模型权重将被加载到模型中名称匹配的层上

    def detect_image(self, image):
        # 检测图片
        image = [np.array(image)]
        molded_images, image_metas, windows = mold_inputs(self.config, image)   # 返回处理后的图像、图像元数据、窗口信息
        image_shape = molded_images[0].shape
        anchors = get_anchors(self.config, image_shape)
        anchors = np.broadcast_to(anchors, (1,) + anchors.shape)

        detections, _, _, mask, _, _, _ =\
            self.model.predict(
            [molded_images, image_metas, anchors], verbose=0
        )

        final_rois, final_class_ids, final_scores, final_masks =\
            unmold_detections(detections[0], mask[0], image[0].shape, molded_images[0].shape, windows[0])
        # 使用unmold_detections函数
        # 这个函数根据原始图像尺寸image[0].shape、处理后的图像尺寸molded_images[0].shape和窗口信息windows[0]
        # 将检测结果转换为更易于理解和使用的格式

        r = {
            'rois': final_rois,     # 目标区域
            'class_ids': final_class_ids,       # 类别id
            'scores': final_scores,     # 置信度分数
            'masks': final_masks        # 掩膜
        }

        visualize.display_instance(image[0], r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'])
        # 使用visualize.display_instances函数将检测到的目标显示在原始图像上。
        # 接受原始图像、目标区域（rois）、掩码（masks）、类别ID（class_ids）、类别名称（self.class_names）和置信度分数（scores）作为输入，并生成一个带有标注和掩码的图像

    def close_session(self):
        self.sess.close()

