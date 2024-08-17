import os

import keras.backend as K
import numpy as np

from nets.mrcnn import get_predict_model
from utils import visualize
from utils.anchors import get_anchors
from utils.config import Config
from utils.utils import mold_inputs, unmold_detections


class MASK_RCNN(object):
    _defaults = {
        "model_path": 'model_data/mask_rcnn_coco.h5',
        "classes_path": 'model_data/coco_classes.txt',
        "confidence": 0.7,
        # 使用coco数据集检测的时候，IMAGE_MIN_DIM=1024，IMAGE_MAX_DIM=1024, RPN_ANCHOR_SCALES=(32, 64, 128, 256, 512)
        "RPN_ANCHOR_SCALES": (32, 64, 128, 256, 512),
        "IMAGE_MIN_DIM": 1024,
        "IMAGE_MAX_DIM": 1024,

        # 在使用自己的数据集进行训练的时候，如果显存不足要调小图片大小
        # 同时要调小anchors
        # "IMAGE_MIN_DIM": 512,
        # "IMAGE_MAX_DIM": 512,
        # "RPN_ANCHOR_SCALES": (16, 32, 64, 128, 256)
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name %s'" % n

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.num_classes = len(self.class_names)
        self.sess = K.get_session()
        self.config = self._get_config()
        self.model = self._get_model()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path, encoding='utf-8') as file:
            class_names = file.readlines()
        class_names = [c.strip() for c in class_names]
        class_names.insert(0, "BG")
        return class_names

    def _get_config(self):
        class InferenceConfig(Config):
            NUM_CLASSES = len(self.class_names)
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = self.confidence

            NAME = 'shapes'
            RPN_ANCHOR_SCALES = self.RPN_ANCHOR_SCALES
            IMAGE_MAX_DIM = self.IMAGE_MAX_DIM
            IMAGE_MIN_DIM = self.IMAGE_MIN_DIM

        config = InferenceConfig()
        config.display()
        return config

    def _get_model(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(".h5"), "Keras model or weights must be a .h5 file."
        model = get_predict_model(self.config)
        model.load_weights(model_path, by_name=True)
        return model

    def detect_image(self, image):
        image = [np.array(image)]
        molded_images, image_metas, windows = mold_inputs(self.config, image)
        image_shape = molded_images[0].shape
        anchors = get_anchors(self.config, image_shape)
        anchors = np.broadcast_to(anchors, (1,) + anchors.shape)

        detections, _, _, mrcnn_mask, _, _, _ = \
            self.model.predict([molded_images, image_metas, anchors], verbose=0)

        final_rois, final_class_ids, final_scores, final_masks = \
            unmold_detections(detections[0], mrcnn_mask[0],
                              image[0].shape, molded_images[0].shape,
                              windows[0])

        r = {
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        }

        visualize.display_instances(image[0], r['rois'], r['masks'], r['class_ids'],
                                    self.class_names, r['scores'])
    def close_session(self):
        self.sess.close()
