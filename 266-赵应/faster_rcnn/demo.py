import colorsys
import copy
import math
import os

import keras.backend as backend
from PIL import ImageDraw, ImageFont
from keras.applications.imagenet_utils import preprocess_input

from faster_rcnn.nets.frcnn import get_predict_model
from faster_rcnn.nets.frcnn_training import get_new_img_size
from faster_rcnn.utils.anchors import get_anchors
from faster_rcnn.utils.config import Config
from faster_rcnn.utils.utils import BBoxUtility


class FRCNN(object):
    _defaults = {
        "model_path": 'model/voc_weights.h5',
        "classes_path": 'model/voc_classes.txt',
        "confidence": .7
    }

    @classmethod
    def get_config(cls, key):
        if key in cls._defaults:
            return cls._defaults[key]
        else:
            raise Exception("Unrecognized attribute name：{}".format(key))

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_classes()
        self.num_classes = len(self.class_names) + 1
        self.config = Config()
        self.model_rpn, self.model_classifier, self.colors = self.load_model()
        self.session = backend.get_session()
        self.bbox_util = BBoxUtility()

    def _get_classes(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as file:
            class_names = file.readlines()
        class_names = [class_name.strip() for class_name in class_names]
        return class_names

    def load_model(self):
        # 解析模型路径
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(".h5"), 'model or weights must be a .h5 file.'

        # 加载模型
        model_rpn, model_classifier = get_predict_model(self.config, self.num_classes)
        model_rpn.load_weights(self.model_path, by_name=True)
        model_classifier.load_weights(self.model_path, by_name=True, skip_mismatch=True)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        return model_rpn, model_classifier, colors

    def get_img_output_length(self, width, height):
        def get_output_length(input_length):
            # input_length += 6
            filter_sizes = [7, 3, 1, 1]
            padding = [3, 1, 0, 0]
            stride = 2
            for i in range(4):
                # input_length = (input_length - filter_size + stride) // stride
                input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
            return input_length

        return get_output_length(width), get_output_length(height)

        # ---------------------------------------------------#

    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        old_width = image_shape[1]
        old_height = image_shape[0]
        old_image = copy.deepcopy(image)
        width, height = get_new_img_size(old_width, old_height)

        image = image.resize([width, height])
        photo = np.array(image, dtype=np.float64)

        # 图片预处理，归一化
        photo = preprocess_input(np.expand_dims(photo, 0))
        preds = self.model_rpn.predict(photo)
        # 将预测结果进行解码
        anchors = get_anchors(self.get_img_output_length(width, height), width, height)

        rpn_results = self.bbox_util.detection_out(preds, anchors, 1, confidence_threshold=0)
        # print(rpn_results)
        # R = rpn_results[0][:,2:]
        R = rpn_results[0][:, 2:]

        R[:, 0] = np.array(np.round(R[:, 0] * width / self.config.rpn_stride), dtype=np.int32)
        R[:, 1] = np.array(np.round(R[:, 1] * height / self.config.rpn_stride), dtype=np.int32)
        R[:, 2] = np.array(np.round(R[:, 2] * width / self.config.rpn_stride), dtype=np.int32)
        R[:, 3] = np.array(np.round(R[:, 3] * height / self.config.rpn_stride), dtype=np.int32)

        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        base_layer = preds[2]

        delete_line = []
        for i, r in enumerate(R):
            if r[2] < 1 or r[3] < 1:
                delete_line.append(i)
        R = np.delete(R, delete_line, axis=0)

        bboxes = []
        probs = []
        labels = []
        for jk in range(R.shape[0] // self.config.num_rois + 1):
            ROIs = np.expand_dims(R[self.config.num_rois * jk:self.config.num_rois * (jk + 1), :], axis=0)

            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // self.config.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], self.config.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = self.model_classifier.predict([base_layer, ROIs])

            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :]) < self.confidence or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                label = np.argmax(P_cls[0, ii, :])

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])

                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= self.config.classifier_regr_std[0]
                ty /= self.config.classifier_regr_std[1]
                tw /= self.config.classifier_regr_std[2]
                th /= self.config.classifier_regr_std[3]

                cx = x + w / 2.
                cy = y + h / 2.
                cx1 = tx * w + cx
                cy1 = ty * h + cy
                w1 = math.exp(tw) * w
                h1 = math.exp(th) * h

                x1 = cx1 - w1 / 2.
                y1 = cy1 - h1 / 2.

                x2 = cx1 + w1 / 2
                y2 = cy1 + h1 / 2

                x1 = int(round(x1))
                y1 = int(round(y1))
                x2 = int(round(x2))
                y2 = int(round(y2))

                bboxes.append([x1, y1, x2, y2])
                probs.append(np.max(P_cls[0, ii, :]))
                labels.append(label)

        if len(bboxes) == 0:
            return old_image

        # 筛选出其中得分高于confidence的框
        labels = np.array(labels)
        probs = np.array(probs)
        boxes = np.array(bboxes, dtype=np.float32)
        boxes[:, 0] = boxes[:, 0] * self.config.rpn_stride / width
        boxes[:, 1] = boxes[:, 1] * self.config.rpn_stride / height
        boxes[:, 2] = boxes[:, 2] * self.config.rpn_stride / width
        boxes[:, 3] = boxes[:, 3] * self.config.rpn_stride / height
        results = np.array(
            self.bbox_util.nms_for_out(np.array(labels), np.array(probs), np.array(boxes), self.num_classes - 1, 0.4))

        top_label_indices = results[:, 0]
        top_conf = results[:, 1]
        boxes = results[:, 2:]
        boxes[:, 0] = boxes[:, 0] * old_width
        boxes[:, 1] = boxes[:, 1] * old_height
        boxes[:, 2] = boxes[:, 2] * old_width
        boxes[:, 3] = boxes[:, 3] * old_height

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(old_image)[0] + np.shape(old_image)[1]) // width
        image = old_image
        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]

            left, top, right, bottom = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def close_session(self):
        self.session.close()


if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    import cv2

    frcnn = FRCNN()

    # 格式转变，BGRtoRGB
    image = cv2.imread("img/street.jpg")
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))

    # 进行检测
    frame = np.array(frcnn.detect_image(frame))
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("video", frame)
    cv2.waitKey(0)
frcnn.close_session()
