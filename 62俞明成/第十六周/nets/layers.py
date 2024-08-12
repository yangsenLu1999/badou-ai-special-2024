import tensorflow as tf
from utils import utils
from keras.engine import Layer
import numpy as np


# ----------------------------------------------------------#
#   Proposal Layer
#   该部分代码用于将先验框转化成建议框
# ----------------------------------------------------------#
def apply_box_deltas_graph(boxes, deltas):
    # 计算先验框的中心和宽高
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # 计算出调整后的先验框的中心和宽高
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # 计算左上角和右下角的点的坐标
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(Layer):

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    # [rpn_class, rpn_bbox, anchors]
    def call(self, inputs, **kwargs):
        # 代表这个先验框内部是否有物体[batch, num_rois, 1]
        scores = inputs[0][:, :, 1]

        # 代表这个先验框的调整参数[batch, num_rois, 4]
        deltas = inputs[1]

        # [0.1 0.1 0.2 0.2]，改变数量级
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])

        # Anchors
        anchors = inputs[2]

        # 筛选出得分前6000个的框
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        # 获得这些框的索引
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices

        # 获得这些框的得分
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        # 获得这些框的调整参数
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        # 获得这些框对应的先验框
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                            self.config.IMAGES_PER_GPU, names=["pre_nms_anchors"])

        # [batch, N, (y1, x1, y2, x2)]
        # 对先验框进行解码
        boxes = utils.batch_slice([pre_nms_anchors, deltas], lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU, names=["refined_anchors"])

        # [batch, N, (y1, x1, y2, x2)]
        # 防止超出图片范围
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes, lambda x: clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU, names=["refined_anchors_clipped"])

        # 非极大抑制
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(boxes, scores, self.proposal_count,
                                                   self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # 如果数量达不到设置的建议框数量的话
            # 就padding
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        proposals = utils.batch_slice([boxes, scores], nms,
                                      self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


# ----------------------------------------------------------#
#   ROIAlign Layer
#   利用建议框在特征层上截取内容
# ----------------------------------------------------------#
def log2_graph(x):
    # 返回以2为底取对数
    return tf.log(x) / tf.log(2.0)


def parse_image_meta_graph(meta):
    """将meta里面的参数进行分割

    :param meta:
    :return:
        image_id: 图像的ID；
        original_image_shape: 图像的原始尺寸(高度、宽度、通道数)；
        image_shape: 图像的实际尺寸(经过缩放后的高度、宽度、通道数)；
        window: 图像在缩放后的窗口位置(包括四个坐标：y1, x1, y2, x2)；
        scale: 图像缩放比例；
        active_class_ids: 活跃的类别 ID 列表
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


class PyramidROIAlign(Layer):
    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs, **kwargs):
        # 建议框的位置
        boxes = inputs[0]

        # image_meta包含了一些必要的图片信息
        image_meta = inputs[1]

        # 取出所有的特征层[batch, height, width, channels]
        feature_maps = inputs[2:]

        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1

        # 获得输入进来的图像的大小
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]

        # 通过建议框的大小找到这个建议框属于哪个特征层
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        # batch_size, box_num
        roi_level = tf.squeeze(roi_level, 2)

        # 循环遍历各个级别，并对每个级别应用ROI池. P2至P5
        pooled = []
        box_to_level = []
        # 分别在P2-P5中进行截取
        for i, level in enumerate(range(2, 6)):
            # 找到每个特征层对应box
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)
            box_to_level.append(ix)

            # 获得这些box所属的图片
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # 停止梯度下降
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices,
                self.pool_shape, method="bilinear"))

        pooled = tf.concat(pooled, axis=0)

        # 将顺序和所属的图片进行堆叠
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # box_to_level[:, 0]表示第几张图
        # box_to_level[:, 1]表示第几张图里的第几个框
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        # 进行排序，将同一张图里的某一些聚集在一起
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]

        # 按顺序获得图片的索引
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # 重新reshape为原来的格式
        # 也就是
        # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)


# ----------------------------------------------------------#
#   Detection Layer
#
# ----------------------------------------------------------#

def refine_detections_graph(rois, probs, deltas, window, config):
    """细化分类建议并过滤重叠部分并返回最终结果探测。
    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # 找到得分最高的类
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # 序号+类
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    # 取出成绩
    class_scores = tf.gather_nd(probs, indices)
    # 还有框的调整参数
    deltas_specific = tf.gather_nd(deltas, indices)
    # 进行解码
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # 防止超出0-1
    refined_rois = clip_boxes_graph(refined_rois, window)

    # 去除背景
    keep = tf.where(class_ids > 0)[:, 0]
    # 去除背景和得分小的区域
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # 获得除去背景并且得分较高的框还有种类与得分
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]

        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD)

        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))

        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)

        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. 进行非极大抑制
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. 找到符合要求的需要被保留的建议框
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]

    # 寻找得分最高的num_keep个框
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
    ], axis=1)

    # 如果达不到数量的话就padding
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


def norm_boxes_graph(boxes, shape):
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


class DetectionLayer(Layer):

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs, **kwargs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # 找到window的小数形式
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)



