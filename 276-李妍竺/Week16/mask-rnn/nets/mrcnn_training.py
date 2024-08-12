import tensorflow as tf
import keras.backend as k
import random
import numpy as np
import logging
from utils import utils
from utils.anchors import compute_backbone_shapes, generate_pyramid_anchors


# ---------------------------------------------------------#
#  Loss Functions
# ---------------------------------------------------------#
def batch_pack_graph(x, counts, num_rows):
    """
    从每行中拾取不同数量的值，以x表示，具体取决于计数中的值。
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def smooth_l1_loss(y_true, y_pred):
    """
    实现平滑L1损失。
    y_true和y_pred通常为：[N，4]，但可以是任何形状。
    """
    diff = k.abs(y_true - y_pred)
    less_than_one = k.cast(k.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """
    RPN anchors classifier损失。
    rpn_match:[batch，anchors，1]。锚匹配类型。1=阳性，-1=负，0=中性锚。
    rpn_class_logits：[batch，anchors，2]。BG/FG的RPN classifier logits。
    """
    # 删除最后一个维度以简化
    rpn_match = tf.squeeze(rpn_match, -1)
    # 获取anchor_class。两个值0，1 -> BG，FG
    anchor_class = k.cast(k.equal(rpn_match, 1), tf.int32)
    # 正样本和负样本导致了损失，
    # 但是需要忽略的（匹配值=0）没有。不需要这些框
    indices = tf.where(k.not_equal(rpn_match, 0))
    # 选择造成损失的行，过滤掉其余的行。
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = k.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = k.switch(tf.size(loss) > 0, k.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """
    返回rpn_bbox的损失
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].有使用0填充
    rpn_match: [batch, anchors, 1]. 锚框类型. 1=positive,-1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # 正样本会造成损失，但是负样本和中性样本不会造成损失
    # 简化shape
    rpn_match = k.squeeze(rpn_match, -1)
    # 正样本的索引
    indices = tf.where(k.equal(rpn_match, 1))

    # 选择计算损失的值
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # 去除填充
    batch_counts = k.sum(k.cast(k.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)

    loss = k.switch(tf.size(loss) > 0, k.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """mask_rcnn分类损失

    target_class_ids: [batch, num_rois]. 整数类ID。使用了零填充来填充数组。
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. 对于图像数据集中的类，其值为1，而对于不在数据集中的类，其值则为0。
    """
    # 在模型构建过程中，Keras使用float32类型的target_class_ids。
    # 变成int
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # 最有可能为的类的索引 ？？？
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # 更新此行以使用批处理>1。现在它假设所有批处理中的图像具有相同的active_class_id ？？？
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # 删除未处于活动状态的类的预测丢失图像的类。
    loss = loss * pred_active

    # 计算均值，仅使用有贡献的预测以获得正确的平均值。
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """mrcnn分类网络后bbox的损失

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois].
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # 重塑形状以合并批次和roi维度以简化
    target_class_ids = k.reshape(target_class_ids, (-1,))
    target_bbox = k.reshape(target_bbox, (-1, 4))
    pred_bbox = k.reshape(pred_bbox, (-1, k.int_shape(pred_bbox)[2], 4))

    # 只有正样本参与损失
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # 获取造成损失的值
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = k.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    # 计算均值
    loss = k.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """
    mrcnn mask损失，以下两个masks值均已归一化
    target_masks: [batch, num_rois, height, width].
    target_class_ids: [batch, num_rois].
    pred_masks: [batch, proposals, height, width, num_classes]
    """
    # 简化
    target_class_ids = k.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = k.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = k.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # 将预测掩码置换为[N，num_classes，height，width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # 只有正样本参与计算，获取索引
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # 获取参与计算的值
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # shape: [batch, roi, num_classes]
    loss = k.switch(tf.size(y_true) > 0,
                    k.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    # 均值
    loss = k.mean(loss)
    return loss


# ---------------------------------------------------------#
#  Data Generator
# ---------------------------------------------------------#
def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    # 载入图片和语义分割效果
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    # 原始shape
    original_shape = image.shape
    # 获得新图片，原图片在新图片中的位置，变化的尺度，填充的情况等
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        mode=config.IMAGE_RESIZE_MODE
    )
    mask = utils.resize_mask(mask, scale, padding, crop)
    # 可以有概率把图片进行翻转
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    if augmentation:
        import imgaug
        # 这个库可用于图像增强
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad", "Affine",
                           "PiecewiseAffine"]

        def hook(images, augmenter, parents, defaults):
            """确定要应用于mask的增强因子。"""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        image_shape = image.shape
        mask_shape = mask.shape
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change image size"
        mask = mask.astype(np.bool)

    # 捡漏，防止某些层内部实际上不存在语义分割情况
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # 找到mask对应的box
    bbox = utils.extract_bboxes(mask)

    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)
    # 生成image_meta
    image_meta = utils.compose_image_meta(image_id, original_shape, image.shape, window,
                                          scale, active_class_ids)
    return image, image_meta, class_ids, bbox, mask


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    # 1代表正样本
    # -1代表负样本
    # 0代表忽略
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # 创建该部分内容利用先验框和真实框进行编码
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))
    '''
        iscrowd=0的时候，表示这是一个单独的物体，轮廓用Polygon(多边形的点)表示，
        iscrowd=1的时候表示两个没有分开的物体，轮廓用RLE编码表示，比如说一张图片里面有三个人，
        一个人单独站一边，另外两个搂在一起（标注的时候距离太近分不开了），这个时候，
        单独的那个人的注释里面的iscrowing=0,segmentation用Polygon表示，
        而另外两个用放在同一个anatation的数组里面用一个segmention的RLE编码形式表示
    '''
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = crowd_iou_max < 0.001
    else:
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # 计算先验框和真实框的重合程度[num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # 1. 重合程度小于0.3则代表为负样本
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    # 每个先验框相对于所有标定框的iou的最大值，在这里面再确定哪些是什么样本
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & no_crowd_bool] = -1
    # 2. 每个真实框重合度最大的先验框是正样本
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
    rpn_match[gt_iou_argmax] = 1
    # 3. 重合度大于等于0.7，则代表为正样本
    rpn_match[anchor_iou_max >= 0.7] = 1

    # 正负样本平衡
    # 找到正样本的索引
    ids = np.where(rpn_match == 1)[0]
    # 如果大于(config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)则删掉一些
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # 找到负样本的索引
    ids = np.where(rpn_match == -1)[0]
    # 改变数量
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1))
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # 找到内部真实存在物体的先验框，进行编码
    ids = np.where(rpn_match == 1)[0]
    ix = 0
    for i, a in zip(ids, anchors[ids]):
        gt = gt_boxes[anchor_iou_argmax[i]]
        # 计算真实框的中心，高宽
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5*gt_h
        gt_center_x = gt[1] + 0.5*gt_w
        # 计算先验框的中心，宽高
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5*a_h
        a_center_x = a[1] + 0.5*a_w
        # 编码运算
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w)
        ]

        # 改变数量级
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1
    return rpn_match, rpn_bbox


def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None,
                   batch_size=1, detection_targets=False, no_augmentation_sources=None):
    """
    input：
    -images：[批次，H，W，C]
    -image_meta:[batch，（meta data）]图像详细信息。请参见compose_image_meta（）
    -rpn_match:[batch，N]整数（1=正锚点，-1=负锚点，0=中性锚点）
    -rpn_bbox:[batch，N，（dy，dx，log（dh），log（dw））]锚定bbox增量。
    -gt_class_ids:[batch，MAX_gt_INSTANCES]整数类ID
    -gt_boxes:[batch，MAX_gt_INSTANCES，（y1，x1，y2，x2）]
    -gt_masks:[batch，高度，宽度，MAX_gt_INSTANCES]。高度和宽度
    是图像的值，除非use_mini_mask为True，如果它们是在MINI_MASK_SHAPE中定义的。
    output：在常规训练中通常为空。但如果检测目标如果为True，包含class_ids,bbox_deltas,masks。
    """
    b = 0  # batch 的索引
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    no_augmentation_sources = no_augmentation_sources or []

    # [anchors_count, (y1,x1,y2,x2)]
    # 计算获得先验框
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS,
                                       backbone_shapes, config.BACKBONE_STRIDES,
                                       config.RPN_ANCHOR_STRIDE)

    while True:
        # 为0和len(image_ids)时打乱顺序
        image_index = (image_index + 1) % len(image_ids)
        if shuffle and image_index == 0:
            np.random.shuffle(image_ids)

        # 获得id
        image_id = image_ids[image_index]

        # 获得图片，真实框，语义分割结果等
        if dataset.image_info[image_id]['source'] in no_augmentation_sources:
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = load_image_gt(dataset, config,
                                                                                image_id,
                                                                                augment=augment,
                                                                                augmentation=None,
                                                                                use_mini_mask=config.USE_MINI_MASK)
        else:
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = load_image_gt(dataset, config,
                                                                                image_id,
                                                                                augment=augment,
                                                                                augmentation=augmentation,
                                                                                use_mini_mask=config.USE_MINI_MASK)

        if not np.any(gt_class_ids > 0):
            continue

        # RPN targets
        rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors, gt_class_ids, gt_boxes, config)

        # 如果某张图片里的物体数量大于最大值的话，则进行筛选，防止过大
        if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
            ids = np.random.choice(np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]

        # 初始化用于训练的内容
        if b == 0:
            batch_image_meta = np.zeros((batch_size,) + image_meta.shape, dtype=image_meta.dtype)
            batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
            batch_rpn_bbox = np.zeros([batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
            batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
            batch_gt_class_ids = np.zeros([batch_size, config.MAX_GT_INSTANCES], dtype=np.int32)
            batch_gt_boxes = np.zeros([batch_size, config.MAX_GT_INSTANCES, 4], dtype=np.int32)
            batch_gt_masks = np.zeros([batch_size, gt_masks.shape[0], gt_masks.shape[1], config.MAX_GT_INSTANCES],
                                      dtype=gt_masks.dtype)

        # 赋值
        batch_image_meta[b] = image_meta
        batch_rpn_match[b] = rpn_match[:, np.newaxis]  # 凑维度
        batch_rpn_bbox[b] = rpn_bbox
        batch_images[b] = utils.mold_image(image.astype(np.float32), config)
        batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
        batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
        batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks

        b += 1

        # batch full
        if b >= batch_size:
            inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids,
                      batch_gt_boxes, batch_gt_masks]
            outputs = []

            yield inputs, outputs

            # start a new batch
            b = 0