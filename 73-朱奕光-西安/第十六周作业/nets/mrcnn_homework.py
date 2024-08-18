from keras.layers import (Conv2D, Reshape, Activation, Input,
                          Lambda, Add, UpSampling2D, MaxPooling2D,
                          Concatenate, TimeDistributed, Dense, BatchNormalization,
                          Conv2DTranspose)
from keras.models import Model
import tensorflow as tf
import sys
sys.path.append("..")
from nets.resnet_homework import get_resnet
from nets.layers import ProposalLayer, PyramidROIAlign, DetectionLayer
import keras.backend as K
from utils.utils import parse_image_meta_graph, norm_boxes_graph



def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    # ROI Align，利用建议框在特征层上进行截取
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = TimeDistributed(BatchNormalization(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = Activation('relu')(x)

    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = TimeDistributed(BatchNormalization(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = Activation('relu')(x)

    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = TimeDistributed(BatchNormalization(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = Activation('relu')(x)

    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = TimeDistributed(BatchNormalization(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = Activation('relu')(x)

    # Shape: [batch, num_rois, 2xMASK_POOL_SIZE, 2xMASK_POOL_SIZE, channels]
    x = TimeDistributed(Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    # 反卷积后再次进行一个1x1卷积调整通道，使其最终数量为numclasses，代表分的类
    x = TimeDistributed(Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask")(x)
    return x

def rpn_graph(feature_map, anchor_per_location):
    """
    RPN过程：兵分两路，shared1进行是否有目标的检测，shared2进行bbox-regression检测
    :param feature_map:  经过CNN的特征图
    :param anchor_per_location: 先验框参数
    :return: rpn_class_logits： 未经softmax处理的物体框是否有目标的得分
             rpn_probs： 经softmax处理后的物体框是否有目标的得分
             rpn_bbox： bbox-regression预测的偏移量（为dx，dy，dw，dh）
    """
    shared = Conv2D(512, (3, 3), padding='same', activation='relu',
                    name='rpn_conv_shared')(feature_map)

    x = Conv2D(2 * anchor_per_location, (1, 1), padding='valid', activation='linear',
               name='rpn_class_raw')(shared)

    rpn_class_logits = Reshape([-1, 2])(x)   # 判断每个框内是否有目标

    rpn_probs = Activation("softmax", name='rpn_class_xxx')(rpn_class_logits)

    x = Conv2D(anchor_per_location * 4, (1, 1), padding='valid', activation='linear',
               name='rpn_bbox_pred')(shared)

    rpn_bbox = Reshape([-1, 4])(x)   # 判断每个框的regression偏移量

    return [rpn_class_logits, rpn_probs, rpn_bbox]

def build_rpn_model(anchors_per_location, depth):
    """
    建立RPN网络
    :param anchors_per_location:  anchors预设值
    :param depth:  通道数
    :return:  RPN网络
    """
    input_feature_map = Input(shape=[None, None, depth], name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location)
    return Model([input_feature_map], outputs, name="rpn_model")

def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    # ROI Pooling，利用建议框在特征层上进行截取
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)

    # Shape: [batch, num_rois, 1, 1, fc_layers_size]，相当于两次全连接
    x = TimeDistributed(Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = Activation('relu')(x)

    # Shape: [batch, num_rois, 1, 1, fc_layers_size]
    x = TimeDistributed(Conv2D(fc_layers_size, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = Activation('relu')(x)

    # Shape: [batch, num_rois, fc_layers_size]
    shared = Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    # Classifier head
    # 这个的预测结果代表这个先验框内部的物体的种类
    mrcnn_class_logits = TimeDistributed(Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = TimeDistributed(Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)


    # BBox head
    # 这个的预测结果会对先验框进行调整
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = TimeDistributed(Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    mrcnn_bbox = Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox



def get_predict_model(config):
    h, w = config.IMAGE_SHAPE[:2]
    if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")

    input_image = Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
    input_image_meta = Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")

    # 输入进来的先验框
    input_anchors = Input(shape=[None, 4], name="input_anchors")

    #从resnet网络中获取压缩程度不通的层
    _, C2, C3, C4, C5 = get_resnet(input_image, stage5=True, train_bn=config.TRAIN_BN)
    # C2:height/4, weight/4, 256
    # C3:height/8, weight/8, 512
    # C4:height/16, weight/16, 1024
    # C5:height/32, weight/32, 2048
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    P4 = Add(name='fpn_p4add')([
        UpSampling2D(size=(2, 2), name='fpn_p5upsampled')(P5),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1 ,1), name='fpn_c4p4')(C4)
    ])
    P3 = Add(name='fpn_p3add')([
        UpSampling2D(size=(2, 2), name='fpn_p4upsampled')(P4),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)
    ])
    P2 = Add(name='fpn_p2add')([
        UpSampling2D(size=(2, 2), name='fpn_p3upsampled')(P3),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)
    ])

    P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='SAME', name='fpn_p2')(P2)   # height/4, weight/4
    P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='SAME', name='fpn_p3')(P3)   # height/8, weight/8
    P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='SAME', name='fpn_p4')(P4)   # height/16, weight/16
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='SAME', name='fpn_p5')(P5)   # height/32, weight/32
    # P6用于获取建议框
    P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name='fpn_p6')(P5)                                # height/64, weight/64

    # P2, P3, P4, P5, P6用于获取建议框，用于训练过程
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    # P2, P3, P4, P5用于获取mask信息，用于推理过程
    mrcnn_feature_maps = [P2, P3, P4, P5]

    anchors = input_anchors

    rpn = build_rpn_model(len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)

    rpn_class_logits, rpn_class, rpn_bbox = [], [], []
    for p in rpn_feature_maps:
        logits, classes, bbox = rpn([p])
        rpn_class_logits.append(logits)
        rpn_class.append(classes)
        rpn_bbox.append(bbox)

    rpn_class_logits = Concatenate(axis=1, name='rpn_class_logits')(rpn_class_logits)   # Batch_size, num_anchors, 2
    rpn_class = Concatenate(axis=1, name='rpn_class')(rpn_class)   # Batch_size, num_anchors, 2
    rpn_bbox = Concatenate(axis=1, name='rpn_bbox')(rpn_bbox)   # Batch_size, num_anchors, 4
    proposal_count = config.POST_NMS_ROIS_INFERENCE

    rpn_rois = ProposalLayer(
        proposal_count=proposal_count,
        nms_threshold=config.RPN_NMS_THRESHOLD,
        name='ROI',
        config=config
    )([rpn_class, rpn_bbox, anchors])

    # 获取classifier的结果
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(
        rpn_rois, mrcnn_feature_maps, input_image_meta,
        config.POOL_SIZE, config.NUM_CLASSES,
        train_bn=config.TRAIN_BN,fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE
    )

    detections = DetectionLayer(config, name='mrcnn_detection')(
        [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta]
    )

    detection_boxes = Lambda(lambda x: x[..., :4])(detections)

    mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                      input_image_meta,
                                      config.MASK_POOL_SIZE,
                                      config.NUM_CLASSES,
                                      train_bn=config.TRAIN_BN)

    model = Model([input_image, input_image_meta, input_anchors],
                        [detections, mrcnn_class, mrcnn_bbox,
                            mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                        name='mask_rcnn')
    return model

