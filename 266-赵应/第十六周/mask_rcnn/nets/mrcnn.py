from keras import Input
from keras.layers import Conv2D, UpSampling2D, Add, MaxPooling2D, Reshape, Activation, Concatenate, TimeDistributed, \
    BatchNormalization, Lambda, Dense, Conv2DTranspose
from keras.models import Model

from nets.layers import ProposalLayer, PyramidROIAlign, DetectionLayer
from nets.resnet import get_resnet
from keras import backend as K


def get_predict_model(config):
    h, w = config.IMAGE_SHAPE[:2]
    # 输入进来的图片必须是2的6次方以上的倍数
    if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")
    input_image = Input(shape=[1024, 1024, config.IMAGE_SHAPE[2]], name='input_image')
    # meta包含了一些必要信息
    input_image_meta = Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")
    # 先验框
    input_anchors = Input(shape=[None, 4], name='input_anchors')

    # 使用resnet网络提取公共特征，返回4种抽象程度不同的特征
    _, C2, C3, C4, C5 = get_resnet(input_image, stage5=True, train_bn=config.TRAIN_BN)

    # 将C2~C5组合成FPN（特征金字塔网络）
    # 32x32
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, kernel_size=(1, 1), name='fpn_c5p5')(C5)

    P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, kernel_size=1, name='fpn_c4p4')(C4)
    upsampling = UpSampling2D(size=(2, 2), name='fpn_p5upsampled')(P5)
    # 64x64
    P4 = Add(name='fpn_p4add')([upsampling, P4])

    P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, kernel_size=1, name='fpn_c3p3')(C3)
    upsampling1 = UpSampling2D(size=2, name='fpn_p4upsampled')(P4)
    # 128x128
    P3 = Add(name='fpn_p3add')([P3, upsampling1])

    P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, kernel_size=1, name='fpn_c2p2')(C2)
    upsampling2 = UpSampling2D(size=2, name='fpn_p3upsampled')(P3)
    # 256x256
    P2 = Add(name='fpn_p2add')([P2, upsampling2])

    # 特征金字塔每层进行一次256的通道卷积使得其通道数相同
    P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, kernel_size=3, padding='SAME', name='fpn_p2')(P2)
    P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, kernel_size=3, padding='SAME', name='fpn_p3')(P3)
    P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, kernel_size=3, padding='SAME', name='fpn_p4')(P4)
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, kernel_size=3, padding='SAME', name='fpn_p5')(P5)
    # 32x32 -> 16x16
    P6 = MaxPooling2D(pool_size=1, strides=2, name='fpn_p6')(P5)

    # P2~P6送入RPN网络生成提议框
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    # P2~P5送入FCN网络生成mask掩膜（mask分支）
    mrcnn_feature_maps = [P2, P3, P4, P5]

    anchors = input_anchors
    # 建立RPN网络
    rpn = build_rpn_model(len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)

    # 获得RPN网络的预测结果，进行格式调整，把五个特征层的结果进行堆叠
    rpn_class_logits, rpn_class, rpn_bbox = [], [], []
    for p in rpn_feature_maps:
        logits, probs, bbox =rpn([p])
        rpn_class_logits.append(logits)
        rpn_class.append(probs)
        rpn_bbox.append(bbox)

    # 将list类型的结果拼接为一个张量， axis=1 代表从第一个维度拼接，第0个维度是batch_size
    rpn_class_logits = Concatenate(axis=1, name="rpn_class_logits")(rpn_class_logits)
    rpn_class = Concatenate(axis=1, name='rpn_probs')(rpn_class)
    rpn_bbox = Concatenate(axis=1, name='rpn_bbox')(rpn_bbox)

    # 此时获得的rpn_class_logits、rpn_class、rpn_bbox的维度是
    # rpn_class_logits : Batch_size, num_anchors, 2
    # rpn_class : Batch_size, num_anchors, 2
    # rpn_bbox : Batch_size, num_anchors, 4
    proposal_count = config.POST_NMS_ROIS_INFERENCE
    # Batch_size, proposal_count, 4
    # 对先验框进行解码
    rpn_rois = ProposalLayer(
        proposal_count=proposal_count,
        nms_threshold=config.RPN_NMS_THRESHOLD,
        name="ROI",
        config=config)([rpn_class, rpn_bbox, anchors])

    # 获得classifier的结果
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
        fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                             config.POOL_SIZE, config.NUM_CLASSES,
                             train_bn=config.TRAIN_BN,
                             fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

    detections = DetectionLayer(config, name="mrcnn_detection")(
        [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

    detection_boxes = Lambda(lambda x: x[..., :4])(detections)
    # 获得mask的结果
    mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                      input_image_meta,
                                      config.MASK_POOL_SIZE,
                                      config.NUM_CLASSES,
                                      train_bn=config.TRAIN_BN)

    # 作为输出
    model = Model([input_image, input_image_meta, input_anchors],
                  [detections, mrcnn_class, mrcnn_bbox,
                   mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                  name='mask_rcnn')
    return model





# ------------------------------------#
#   建立建议框网络模型
#   RPN模型
# ------------------------------------#
def build_rpn_model(anchors_per_location, depth):
    input_feature_map = Input(shape=[None, None, depth],
                              name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location)
    return Model([input_feature_map], outputs, name="rpn_model")


# ------------------------------------#
#   五个不同大小的特征层会传入到
#   RPN当中，获得建议框
# ------------------------------------#
def rpn_graph(feature_map, anchors_per_location):
    """
    TimeDistributed:
    对FPN网络输出的多层卷积特征进行共享参数。
    TimeDistributed的意义在于使不同层的特征图共享权重。
    :param feature_map:
    :param anchors_per_location: 每个锚点对应的预选框个数
    :return:
    """
    # 3x3x512的滑动窗口提取特征，用于roi框分类（预选框内是否有对象）及（对象）位置检测
    shared = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name='rpn_conv_shared')(feature_map)

    # 进行分类预测
    # 将特征结构修改为batch_size*num_anchors*2的形式，便于后续进行softmax分类。2代表预选框内是否有对象，参数-1代表自动进行维度推断
    x = Conv2D(2 * anchors_per_location, kernel_size=1, activation='linear', name='rpn_class_raw')(shared)
    rpn_class_logits = Reshape([-1, 2])(x)
    rpn_probs = Activation('softmax', name='rpn_class_xxx')(rpn_class_logits)

    # 进行对象位置预测
    x = Conv2D(anchors_per_location * 4, kernel_size=1, activation='linear', name='rpn_bbox_pred')(shared)
    # 预选框参数调整
    rpn_bbox = Reshape([-1, 4])(x)
    return [rpn_class_logits, rpn_probs, rpn_bbox]


#------------------------------------#
#   建立classifier模型
#   这个模型的预测结果会调整建议框
#   获得最终的预测框
#------------------------------------#
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