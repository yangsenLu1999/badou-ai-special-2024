import numpy as np
import math
from utils.utils import norm_boxes


# ----------------------------------------------------------#
#  Anchors
# ----------------------------------------------------------#
def generate_anchors(scale, ratios, shape, feature_stride, anchor_stride):
    # 获取所有框的长度和比例的组合
    scale, ratios = np.meshgrid(np.array(scale), np.array(ratios))
    # 例如[4, 4, 4]
    scale = scale.flatten()
    # [0.5, 0.5, 0.5, 1, 1, 1, 2, 2, 2]
    ratios = ratios.flatten()
    # 9种不同的锚框比
    heights = scale / np.sqrt(ratios)
    widths = scale * np.sqrt(ratios)

    # 生成网格中心(映射到原图需要乘以stride)
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # 获得先验框的中心和宽高
    # shifts_x,shifts_y的网格上每个网格生成了9个不同的先验框
    box_widths, box_center_x = np.meshgrid(widths, shifts_x)
    box_heights, box_center_y = np.meshgrid(heights, shifts_y)

    # 变更格式
    box_centers = np.stack([box_center_y, box_center_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # 计算出y1, x1, y2, x2
    boxes = np.concatenate([box_centers - .5 * box_sizes,
                            box_centers + .5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides, anchor_stride):
    """
    生成不同特征层的anchors，利用concatenate进行堆叠在一起
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    # P2对应的scale是32
    # P3对应的scale是64
    # P4对应的scale是128
    # P5对应的scale是256
    # P6对应的scale是512
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i], feature_strides[i],
                                        anchor_stride))
    return np.concatenate(anchors, axis=0)


def compute_backbone_shapes(config, image_shape):
    # 用于计算主干特征提取网络的shape
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)
    # 其实就是计算P2、P3、P4、P5、P6这些特征层的宽和高
    assert config.BACKBONE in ['resnet50', 'resnet101']
    return np.array([
        [int(math.ceil(image_shape[0] / stride)), int(math.ceil(image_shape[1] / stride))]
        for stride in config.BACKBONE_STRIDES
    ])


def get_anchors(config, image_shape):
    backbone_shapes = compute_backbone_shapes(config, image_shape)
    anchor_cache = {}
    if not tuple(image_shape) in anchor_cache:
        a = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                     config.RPN_ANCHOR_RATIOS,
                                     backbone_shapes,
                                     config.BACKBONE_STRIDES,
                                     config.RPN_ANCHOR_STRIDE)
        anchor_cache[tuple(image_shape)] = norm_boxes(a, image_shape[:2])
    return anchor_cache[tuple(image_shape)]