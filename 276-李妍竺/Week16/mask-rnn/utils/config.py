import numpy as np

class Config(object):
    """
    基本配置类。对于自定义配置，请创建从该子类继承并重写属性的子类
    这需要改变。
    """
    # 命名配置
    NAME = None

    # 要使用的GPU数量，当仅使用CPU时，需要将其设置为1
    GPU_COUNT = 1

    # 在每个GPU上训练的图像数。12GB GPU通常可以处理2个1024x1024px的图像。
    # 根据GPU内存和图像大小进行调整。使用最高GPU可以处理的数字以获得最佳性能。
    IMAGES_PER_GPU = 2

    # 每代的训练迭代次数
    # 这不需要与训练集的大小相匹配。Tensorboard 更新保存在每个epoch的末尾，因此将其设置为数字越小，TensorBoard的更新频率就越高。
    # 验证统计数据也会在每代结束时进行计算  可能需要一段时间，所以不要把这个设定得太小以避免支出
    # 在验证统计数据上花了很多时间。
    STEPS_PER_EPOCH = 1000   # 每代的训练迭代次数

    # 在每个训练时期结束时要运行的验证步骤数。
    # 每个世代的训练迭代次数中的有效次数（次数越大准确率会提高，但训练会更好耗时）
    VALIDATION_STEPS = 50

    # 骨干网络架构
    # 支持的值为：resnet50、resnet101。
    # 也可以提供一个具有 model.resnet_graph 签名的可调用程序。如果这样做，您还需要为 COMPUTE_BACKBONE_SHAPE 提供一个可调用函数
    BACKBONE = 'resnet101'

    # None #只有调用 backbone 时有效，用于计算每一层 特征金字塔网络的 shape
    # 参阅model.compute_backline_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # 每层特征金字塔网络 的步长（基于 Resnet101）
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # 分类图中完全连接层的size
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # 用于构建要素棱锥体的自上而下的图层的大小
    TOP_DOWN_PYRAMID_SIZE = 256

    # 分类类别数量（包括背景）
    NUM_CLASSES = 1

    # 方形锚定边的长度（像素） 候选区域网络的anchor大小
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # 每个单元的锚固件比率（宽度/高度）
    # 值1表示方形锚点，0.5表示宽锚点
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # 锚定步幅
    # 如果为1，则为主干特征图中的每个单元创建锚点。
    # 如果为2，则每隔一个单元创建一个锚点，依此类推。
    RPN_ANCHOR_STRIDE = 1

    # 用于过滤RPN建议的非最大抑制阈值。  过滤 候选区域框
    # 可以在训练中增加这一点，以产生更多的推动力。
    RPN_NMS_THRESHOLD = 0.7

    # 每个图像要用于RPN训练的锚数量  每个图片有多少anchors 用于 RPN 训练
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # 在tf.nn.top_k之后和非最大抑制之前保留的ROI
    PRE_NMS_LIMIT = 6000

    # 非最大抑制（训练和推理）后保留的ROI
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # 如果启用，则将实例掩码调整为较小的大小以减小内存负载。建议在使用高分辨率图像时使用。
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)

    # 输入图像大小调整

    IMAGE_RESIZE_MODE = "square"  #一般为 square，调整并通过填充0得到一个正方形图像 在大多数情况下它应该运行良好
    IMAGE_MIN_DIM = 800  # 在此模式下，图像被缩放向上使得小边 = IMAGE_MIN_DIM，但确保 缩放不会使长边> IMAGE_MAX_DIM
    IMAGE_MAX_DIM = 1024 # 然后用零填充以使其成为正方形，以便可以放置多个图像到一个批次中
    '''
    其他模式 None：返回不做处理的图像
           pad64：pad的宽度和高度用零表示，使它们成为 64 的倍数，如果IMAGE_MIN_DIM或IMAGE_MIN_SCALE不是 None
                 则它会在填充之前向上扩展。在此模式下IMAGE_MAX_DIM将被忽略。 需要 64 的倍数来确保功能的平滑缩放
                 映射 FPN 金字塔的 6 个级别 （2**6=64）
           crop：从图像中选取随机裁剪。首先，根据IMAGE_MIN_DIM和IMAGE_MIN_SCALE缩放图像，
                 然后随机选择尺寸 IMAGE_MIN_DIM x IMAGE_MIN_DIM。只能在训练中使用。
                 在此模式下不使用IMAGE_MAX_DIM
     '''

    # 最小缩放比率。检查MIN_IMAGE_DIM后，可以强制进一步扩展。例如
    # 如果设置为 2，则图像将放大到宽度和高度的两倍或更多，即使MIN_IMAGE_DIM不要求它。
    # 但是，在“square”模式下，它可以被IMAGE_MAX_DIM否决

    IMAGE_MIN_SCALE = 0

    # 每个图像的颜色通道数。RGB=3，灰度=1，RGB-D=4
    # 更改此项需要对代码进行其他更改。查看WIKI了解更多信息 详细信息：https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3

    # 图像平均值（RGB）
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # 每个图像要馈送到分类器/掩模头的ROI数量
    # Mask RCNN论文使用512，但RPN通常不会生成
    # 足够的 positive 候选框来保证 pos ：neg
    # 比例为1：3。可以通过调整NMS阈值来增加候选框数量
    # RPN NMS阈值。
    TRAIN_ROIS_PER_IMAGE = 200

    # positive ROIS 用于训练 分类/掩膜 训练的百分比
    ROI_POSITIVE_RATIO = 0.33

    # Pool 所需 候选框数
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # 输出掩膜大小（改变这个神经网络中的掩膜分支也要改变）
    MASK_SHAPE = [28, 28]

    # 在一个图像中使用的真实例的最大数量
    MAX_GT_INSTANCES = 100

    # RPN和最终检测的边界框细化标准偏差。
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # 最终检测的最大次数/数量
    DETECTION_MAX_INSTANCES = 100

    # 接受检测到的实例的最小概率值    跳过低于此阈值的ROI
    DETECTION_MIN_CONFIDENCE = 0.7

    # 检测的非最大抑制阈值
    DETECTION_NMS_THRESHOLD = 0.3

    # 学习率和动力
    # Mask RCNN论文使用lr=0.02，但在TensorFlow上它会导致要爆炸的重量。可能是由于优化器的差异执行。
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # 权重衰减正则化
    WEIGHT_DECAY = 0.0001

    # 损失权重以实现更精确的优化。
    # 可用于R-CNN训练设置。
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # 使用 RPN ROI 或外部生成的 ROI 进行训练
    # 在大多数情况下保持此 True
    # 如果要根据代码生成的 ROI 而不是 RPN 的 ROI 来训练分支，请设置为 False。例如，调试分类器而无需训练 RPN。
    USE_RPN_ROIS = True

    # 训练或冻结批量归一化层
    # None：训练 BN 层。这是正常模式
    # False：冻结 BN 图层。使用小批量时很好
    # True：（不使用）即使在预测时也设置训练模式中的图层（Set layer in training mode even when predicting）
    TRAIN_BN = False  #默认设置为 False(因为批量一般都很小)

    # 渐变范数剪裁
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
        """设置计算属性的值"""
        # 有效批量大小
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # 输入图片大小
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                                         self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                                         self.IMAGE_CHANNEL_COUNT])

        # 图象元数据长度
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        """显示配置信息"""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")