import os
import torch
# ------------------------------------------------------#
#   定义全局参数
# ------------------------------------------------------#

# 定义各种文件路径
PATH_BASE = "." # 根目录,根据情况修改
PATH_DATA = os.path.join(PATH_BASE, "data") # 训练数据与测试路径存放在data下
PATH_MODEL_DATA = os.path.join(PATH_BASE, "model_data") # 训练模型存放在logs下
PATH_LAST_MODEL = os.path.join(PATH_MODEL_DATA, "yolo_weights.pth") # 最终模型存放路径
PATH_PRE_TRAINED_MODEL = os.path.join(PATH_MODEL_DATA, "darknet53_backbone_weights.pth") # 预训练模型存放路径
PATH_SAMPLES = os.path.join(PATH_DATA, "samples") # 存放待检测图片与检测结果

# 模型相关参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 有GPU时选择GPU否则选择CPU
INPUT_SHAPE = (416, 416, 3) # 输入图片shape:（h,w,c)
NUM_CLASS = 80
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# COCO数据集类型标签
COCO_CLASSES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "trafficlight",
             "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
             "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
             "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard",
             "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
             "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "sofa",
             "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
             "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
             "teddybear", "hairdrier", "toothbrush"]

# 根据K-means计算得到的COCO数据集的候选框尺寸
BASE_ACHORS = [
            [[116,90],  [156,198],  [373,326]], # 13*13
            [[30,61],  [62,45],  [59,119]], # 26*26
            [[10,13],  [16,30],  [33,23]], # 52*52
        ]
