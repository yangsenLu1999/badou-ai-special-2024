"""
定义全局参数
"""
import os
import torch

PATH_BASE = "."                                         # 根目录,根据情况修改
PATH_DATA = os.path.join(PATH_BASE, "data")             # 训练数据与测试路径存放在data下
PATH_MODEL_DATA = os.path.join(PATH_BASE, "model_data") # 训练模型存放在logs下
PATH_SAMPLES = os.path.join(PATH_DATA, "samples")       # 测试样本保存路径

PATH_PNET_WEIGHT = os.path.join(PATH_MODEL_DATA, "PNet_weight.pth") # PNet预训练权重存放路径
PATH_RNET_WEIGHT = os.path.join(PATH_MODEL_DATA, "RNet_weight.pth") # ONet预训练权重存放路径
PATH_ONET_WEIGHT = os.path.join(PATH_MODEL_DATA, "ONet_weight.pth") # RNet预训练权重存放路径

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 设备

SCORE_THRES_PNET= 0.5   # PNet置信度阈值
SCORE_THRES_RNET = 0.6  # RNet置信度阈值
SCORE_THRES_ONET = 0.7  # ONet置信度阈值
IOU_THRESHOLD = 0.5     # IOU阈值

MIN_SIZE = 12                                           # 输入图片最小边不小于12
MIN_FACE_SIZE = (int(MIN_SIZE*0.7) , int(MIN_SIZE*0.7)) # 人脸最小尺寸

SCALE_FACTOR = 0.709        # Pnet图像金字塔缩放系数
STRIDE_PNET = 2.0           # Pnet步幅
CELL_SIZE = 12              # Pnet网格感受野
IMG_SIZE_RNET = (24,24)     # Rnet输入图像尺寸
IMG_SIZE_ONET = (48,48)     # Onet输入图像尺寸
