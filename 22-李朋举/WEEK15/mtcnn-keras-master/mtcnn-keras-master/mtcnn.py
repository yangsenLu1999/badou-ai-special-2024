import h5py
from keras.layers import Conv2D, Input, MaxPool2D, Reshape, Activation, Flatten, Dense, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential
import tensorflow as tf
import numpy as np
import utils
import cv2
import os


# --------------------------#
#    粗略获取人脸框
#    输出bbox位置和是否有人脸
# --------------------------#
def create_Pnet(weight_path):
    # 输入（12, 12, 3）
    input = Input(shape=[None, None, 3])

    # 添加第一个卷积层，卷积核大小为(3, 3)，输出通道数为 10，步长为 1，边界填充为valid。
    '''
  
          (f,f)       h-f+2p         w-f+2p
    (h,w) ------>  ( ------- + 1 ,  -------- + 1)
           s,p           s              s
           
             (3,3)
    12x12x3 -------> (12-3+1 , 12-3+1) = (10 , 10) , 输出通道由卷积核决定, ∴ （10, 10, 10）
              1,1
    '''
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    # 添加 PReLU 激活函数，共享轴为[1, 2]。
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    # 添加最大池化层，池化窗口大小为 2。
    '''
    (10/2, 10/2, 10 )
    '''
    x = MaxPool2D(pool_size=2)(x)

    # 添加第二个卷积层，卷积核大小为(3, 3)，输出通道数为 16，步长为 1，边界填充为valid。
    '''
             (3,3)
    5x5x10 -------> (5-3+1 , 5-3+1) = (3 , 3) , 输出通道由卷积核决定, ∴ （3, 3, 16）
              1,1
    '''
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    # 添加 PReLU 激活函数，共享轴为[1, 2]
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)

    # 添加第三个卷积层，卷积核大小为(3, 3)，输出通道数为 32，步长为 1，边界填充为valid。
    '''
             (3,3)
    3x3x16 -------> (3-3+1 , 3-3+1) = (1 , 1) , 输出通道由卷积核决定, ∴ （1, 1, 32）
              1,1
    '''
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    # 添加 PReLU 激活函数，共享轴为[1, 2]。
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    # 添加分类层，卷积核大小为(1, 1)，输出通道数为 2，激活函数为softmax。
    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    # 加边界框回归层，卷积核大小为(1, 1)，输出通道数为 4。 无激活函数，线性。
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    # 创建模型，将输入层和输出层连接起来。
    model = Model([input], [classifier, bbox_regress])

    # 1. 检查文件路径
    # if os.path.exists(weight_path):
    #     # 文件路径存在，进行打开操作
    #     f = h5py.File(weight_path, "r")
    # else:
    #     print("File path not found!")

    # 2. 确认文件格式
    if h5py.is_hdf5(weight_path):
        # 文件格式正确，进行打开操作
        f = h5py.File(weight_path, "r")
    else:
        print("Invalid file format!")

    # 3. 检查H5PY版本
    # 如果文件路径和文件格式都正确，但仍然无法打开文件，可能是H5PY库的版本与HDF5文件的版本不兼容。可以尝试升级H5PY库或降级HDF5文件版本来解决兼容性问题。
    # 升级：pip install --upgrade h5py   降级： 降级HDF5文件版本则需要使用HDF5工具进行操作


    # 加载预训练的权重
    model.load_weights(weight_path, by_name=True)
    return model


# -----------------------------#
#   mtcnn的第二段
#   精修框
# -----------------------------#
def create_Rnet(weight_path):
    # 接收一个大小为24x24x3的输入图像
    input = Input(shape=[24, 24, 3])

    # 24,24,3 -> 11,11,28
    '''
    添加第一个卷积层，卷积核大小为3x3，输出通道数为 28，步长为 1，边界填充为valid。
    输出形状：(22, 22, 28) （计算过程：(24 - 3 + 1) = 22，(24 - 3 + 1) = 22，卷积核数量为 28）
    '''
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    '''
    添加最大池化层，池化窗口大小为 3，步长为 2，边界填充为same。
    输出形状：(11, 11, 28) （计算过程：(22 - 3) / 2 + 1 = 11，(22 - 3) / 2 + 1 = 11）
    '''
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 11,11,28 -> 4,4,48
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 4,4,48 -> 3,3,64
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)

    # 3,3,64 -> 64,3,3
    x = Permute((3, 2, 1))(x)  # 对特征图进行重排，将通道维度移到最后。
    x = Flatten()(x)  # 将特征图展平为一维向量
    # 576 -> 128
    x = Dense(128, name='conv4')(x)  # 添加一个全连接层，神经元数量为 128。
    x = PReLU(name='prelu4')(x)

    # 128 -> 2 128 -> 4
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)  # 添加分类层，神经元数量为 2，激活函数为softmax。
    bbox_regress = Dense(4, name='conv5-2')(x)  # 添加边界框回归层，神经元数量为 4。
    model = Model([input], [classifier, bbox_regress])  # 创建模型，将输入层和输出层连接起来。
    model.load_weights(weight_path, by_name=True)  # 加载预训练的权重。
    return model


# -----------------------------#
#   mtcnn的第三段
#   精修框并获得五个点
# -----------------------------#
def create_Onet(weight_path):
    input = Input(shape=[48, 48, 3])

    # 48,48,3 -> 23,23,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 23,23,32 -> 10,10,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 8,8,64 -> 4,4,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)

    # 4,4,64 -> 3,3,128
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)
    # 3,3,128 -> 128,12,12
    x = Permute((3, 2, 1))(x)

    # 1152 -> 256
    x = Flatten()(x)
    x = Dense(256, name='conv5')(x)
    x = PReLU(name='prelu5')(x)

    # 鉴别
    # 256 -> 2   256 -> 4   256 -> 10
    classifier = Dense(2, activation='softmax', name='conv6-1')(x)  # 256 -> 2 是否有人脸：2个输出
    bbox_regress = Dense(4, name='conv6-2')(x)  # 256 -> 4 回归得到的框的起始点（或中心点）的瀋 瀌 坐标和框的长宽，4个输出 (x,y,w,h)
    landmark_regress = Dense(10, name='conv6-3')(x)  # 人脸特征点定位：5个人脸特征点的(x,y)坐标，10个输出

    model = Model([input], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)

    return model


class mtcnn():
    def __init__(self):
        '''
        1. 第一层P-Net将经过卷积，池化操作后输出分类（对应像素点是否存在人脸）和回归（回归box)结果。
        '''
        self.Pnet = create_Pnet('model_data/pnet.h5')
        '''
        2. 第二层网络将第一层输出的结果使用非极大抑制（NMS）来去除高度重合的候选框，并将这些候选框放入R-Net中进行精细的操作，拒绝大量错误框，
           再对回归框做校正，并使用NMS去除重合框，输出分支同样两个分类和回归。
        '''
        self.Rnet = create_Rnet('model_data/rnet.h5')
        '''
        3. 最后将R-Net输出认为是人脸的候选框输入到O-Net中再一次进行精细操作，拒绝掉错误的框，
            此时输出分支包含三个分类：
            a. 是否有人脸：2个输出；
            b. 回归：回归得到的框的起始点（或中心点）的xy坐标和框的长宽，4个输出；
            c. 人脸特征点定位：5个人脸特征点的xy坐标，10个输出。
        '''
        self.Onet = create_Onet('model_data/onet.h5')

    def detectFace(self, img, threshold):
        # -----------------------------#
        #   归一化，加快收敛速度
        #   把[0,255]映射到(-1,1)
        # -----------------------------#
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape
        # -----------------------------#
        #   计算原始输入图像
        #   每一次缩放的比例
        # -----------------------------#
        scales = utils.calculateScales(img)

        out = []
        # -----------------------------------------------------------------------------------#
        #   粗略计算人脸框
        #   pnet部分
        #   Pnet 输出的矩形框和置信度得分只是初步的检测结果 - 矩形框（Bounding Boxes）和对应的置信度得分
        # -----------------------------------------------------------------------------------#
        for scale in scales:
            hs = int(origin_h * scale)  # 计算原始图像的高度 origin_h 和宽度 origin_w 乘以当前缩放比例 scale 后的新高度 hs
            ws = int(origin_w * scale)  # 新宽度 ws
            scale_img = cv2.resize(copy_img, (ws, hs))  # 将复制的图像 copy_img 调整大小为新的宽度和高度，得到 scale_img
            inputs = scale_img.reshape(1, *scale_img.shape)  # 重塑为一个形状为 (1, *scale_img.shape) 的输入数组 inputs
            # 图像金字塔中的每张图片分别传入Pnet得到output [一系列的矩形框（Bounding Boxes）和对应的置信度得分（Confidence Scores）] ?
            output = self.Pnet.predict(inputs)
            # 将所有output加入out
            out.append(output)

        image_num = len(scales)  # 缩放比例列表
        rectangles = []  # 存储检测到的人脸矩形框
        for i in range(image_num):  # 遍历缩放比例列表中的每个元素
            # 有人脸的概率
            cls_prob = out[i][0][0][:, :, 1]
            # 其对应的框的位置
            roi = out[i][1][0]

            # 取出每个缩放后图片的长宽
            out_h, out_w = cls_prob.shape  # 获取人脸概率的高度和宽度
            out_side = max(out_h, out_w)  # 获取人脸概率的最大边长
            print(cls_prob.shape)
            # 解码过程  使用 utils.detect_face_12net 函数对人脸概率和人脸框位置进行解码，得到人脸矩形框。
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h,
                                                threshold[0])
            rectangles.extend(rectangle)  # 将人脸矩形框添加到 rectangles 列表中。

        # 进行非极大抑制
        rectangles = utils.NMS(rectangles, 0.7)

        if len(rectangles) == 0:
            return rectangles

        # -------------------------------------------------------------------------#
        #   稍微精确计算人脸框
        #   Rnet部分
        #   通过裁剪、调整大小和使用 Rnet 网络进行预测，对图像中的人脸进行进一步的检测和筛选
        # -------------------------------------------------------------------------#
        predict_24_batch = []  # 存储裁剪和调整大小后的图像
        for rectangle in rectangles:  # 遍历 rectangles 列表中的每个矩形框
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]),
                       int(rectangle[0]):int(rectangle[2])]  # 根据矩形框的坐标从原始图像 copy_img 中裁剪出相应的区域
            scale_img = cv2.resize(crop_img, (24, 24))  # 将裁剪后的图像调整大小为 24x24
            predict_24_batch.append(scale_img)  # 将调整大小后的图像添加到 predict_24_batch 列表中

        predict_24_batch = np.array(predict_24_batch)  # 列表转换为 NumPy 数组
        out = self.Rnet.predict(predict_24_batch)  # 使用 Rnet 网络对 predict_24_batch 进行预测，得到输出结果 out

        # 分别提取输出结果中的类别概率 cls_prob 和感兴趣区域概率 roi_prob , 将类别概率和感兴趣区域概率转换为 NumPy 数组
        cls_prob = out[0]
        cls_prob = np.array(cls_prob)
        roi_prob = out[1]
        roi_prob = np.array(roi_prob)
        # 函数根据类别概率和感兴趣区域概率对矩形框进行筛选和过滤
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])

        if len(rectangles) == 0:
            return rectangles

        # -----------------------------#
        #   计算人脸框
        #   onet部分
        # -----------------------------#
        predict_batch = []  # 存储裁剪和调整大小后的图像
        for rectangle in rectangles:  # 遍历 rectangles 列表中的每个矩形框
            # 根据矩形框的坐标从原始图像 copy_img 中裁剪出相应的区域
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            # 将裁剪后的图像调整大小为 48x48
            scale_img = cv2.resize(crop_img, (48, 48))
            # 将调整大小后的图像添加到 predict_batch 列表中
            predict_batch.append(scale_img)
        # 将 predict_batch 列表转换为 NumPy 数组
        predict_batch = np.array(predict_batch)
        # 使用 Onet 网络对 predict_batch 进行预测，得到输出结果 output。
        output = self.Onet.predict(predict_batch)
        # 分别提取输出结果中的类别概率、感兴趣区域概率和关键点概率
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]
        # 函数根据类别概率、感兴趣区域概率和关键点概率对矩形框进行筛选和过滤
        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])

        return rectangles
