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
    # 假设输入(12, 12, 3)   Pnet为全卷积神经网络，输入可多尺度
    input = Input(shape=[None, None, 3])   # Tensor("input_1:0", shape=(?, ?, ?, 3), dtype=float32)

    # 添加第一个卷积层，卷积核大小为(3, 3)，输出通道数为 10，步长为 1，边界填充为valid。
    '''
  
          (f,f)       h-f+2p         w-f+2p
    (h,w) ------>  ( ------- + 1 ,  -------- + 1)
           s,p           s              s
           
             (3,3)
    12x12x3 -------> (12-3+1 , 12-3+1) = (10 , 10) , 输出通道由卷积核决定, ∴ （10, 10, 10）
              1,1
    '''
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)  # Tensor("conv1/BiasAdd:0", shape=(?, ?, ?, 10), dtype=float32)
    # 添加 PReLU 激活函数，共享轴为[1, 2]。
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)  # Tensor("PReLU1/add:0", shape=(?, ?, ?, 10), dtype=float32)
    # 添加最大池化层，池化窗口大小为 2。
    '''
    (10/2, 10/2, 10 )
    '''
    x = MaxPool2D(pool_size=2)(x)  # Tensor("max_pooling2d_1/MaxPool:0", shape=(?, ?, ?, 10), dtype=float32)

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
    # if h5py.is_hdf5(weight_path):
    #     # 文件格式正确，进行打开操作
    #     f = h5py.File(weight_path, "r")
    # else:
    #     print("Invalid file format!")
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
    input = Input(shape=[24, 24, 3])  # Tensor("input_2:0", shape=(?, 24, 24, 3), dtype=float32)

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
    # 3,3,128 -> 128,3,3
    x = Permute((3, 2, 1))(x)  # Tensor("permute_2/transpose:0", shape=(?, 128, 3, 3), dtype=float32)

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
        """
        1. 第一层P-Net将经过卷积，池化操作后输出分类（对应像素点是否存在人脸）和回归（回归box)结果。
        """
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
        copy_img = (img.copy() - 127.5) / 127.5  # (378,499,3) [[[-0.12156863 -0.1372549  -0.1372549 ], ...
        origin_h, origin_w, _ = copy_img.shape  # origin_h -> 378  origin_w -> 499  _ -> 3
        # -----------------------------#
        #   计算原始输入图像
        #   每一次缩放的比例
        # -----------------------------#
        scales = utils.calculateScales(img)
        # scales-> {list:11} [1.002004008016032, 0.7104208416833666, ..., 0.032161167842200765]

        out = []
        # -----------------------------------------------------------------------------------------#
        #   粗略计算人脸框
        #   pnet部分
        #   Pnet 第一层P-Net将经过卷积，池化操作后输出分类（对应像素点是否存在人脸）和回归（回归box)结果
        # -----------------------------------------------------------------------------------------#
        for scale in scales:  # 1.002004008016032
            hs = int(origin_h * scale)  # 计算原始图像的高度 hs  378 * 1.002004008016032 ≈ 378
            ws = int(origin_w * scale)  # 新宽度 ws ≈ 499
            scale_img = cv2.resize(copy_img, (ws, hs))  # 将复制的图像 copy_img 调整大小为新的宽度和高度，得到 scale_img (378,499,3)
            inputs = scale_img.reshape(1, *scale_img.shape)  # 重塑为一个形状为 (1, *scale_img.shape) 的输入数组 inputs (1,378,499,3)
            # 图像金字塔中的每张图片分别传入Pnet得到output
            output = self.Pnet.predict(inputs)
            '''
            {list:2}
              分类（对应像素点是否存在人脸）-> (1,184,244,2) [[[[0.9951456  0.00485442],   [0.99255896 0.00744099], 
              回归（回归box) -> (1,184,244,4) [[[[-0.0249768   0.01182976 -0.01099911  0.3040731 ],   [-0.0371667   0.00759065 -0.02612752  0.2954429 ], 
            '''
            # 将所有output加入out {list:11}
            out.append(output)

        image_num = len(scales)  # 缩放比例列表  {list:11}
        rectangles = []  # 存储检测到的人脸矩形框
        for i in range(image_num):  # 遍历缩放比例列表中的每个比例, 取出 每张图片的对应像素点的 分类和回归
            # 分类（对应像素点是否存在人脸）  有人脸的概率
            '''
                out[i][0][0] 表示取出 out 列表中第 i 个元素的第一个元素的第一个元素，这是一个三维数组，
                [:, :, 1]    表示取出该三维数组的所有行和列，以及索引为 1 的第三维元素，即取出了该三维数组中所有位置上的第二个元素，这是一个二维数组
                cls_prob ->  (184,244) -> [[0.00485442 0.00744099 0.01235654 ... 0.0062199  0.00941012 0.01643363], [0.00473192 0.00622642 0.00710934 ... 0.00173196 0.00245852 0.00385891], 
            '''
            cls_prob = out[i][0][0][:, :, 1]
            # 回归（回归box)  其对应的框的位置
            '''
                roi(x,y,w,h) ->   (184,244,4) -> [[[-2.49768011e-02  1.18297562e-02 -1.09991208e-02  3.04073066e-01],  [-3.71667072e-02  7.59064406e-03 -2.61275135e-02  2.95442939e-01],  [-3.63098793e-02 -2.05379967e-02 -4.59153801e-02  2.27480307e-01],  ...,  [-1.37009658e-03 -1.62871584e-01 -4.16391008 
            '''
            roi = out[i][1][0]

            # 取出每个缩放后图片的长宽
            out_h, out_w = cls_prob.shape  # 获取高度和宽度  out_h 244  out_w 184
            out_side = max(out_h, out_w)  # 获取最大边长  out_side 244
            # 解码过程  使用 utils.detect_face_12net 函数对人脸概率和人脸框位置进行解码，得到人脸矩形框。
            #          第一张图片    {list:187} [[379.0, 41.0, 390.0, 52.0, 0.997905969619751], [400.0, 131.0, 410.0, 142.0, 0.9949091076850891], ...
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h,
                                                threshold[0])
            # 将人脸矩形框添加到 rectangles 列表中。 {list:450}
            rectangles.extend(rectangle)

        # 进行非极大抑制  {list：442} [[285.0, 19.0, 319.0, 54.0, 0.9999483823776245], [319.0, 33.0, 353.0, 67.0, 0.999729573726654], ...
        rectangles = utils.NMS(rectangles, 0.7)

        if len(rectangles) == 0:
            return rectangles

        # -------------------------------------------------------------------------#
        #   稍微精确计算人脸框
        #   Rnet部分
        #   通过裁剪、调整大小和使用 Rnet 网络进行预测，对图像中的人脸进行进一步的检测和筛选
        # -------------------------------------------------------------------------#
        predict_24_batch = []  # 存储裁剪和调整大小后的图像
        for rectangle in rectangles:  # 遍历 rectangles 列表中的每个矩形框   [285.0, 19.0, 319.0, 54.0, 0.9999483823776245]
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]),
                       int(rectangle[0]):int(rectangle[2])]  # 根据矩形框的坐标从原始图像 copy_img 中裁剪出相应的区域
            scale_img = cv2.resize(crop_img, (24, 24))  # 将裁剪后的图像调整大小为 24x24
            predict_24_batch.append(scale_img)  # 将调整大小后的图像添加到 predict_24_batch 列表中

        predict_24_batch = np.array(predict_24_batch)  # 列表转换为 NumPy 数组  (442,24,24,3)
        out = self.Rnet.predict(predict_24_batch)  # 使用 Rnet 网络对 predict_24_batch 进行预测，得到输出结果 out

        # 分别提取输出结果中的类别概率 cls_prob 和感兴趣区域概率 roi_prob , 将类别概率和感兴趣区域概率转换为 NumPy 数组
        cls_prob = out[0]
        cls_prob = np.array(cls_prob)
        roi_prob = out[1]
        roi_prob = np.array(roi_prob)
        # 函数根据类别概率和感兴趣区域概率对矩形框进行筛选和过滤 {list:11}
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
        cls_prob = output[0]  # (11,2)  [[0.01723065 0.98276937], [0.04660699 0.95339304], [0.00133841 0.99866164], [0.00922056 0.9907795 ], [0.09331412 0.9066859 ], [0.07324829 0.92675173], [0.88085246 0.1191475 ], [0.9560685  0.04393155], [0.59710276 0.40289724], [0.6569412  0.34305882], [0.02
        roi_prob = output[1]  # (11,4)  [[ 0.09243366 -0.01021912 -0.09710566  0.0259289 ], [ 0.17422554  0.05714664 -0.16603123 -0.044655  ], [ 0.11624187  0.01893055 -0.08297026  0.05577926], [ 0.15789732 -0.00504237 -0.14608765  0.00219431], [ 0.13483942  0.01055627 -0.00600987  0.13053264],
        pts_prob = output[2]  # (11,10) [[0.46909094 0.8089122  0.7219902  0.49472332 0.76121044 0.35374138,  0.3655261  0.57597923 0.780297   0.79586655], [0.513709   0.7867383  0.7386793  0.5177196  0.7485817  0.39714235,  0.43470615 0.57667303 0.7568735  0.7871812 ], [0.49898365 0.83252174 0.
        # 函数根据类别概率、感兴趣区域概率和关键点概率对矩形框进行筛选和过滤
        # (11,7) [[57.0, 48.0, 84.0, 83.0, 0.9986616373062134, 70.46646049618721, 62.35135769844055, 81.47321730852127, 63.550368666648865, 78.63705557584763, 70.09922099113464, 70.54819625616074, 75.77191126346588, 79.62026077508926, 77.02722406387329],
        #         [323.0, 34.0, 347.0, 71.0, 0.9907795190811157, 335.48441219329834, 48.95401322841644, 345.3862200975418, 50.88470482826233, 342.9990200996399, 57.39930558204651, 334.9761750102043, 62.90861439704895, 342.8069304227829, 64.42674684524536],
        #         ... ]
        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])

        return rectangles
