import sys
from operator import itemgetter
import numpy as np
import cv2
import matplotlib.pyplot as plt


# -----------------------------#
#   计算原始输入图像
#   每一次缩放的比例
# -----------------------------#
def calculateScales(img):
    copy_img = img.copy()  # 创建一个与输入图像 img 相同的副本
    pr_scale = 1.0  # 初始化缩放比例为 1.0
    h, w, _ = copy_img.shape  # 获取图像的高度 h、宽度 w  通道 _  : origin_h -> 378  origin_w -> 499  _ -> 3
    # 先把图像缩放到一定大小(500，500), 再通过factor对这个大小进行缩放，可以减少计算量。保证图像金字塔在一定的尺度空间内 不小于12不大于500
    # 引申优化项  = resize(h*500/min(h,w), w*500/min(h,w))
    if min(w, h) > 500:  # 如果图像的最小边长大于 500
        pr_scale = 500.0 / min(h, w)  # 计算缩放比例
        w = int(w * pr_scale)  # 将宽度乘以缩放比例进行缩(放)
        h = int(h * pr_scale)  # 将高度乘以缩放比例进行缩(放)
    elif max(w, h) < 500:
        pr_scale = 500.0 / max(h, w)  # 1.002004008016032
        w = int(w * pr_scale)  # 499
        h = int(h * pr_scale)  # 378

    scales = []  # 存储计算得到的缩放比例
    factor = 0.709  # 定义一个缩放因子 factor
    factor_count = 0  # 初始化缩放因子的计数器
    minl = min(h, w)  # 获取图像的最小边长  378
    while minl >= 12:  # 只要最小边长大于等于 12
        # 将当前的缩放比例 pr_scale 乘以缩放因子的幂次方，并添加到 scales 列表中。 [pow(factor, factor_count) -> 计算 factor 的 factor_count 次幂]
        scales.append(pr_scale * pow(factor, factor_count))  # {list:11} [1.002004008016032]  [1.002004008016032, 0.7104208416833666] ...
        minl *= factor  # 将最小边长乘以缩放因子  268.002  190.013418 ... 8.602018789437398
        factor_count += 1  # 递增缩放因子的计数器 1 2 .. 11
    return scales # [1.002004008016032, 0.7104208416833666, 0.5036883767535069, 0.3571150591182364, 0.25319457691482955, 0.17951495503261417, 0.12727610311812346, 0.09023875711074951, 0.06397927879152139, 0.04536130866318867, 0.032161167842200765]


# ---------------------------------------------------------------------------------------------------------------------------------------------------------#
#   对pnet处理后的结果进行处理
#      在输入的 cls_prob 和 roi 数据中检测人脸，并返回检测到的人脸矩形框及其置信度得分。函数中涉及了一系列的坐标计算、边界框调整和非极大值抑制等操作，以实现准确的人脸检测
# ---------------------------------------------------------------------------------------------------------------------------------------------------------#
def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    cls_prob = np.swapaxes(cls_prob, 0, 1)  # 交换 cls_prob 数组的维度
    roi = np.swapaxes(roi, 0, 2)  # 交换 roi 数组的维度

    stride = 0  # 初始化步长变量 stride 为 0
    # stride略等于2
    if out_side != 1:  # 如果输出边长 out_side 不等于 1，则计算步长 stride 的值
        stride = float(2 * out_side - 1) / (out_side - 1)
    (x, y) = np.where(cls_prob >= threshold)  # 使用 np.where 函数找到 cls_prob 中大于或等于阈值 threshold 的元素的坐标 (x, y)

    boundingbox = np.array([x, y]).T  # 将坐标 (x, y) 转换为列向量，并存储在 boundingbox 数组中。
    # 找到对应原图的位置  计算边界框的左上角坐标 bb1 和右下角坐标 bb2。
    bb1 = np.fix((stride * (boundingbox) + 0) * scale)
    bb2 = np.fix((stride * (boundingbox) + 11) * scale)
    # plt.scatter(bb1[:,0],bb1[:,1],linewidths=1)
    # plt.scatter(bb2[:,0],bb2[:,1],linewidths=1,c='r')
    # plt.show()
    # 将左上角坐标和右下角坐标连接起来，形成完整的边界框。
    boundingbox = np.concatenate((bb1, bb2), axis=1)

    # 获取边界框的偏移量
    dx1 = roi[0][x, y]
    dx2 = roi[1][x, y]
    dx3 = roi[2][x, y]
    dx4 = roi[3][x, y]

    # 获取对应位置的置信度得分
    score = np.array([cls_prob[x, y]]).T

    # 将偏移量存储在 offset 数组中。
    offset = np.array([dx1, dx2, dx3, dx4]).T

    # 根据偏移量调整边界框的位置
    boundingbox = boundingbox + offset * 12.0 * scale

    # 将边界框和置信度得分连接起来，形成矩形数组
    rectangles = np.concatenate((boundingbox, score), axis=1)

    # 将矩形转换为正方形
    rectangles = rect2square(rectangles)

    pick = []  # 存储最终筛选后的矩形框
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))  # 获取矩形框左上角的横坐标，并确保不小于 0。
        y1 = int(max(0, rectangles[i][1]))  # 获取矩形框左上角的纵坐标，并确保不小于 0。
        x2 = int(min(width, rectangles[i][2]))  # 获取矩形框右下角的横坐标，并确保不大于图像的宽度 width。
        y2 = int(min(height, rectangles[i][3]))  # 获取矩形框右下角的纵坐标，并确保不大于图像的高度 height。
        sc = rectangles[i][4]  # 获取矩形框的置信度得分
        # 如果矩形框的右下角横坐标大于左上角横坐标，且右下角纵坐标大于左上角纵坐标，则将该矩形框的坐标和置信度得分添加到 pick 列表中
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    # 使用 NMS 函数对 pick 列表进行处理，并设置阈值为 0.3,  返回经过 NMS 处理后的结果
    return NMS(pick, 0.3)


# -----------------------------#
#   将长方形调整为正方形
# -----------------------------#
def rect2square(rectangles):
    # 计算矩形的宽度 w 和高度 h
    w = rectangles[:, 2] - rectangles[:, 0]
    h = rectangles[:, 3] - rectangles[:, 1]
    # 取宽度 w 和高度 h 中的最大值，并将其转换为列向量 l。
    l = np.maximum(w, h).T
    # 调整矩形的左上角坐标，使其位于正方形的中心。
    rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
    rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5
    # 根据正方形的边长 l，计算矩形的右下角坐标
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T
    # 返回转换后的正方形坐标
    return rectangles


# ---------------------------------------------------------------------------------------------------------------#
#   非极大抑制
#      在检测到多个重叠的目标时，只保留置信度最高的一个，以避免重复检测。通过计算重叠比例并与阈值进行比较，可以筛选出需要保留的目标。
# ---------------------------------------------------------------------------------------------------------------#
def NMS(rectangles, threshold):  # rectangles 表示要进行非极大值抑制的矩形框列表，threshold 表示阈值
    if len(rectangles) == 0:
        return rectangles
    # 将 rectangles 转换为 NumPy 数组 boxes，并提取出每个矩形框的坐标和置信度得分
    boxes = np.array(rectangles)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    # 计算每个矩形框的面积 area
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    # 获取置信度得分的排序索引 I
    I = np.array(s.argsort())
    # 存储最终保留的矩形框索引
    pick = []
    while len(I) > 0:
        # 每次循环取出置信度最高的矩形框索引 I[-1]
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])  # I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        # 计算与其他矩形框的重叠区域的坐标
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        # 计算重叠区域的面积 inter
        inter = w * h
        # 计算重叠比例 o
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        # 如果重叠比例小于等于阈值 threshold，则将该矩形框的索引添加到 pick 列表中
        pick.append(I[-1])
        # 更新索引 I，只保留重叠比例小于阈值的矩形框索引
        I = I[np.where(o <= threshold)[0]]
    # 循环结束后，将保留的矩形框索引对应的矩形框从 boxes 中取出，并转换为列表形式返回
    result_rectangle = boxes[pick].tolist()
    return result_rectangle


# ------------------------------------------------------------------------------------------------------------------#
#   对Rnet处理后的结果进行处理
#      根据类别概率和感兴趣区域对人脸进行筛选，去除低概率的人脸，并使用非极大值抑制去除重叠的人脸框。最终返回筛选后的人脸矩形框
# ------------------------------------------------------------------------------------------------------------------#
def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):  # 根据类别概率和感兴趣区域对人脸进行筛选和过滤
    prob = cls_prob[:, 1]  # 提取类别概率中的第二列（通常表示人脸的概率）
    pick = np.where(prob >= threshold)  # 找到概率大于或等于阈值的索引
    rectangles = np.array(rectangles)  # 将矩形框转换为 NumPy 数组

    # 根据索引 pick 从矩形框数组中提取相关的坐标和概率
    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]

    sc = np.array([prob[pick]]).T

    # 计算偏移量 dx1、dx2、dx3 和 dx4
    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    # 调整矩形框的坐标
    w = x2 - x1
    h = y2 - y1

    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    # 将坐标和概率连接成一个数组
    rectangles = np.concatenate((x1, y1, x2, y2, sc), axis=1)
    # 将矩形框转换为正方形
    rectangles = rect2square(rectangles)
    # 存储最终筛选后的矩形框
    pick = []
    # 遍历所有矩形框
    for i in range(len(rectangles)):
        # 计算矩形框的左上角和右下角坐标 , 确保坐标在图像范围内
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        # 如果矩形框有效，则将其添加到 pick 列表中
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    # 使用非极大值抑制（NMS）算法对 pick 列表进行处理，并设置阈值为 0.3
    return NMS(pick, 0.3)


# -------------------------------------------------------------------------------------------------------------------#
#   对onet处理后的结果进行处理
#   根据类别概率、感兴趣区域和关键点对人脸进行筛选，去除低概率的人脸，并使用非极大值抑制去除重叠的人脸框。最终返回筛选后的人脸矩形框和关键点
# -------------------------------------------------------------------------------------------------------------------#
def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):
    prob = cls_prob[:, 1]  # 提取类别概率中的第二列（通常表示人脸的概率）
    pick = np.where(prob >= threshold)  # 找到概率大于或等于阈值的索引
    rectangles = np.array(rectangles)  # 将矩形框转换为 NumPy 数组

    # 根据索引 pick 从矩形框数组中提取相关的坐标、概率和关键点
    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]

    sc = np.array([prob[pick]]).T

    # 计算偏移量 dx1、dx2、dx3 和 dx4
    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    # 调整矩形框的坐标
    w = x2 - x1
    h = y2 - y1

    pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
    pts1 = np.array([(h * pts[pick, 5] + y1)[0]]).T
    pts2 = np.array([(w * pts[pick, 1] + x1)[0]]).T
    pts3 = np.array([(h * pts[pick, 6] + y1)[0]]).T
    pts4 = np.array([(w * pts[pick, 2] + x1)[0]]).T
    pts5 = np.array([(h * pts[pick, 7] + y1)[0]]).T
    pts6 = np.array([(w * pts[pick, 3] + x1)[0]]).T
    pts7 = np.array([(h * pts[pick, 8] + y1)[0]]).T
    pts8 = np.array([(w * pts[pick, 4] + x1)[0]]).T
    pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T

    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    # 将坐标、概率和关键点连接成一个数组
    rectangles = np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9),
                                axis=1)
    # 创建一个空列表 pick，用于存储最终筛选后的矩形框
    pick = []
    # 遍历所有矩形框: 计算矩形框的左上角和右下角坐标, 确保坐标在图像范围内, 如果矩形框有效，则将其添加到 pick 列表中
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, rectangles[i][4],
                         rectangles[i][5], rectangles[i][6], rectangles[i][7], rectangles[i][8], rectangles[i][9],
                         rectangles[i][10], rectangles[i][11], rectangles[i][12], rectangles[i][13], rectangles[i][14]])
    # 使用非极大值抑制（NMS）算法对 pick 列表进行处理，并设置阈值为 0.3
    return NMS(pick, 0.3)
