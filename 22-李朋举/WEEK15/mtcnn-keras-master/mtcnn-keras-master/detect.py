import cv2
import numpy as np
from mtcnn import mtcnn

img = cv2.imread('img/test1.jpg')  # (378,499,3)

model = mtcnn()
threshold = [0.5,0.6,0.7]  # 三段网络的置信度阈值不同
rectangles = model.detectFace(img, threshold)
draw = img.copy()

for rectangle in rectangles:  # [57.0, 48.0, 84.0, 83.0, 0.9986616373062134, 70.46646049618721, 62.35135769844055, 81.47321730852127, 63.550368666648865, 78.63705557584763, 70.09922099113464, 70.54819625616074, 75.77191126346588, 79.62026077508926, 77.02722406387329]
    if rectangle is not None:
        # 计算矩形的宽度 W 和高度 H，分别为矩形右下角的 x 坐标减去左上角的 x 坐标，以及右下角的 y 坐标减去左上角的 y 坐标
        W = -int(rectangle[0]) + int(rectangle[2])  # 27
        H = -int(rectangle[1]) + int(rectangle[3])  # 35
        # 计算水平方向的填充量 paddingH，为宽度的 1%
        paddingH = 0.01 * W  # 0.27
        # 计算垂直方向的填充量 paddingW，为高度的 2%
        paddingW = 0.02 * H  # 0.7000000000000001
        # 从图像 img 中裁剪出一个矩形区域，该区域的左上角坐标为 (rectangle[1]+paddingH, rectangle[0]-paddingW)，
        #                                   右下角坐标为 (rectangle[3]-paddingH, rectangle[2]+paddingW)。
        crop_img = img[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH), int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]  # (34,28,3)
        if crop_img is None:
            continue
        if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
            continue
        '''
        使用 OpenCV 库的 `cv2.rectangle` 函数在图像上绘制一个矩形,  使用蓝色线条，宽度为 1        
            - `draw`：这是要在其上绘制矩形的图像。
            - `(int(rectangle[0]), int(rectangle[1]))`：这是矩形的左上角坐标。
            - `(int(rectangle[2]), int(rectangle[3]))`：这是矩形的右下角坐标。
            - `(255, 0, 0)`：这是矩形的颜色。在 OpenCV 中，颜色通常以 BGR（蓝、绿、红）顺序表示。这里的 `(255, 0, 0)` 表示蓝色。
            - `1`：这是矩形的边框宽度。
        '''
        cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 1)

        # 对于每个偶数索引 i（从 5 到 15，步长为 2），在图像 draw 上绘制一个绿色的圆，圆心坐标为 (rectangle[i + 0], rectangle[i + 1])，半径为 2。
        for i in range(5, 15, 2):
            cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))

cv2.imwrite("img/out.jpg",draw)

cv2.imshow("test", draw)
c = cv2.waitKey(0)
