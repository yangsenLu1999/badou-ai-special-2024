import glob
import numpy as np
import cv2
from model.unet_model import UNet as network
from keras import backend as K

if __name__ == "__main__":
    # 加载网络，图片单通道，分类为1。
    model = network(n_channels=1, n_classes=1)
    # 加载模型参数
    model.net.load_weights('best_model.h5')
    # 读取所有图片路径
    tests_path = glob.glob('data/test/*.png')
    # 遍历素有图片
    for test_path in tests_path:
        # 保存结果地址
        save_res_path = test_path.split('.')[0] + '_res.png'
        # 读取图片
        img = cv2.imread(test_path)
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = K.variable(img)
        # 预测
        pred = model.net.predict(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        cv2.imwrite(save_res_path, pred)