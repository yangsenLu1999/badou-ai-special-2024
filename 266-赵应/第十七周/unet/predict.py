import glob

import cv2

from model.unet import get_model

if __name__ == '__main__':
    model = get_model(1, 1)
    model.load_weights("inception_traffic_light.h5")
    # 读取所有图片路径
    tests_path = glob.glob('data/test/*.png')
    # 遍历素有图片
    for test_path in tests_path:
        # 保存结果地址
        save_res_path = test_path.replace("test", "predict")
        # 读取图片
        img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img / 255.0
        img = img.reshape(1, img.shape[0], img.shape[1], 1)
        pred = model.predict(img)
        # 提取结果
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        cv2.imwrite(save_res_path, pred[0])
