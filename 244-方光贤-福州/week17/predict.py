import glob
import numpy as np
import torch
import cv2
from model.unet_model import UNet

if __name__ == "__main__":
    # 选择cuda加速 没有cuda就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络 图片同样为单通道1 类别为1 要和训练的图片一致
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce里
    net.to(device=device)
    # 加载最好的一次模型参数
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    # 把网络调整为评估模式
    net.eval()
    # 读取图片路径
    tests_path = glob.glob('data/test/*.png')
    # 遍历图片进行评估
    for test_path in tests_path:
        # 保存结果地址 通过.来分割 这样就没有png后缀了 可以修改文件名
        save_res_path = test_path.split('.')[0] + '_res.png'
        # 读取图片
        img = cv2.imread(test_path)
        # 转为灰度图处理 负责图片为多通道
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 转为batch为1 通道为1 大小为512*512的数组 否则无法处理
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 从numpy数组再重新转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果 [0]用于获取批处理中的第一个（也是唯一的）预测结果 并移除可能的额外维度。
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        cv2.imwrite(save_res_path, pred)