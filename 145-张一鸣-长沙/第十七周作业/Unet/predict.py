import glob
import numpy as np
import torch
import os
import cv2
from Unet.model.unet import Unet


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Unet(n_channels=1, n_classes=1)
    net.to(device=device)   # 将网络拷贝到deivce中
    net.load_state_dict((torch.load('best_model.pth', map_location=device)))
    net.eval()

    img_path = glob.glob(r'data/test/*.png')

    for path in img_path:
        res_path = path.split('.')[0] + '_res.png'
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)

        predict = net(img_tensor)
        predict = np.array(predict.data.cpu()[0])[0]

        predict[predict >= 0.5] = 255
        predict[predict < 0.5] = 0
        cv2.imwrite(res_path, predict)
