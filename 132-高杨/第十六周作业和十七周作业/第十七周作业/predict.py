import glob
import numpy as np
import torch
import os
import cv2
from unet_model import UNet


if __name__ == '__main__':


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(n_channels=1,n_classes=1)

    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model.pth' , map_location=device))
    # 测试模式
    net.eval()

    # 读取测试图片
    tests_path = glob.glob('test/*.png')
    for  test_path in tests_path:
        # 保存结果地址
        save_res_path = test_path.split('.')[0] + '_res.png'
        #读图
        img = cv2.imread(test_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # b c h w
        img = img.reshape(1,1,img.shape[0],img.shape[1])

        img_tensor = torch.from_numpy(img)
        # tensor 拷贝到device中  只用cpu就拷到cpu中， cuda就拷贝到cuda中
        img_tensor = img_tensor.to(device=device,dtype= torch.float32)
        #预测
        pred = net(img_tensor)

        pred = np.array(pred.data.cpu()[0])[0]

        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0

        cv2.imwrite(save_res_path,pred)




