import numpy as np
import utils
import cv2
from keras import backend as K
from model.AlexNet import AlexNet

K.set_image_dim_ordering('tf')

if __name__ == "__main__":
    model = AlexNet()
    model.load_weights("./logs/ep039-loss0.004-val_loss0.652.h5") # 加载训练模型的权重
    img = cv2.imread("./Test.jpg")
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # 转RGB
    img_nor = img_RGB / 255 # 归一化到[0,1]范围
    img_nor = np.expand_dims(img_nor, axis = 0) # 在第0维度上增加一个维度，变成形状为(1, height, width, channels)的数组
    img_resize = utils.resize_image(img_nor, (224, 224)) # 调整图片大小为(224, 224)
    #utils.print_answer(np.argmax(model.predict(img)))
    print(utils.print_answer(np.argmax(model.predict(img_resize)))) # 进行预测，返回预测结果
    cv2.imshow("ooo",img)
    cv2.waitKey(0)