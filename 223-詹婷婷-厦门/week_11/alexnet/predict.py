import numpy as np
import utils
import cv2
from keras import backend as K
import tensorflow as tf

from model.Alexnet import AlexNet

K.set_image_dim_ordering('tf')

if __name__ == "__main__":
    model = AlexNet()
    model.load_weights("./logs/last1.h5")
    img = cv2.imread("./img.png")
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nor = img_RGB / 255
    img_nor = np.expand_dims(img_nor, axis=0) #在数组的指定位置增加一个新的维度,当axis为0时，新维度插入到最前面；当axis为正数时，新维度插入到指定位置；当axis为负数时，新维度插入到从尾部算起的指定位置
    img_resize = utils.resize_image(img_nor, (224, 224))
    print(utils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow("img", img)
    cv2.waitKey()

#
# if __name__ == "__main__":
#     model = AlexNet()
#     model.load_weights("H:/CV/PRE/pythonProject1/week_11/alexnet/logs/ep003-loss0.546-val_loss0.674.h5")
#     img = cv2.imread("./Test.jpg")
#     img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     img_nor = img_RGB/255
#     img_nor = np.expand_dims(img_nor,axis = 0)
#     img_resize = utils.resize_image(img_nor,(224,224))
#     #utils.print_answer(np.argmax(model.predict(img)))
#     print(utils.print_answer(np.argmax(model.predict(img_resize))))
#     cv2.imshow("ooo",img)
#     cv2.waitKey(0)