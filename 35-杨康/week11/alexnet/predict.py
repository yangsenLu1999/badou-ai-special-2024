import numpy as np
import cv2
import utils
from AlexNet import AlexNet
from keras import backend as K

K.image_data_format() == 'channels_first'
if __name__ == '__main__':
    model = AlexNet()
    model.load_weights('./logs/last1.h5')
    img = cv2.imread('Test.jpg')
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_nor = img_rgb/255
    img_nor = np.expand_dims(img_nor,axis=0)
    img_resize = utils.resize_image(img_nor,(224,224))
    argmax = np.argmax(model.predict(img_resize))
    utils.print_answer(argmax)
    cv2.imshow('ooo',img)
    cv2.waitKey(0)


