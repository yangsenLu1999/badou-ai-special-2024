# coding = utf-8

'''
    mask r-cnn的推理
'''

from keras.layers import Input
from mask_rcnn import MASK_RCNN
from PIL import Image

mask_rcnn = MASK_RCNN()

while True:
    img = input(r'./img/street.jpg')
    try:
        image = Image.open(r'./img/street.jpg')
    except:
        print('打开图片失败，请重新尝试！')
        continue
    else:
        mask_rcnn.detect_image(image)

mask_rcnn.close_session()
