from mask_rcnn import MASK_RCNN
from PIL import Image

# 导入网络
mask_rcnn = MASK_RCNN()

while True:
    img = input('img/street.jpg')
    # 添加读取图片的异常处理
    try:
        image = Image.open('img/street.jpg')
    except:
        print('Open Error! Try again!')
        continue
    else:
        # 无异常就进行检测
        mask_rcnn.detect_image(image)
mask_rcnn.close_session()
    
