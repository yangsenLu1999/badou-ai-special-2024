from mask_rcnn import MASK_RCNN
from PIL import Image

mask_rcnn = MASK_RCNN()

if __name__ == '__main__':
    try:
        image = Image.open("img/street.jpg")
    except:
        print('Open Error!, Try again!')
    else:
        mask_rcnn.detect_image(image)
    mask_rcnn.close_session()
