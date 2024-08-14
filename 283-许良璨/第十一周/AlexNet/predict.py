from model.AlexNet import AlexNet
import utils
import cv2
import numpy as np

if __name__=="__main__":
    model=AlexNet()
    model.load_weights("./logs/ep039-loss0.004-val_loss0.652.h5")
    img1=cv2.imread("./Test.jpg")
    img=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)/255
    img_nor = np.expand_dims(img, axis=0)
    img_resize = utils.resize_img(img_nor, (224, 224))

    print(utils.print_answer(np.argmax(model.pridict(img_resize))))
    cv2.imshow("ooo", img1)
    cv2.waitKey(0)