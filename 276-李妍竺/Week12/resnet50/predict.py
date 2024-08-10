import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
from model.resnet50 import ResNet50
K.set_image_data_format('channels_last')

if __name__ == '__main__':
    model = ResNet50()
    model.summary()  #查看神经网络长什么样
    img_path = 'bike.jpg'
    #img_path = 'elephant.jpg'
    print('img_path', img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    print('img', img)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # 进行数据预处理，归一化，白化等  图像从RGB转成BGR

    print('Input image shape:', x.shape)
    preds = model.predict(x)

    print('Predicted:', decode_predictions(preds))







