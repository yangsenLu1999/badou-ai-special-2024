from model.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions,preprocess_input
import numpy as np
if __name__ == '__main__':
    model = ResNet50()
    model.load_weights("./net/resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    model.summary()

    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))