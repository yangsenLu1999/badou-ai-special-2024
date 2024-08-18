import myresnet50
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

if __name__ == '__main__':
    mode = myresnet50.my_resnet_50()
    mode.summary()
    image_path = 'elephant.jpg'
    img = image.load_img(image_path, target_size=(224, 224))
    image_array = image.img_to_array(img)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    print('Input image shape:', image_array.shape)
    preds = mode.predict(image_array)
    print('Predicted:', decode_predictions(preds))
