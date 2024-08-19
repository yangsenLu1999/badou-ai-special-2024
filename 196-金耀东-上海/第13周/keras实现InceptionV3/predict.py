from modules import models
from modules.global_params import PATH_DATA_DIR, IMG_SHAPE, PATH_LAST_MODEL, NUM_CLASS
from os.path import join as path_join
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import decode_predictions, preprocess_input
import numpy as np

if __name__ == '__main__':
    # 加载测试数据
    img_elephant = img_to_array( load_img(path_join(PATH_DATA_DIR, "elephant.jpg"), target_size=IMG_SHAPE[:2]) )
    img_bike = img_to_array(load_img(path_join(PATH_DATA_DIR, "bike.jpg"), target_size=IMG_SHAPE[:2]) )
    img_car = img_to_array(load_img(path_join(PATH_DATA_DIR, "car.jpg"), target_size=IMG_SHAPE[:2]) )
    x_test = preprocess_input( np.stack([img_elephant, img_bike, img_car]) ) # 像素标准化至（-1，1）

    # 创建神经网络
    model = models.InceptionV3(input_shape=IMG_SHAPE, output_dim=NUM_CLASS)

    # 输出模型结构
    model.summary()

    # 加载训练好的模型参数
    model.load_weights(PATH_LAST_MODEL)

    # 进行预测
    y_preds = model.predict(x_test)

    # 展示预测结果
    predictions = decode_predictions(y_preds)
    for i, prediction in enumerate(predictions):
        print(f"img{i+1}:")
        print(f"    prediction:{prediction}")

