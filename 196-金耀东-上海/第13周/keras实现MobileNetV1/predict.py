from modules.models import MobileNetV1
from modules.global_params import IMG_SHAPE, NUM_CLASS, PATH_DATA_DIR, PATH_LAST_MODEL
from keras.preprocessing.image import load_img, img_to_array
from os.path import join as path_join
from numpy import stack as np_stack
from keras.applications.mobilenet import preprocess_input, decode_predictions
if __name__ == '__main__':

    # 加载测试数据
    img_bike = img_to_array( load_img(path_join(PATH_DATA_DIR, "bike.jpg"), target_size=IMG_SHAPE[:2]))
    img_car = img_to_array( load_img(path_join(PATH_DATA_DIR, "car.jpg"), target_size=IMG_SHAPE[:2]))
    img_elephant = img_to_array(load_img(path_join(PATH_DATA_DIR, "elephant.jpg"), target_size=IMG_SHAPE[:2]))
    x_test = preprocess_input( np_stack([img_bike, img_car, img_elephant]) )

    # 创建神经网络
    model = MobileNetV1(input_shape=IMG_SHAPE, output_dim=NUM_CLASS)

    # 展示模型结构
    model.summary()

    # 加载训练好的模型参数
    model.load_weights(PATH_LAST_MODEL)

    # 进行预测
    y_preds = model.predict(x_test)

    # 展示测试结果
    for i, prediction in enumerate( decode_predictions(y_preds) ):
        print(f"img{i+1}:")
        print(f"    prediction:{prediction}")
