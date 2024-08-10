from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

# 假设你已经保存了模型，这里我们使用`model`变量来表示加载的模型
# model = load_model('path_to_your_saved_model.h5')  # 如果你保存的是HDF5文件

# 加载并预处理图片
img_path = 'path_to_your_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 使用模型进行预测
preds = model.predict(x)

# 解释输出
# 因为这里是一个二分类问题，我们使用sigmoid激活函数，所以可以直接查看输出的概率
cat_probability = preds[0][0]  # 假设0表示猫的概率
dog_probability = 1 - cat_probability  # 狗的概率是1减去猫的概率

# 打印结果
if cat_probability > 0.5:
    print("Predicted: Cat")
else:
    print("Predicted: Dog")

# 如果你想要更详细的输出（虽然在这个二分类场景中可能不太需要）
# print(f"Cat Probability: {cat_probability:.2f}, Dog Probability: {dog_probability:.2f}")