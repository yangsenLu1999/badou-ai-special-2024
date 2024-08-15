# 3.推理
# 不管是什么模型都可以用这一套代码
import numpy as np   # 通常用于进行数学运算和数值计算，尤其是处理数组和矩阵
import utils         # 自定义模块
import cv2           # 计算机视觉任务和图像处理
from keras import backend as K      # 从 Keras 中导入后端模块（backend），使你能够与 Keras 的底层设置进行交互
from model.AlexNet import AlexNet   # import上一个文件中自己写的,从一个名为 model 的模块中导入 AlexNet 类。这是一个自定义的模型定义，通常包含神经网络的架构和设置

# K.set_image_dim_ordering('tf')    # 设置图片的维度顺序,这个是h,w,c的
K.image_data_format() == 'channels first'   # 表示是c,h,w,不是h,w,c

if __name__ == "__main__":  # 用于判断该脚本是否被直接运行。如果是直接运行，该脚本内的代码块将被执行，通常用于运行测试
    model = AlexNet()   # 创建一个 AlexNet 模型的实例。此时 model 对象包含了网络结构
    # 只是这里和下一行换，换成别的训练出来的模型，比如ABC，就是调用自己的模型结构，当然图片可以换
    model.load_weights("./logs/ep039-loss0.004-val_loss0.652.h5")    # 训练好的权重文件load进来
    # 加载训练好的模型权重，使用指定路径的文件（"./logs/ep039-loss0.004-val_loss0.652.h5"）。这是一个经过训练的模型，可以直接用于预测
    img = cv2.imread("./Test.jpg")
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_nor = img_RGB/255  # 标准化
    img_nor = np.expand_dims(img_nor,axis = 0)
    # 扩展图像数组的维度，在第一维增加一个维度，结果形成的形状为 (1, height, width, channels)，、
    # 这是模型输入所需的格式（通常是一个批量的输入）
    img_resize = utils.resize_image(img_nor,(224,224))  # 图像调整为指定的大小 (224, 224)。
    # 这一过程通常是在 AlexNet 等模型中要求的，以便模型能够接收正确尺寸的输入
    #utils.print_answer(np.argmax(model.predict(img)))
    print(utils.print_answer(np.argmax(model.predict(img_resize))))
    '''model.predict(img_resize) 会产生模型的输出，
    np.argmax() 会返回预测结果中概率最大的类的索引。
    最后，print_answer 函数用于打印人类可读的预测结果。'''
    cv2.imshow("ooo",img)
    cv2.waitKey(0)