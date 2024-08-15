import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random


class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        # self.data_path=data_path
        # self.imgs_path=glob.glob(os.path.join(data_path,'image/*.png'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        # flip=cv2.flip(image,flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        # image_path = self.imgs_path[index]
        image_path=self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # label_path=image_path.replace('image','label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


# import cv2
# import glob
# import os
# import random
# from torch.utils.data import Dataset
#
#
# class ISBI_Loader(Dataset):
#     def __init__(self, data_path):
#         # 类的初始化方法，用于设置数据集的基本参数
#         # data_path是包含图像和标签文件夹的根目录路径
#         self.data_path = data_path  # 存储根目录路径
#         # 使用glob模块查找data_path下'image'文件夹中所有的'.png'图片文件
#         self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
#         # self.imgs_path现在包含了所有图像文件的路径
#
#     def augment(self, image, flipCode):
#         # 数据增强方法，用于对图像进行翻转操作
#         # image是待增强的图像，flipCode是翻转代码
#         # 使用cv2.flip函数根据flipCode对图像进行翻转
#         flip = cv2.flip(image, flipCode)
#         # 返回翻转后的图像
#         return flip
#
#     def __getitem__(self, index):
#         # 根据索引获取数据集中单个样本的方法
#         # index是样本的索引
#         # 根据index从self.imgs_path中获取对应的图像路径
#         image_path = self.imgs_path[index]
#         # 假设标签文件和图像文件在同一目录下但位于不同的子文件夹，且文件名相同
#         # 通过替换image_path中的'image'为'label'来生成标签路径
#         label_path = image_path.replace('image', 'label')
#         # 使用cv2.imread读取图像和标签文件
#         image = cv2.imread(image_path)
#         label = cv2.imread(label_path)
#         # 将图像和标签从BGR转换为灰度图（如果它们是彩色的）
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
#         # 调整图像和标签的形状以匹配模型输入要求（假设模型需要单通道输入）
#         image = image.reshape(1, image.shape[0], image.shape[1])
#         label = label.reshape(1, label.shape[0], label.shape[1])
#         # 如果标签中的最大值大于1（即原始标签是255表示前景），则进行归一化
#         label = label / 255
#         # 注意：这里的label.max() > 1实际上是有问题的，因为cv2读取的是uint8类型，最大值为255
#         # 正确的归一化应该直接除以255
#
#         # 随机进行数据增强
#         flipCode = random.choice([-1, 0, 1, 2])  # 2表示不进行翻转
#         if flipCode != 2:
#             image = self.augment(image, flipCode)
#             label = self.augment(label, flipCode)
#         # 返回处理后的图像和标签
#         return image, label
#
#     def __len__(self):
#         # 返回数据集中样本的总数
#         # 即self.imgs_path的长度
#         return len(self.imgs_path)


if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("data/train/")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
