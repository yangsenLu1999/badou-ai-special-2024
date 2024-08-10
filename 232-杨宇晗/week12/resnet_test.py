import torchvision

from resnaet_18.a2 import Resnet18
import torch
from PIL import Image

# 读取图像
img = Image.open("9.jpg")
# 数据预处理

# 缩放
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(img)
print(image.shape)

# 根据保存方式加载
model = torch.load("tudui_38.pth", map_location=torch.device('cpu'))

# 注意维度转换，单张图片
image1 = torch.reshape(image, (1, 3, 32, 32))

# 测试开关
model.eval()
# 节约性能
with torch.no_grad():
    output = model(image1)
print(output)
# print(output.argmax(1))
# 定义类别对应字典
dist = {0: "飞机", 1: "汽车", 2: "鸟", 3: "猫", 4: "鹿", 5: "狗", 6: "青蛙", 7: "马", 8: "船", 9: "卡车"}
# 转numpy格式,列表内取第一个
a = dist[output.argmax(1).numpy()[0]]
img.show()
print(a)
