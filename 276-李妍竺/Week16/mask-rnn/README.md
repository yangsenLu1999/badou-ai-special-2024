# FCN
**Fully Convolutional Networks**

全卷积神经网络


**原理:**

FCN将传统卷积网络后面的全连接层换成了卷积层，这样网络输出不再是类别而是heatmap，同时为了解决卷积核池化对图像尺寸的影响，提出使用上采样的方式恢复尺寸

**核心思想** ： 

   - 不含全连接层的全卷积网络。**可适应任意尺寸输入**
   - 增大数据尺寸的**反卷积层**。**能够输出精细的结果**
   - **进行像素级的分类**

上采样：放大图片，增加图片的分辨率 将提取到的feature map 进行放大
下采样：降维L减少图片尺寸，减轻计算量，防止过拟合。  池化pooling

## 反卷积

*作用：*

采用**反卷积**对最后一个卷积层的feature map **进行上采样**，使其**恢复到输入图像相同的尺寸**。从而可以对**每个像素**都产生一个预测，同时保留了原始输入图像中的空间信息，最后**在上采样的特征图上进行逐像素分类**



# mask-rcnn-keras
mask-rcnn的库，可以用于训练自己的实例分割模型。

## Mask R-CNN 过程
1. Backbone (resnet101)
2. RPN
3. ProposalLayer
4. DetectionTargetLayer
5. ROIAlign
6. bbox检测
7. Mask分割


- backbone： Resnet101为主干
- 在进行特征提取后，利用长宽压缩了两次、三次、四次、五次的特征层来进行**特征金字塔（FPN)** 结构的构造
- RPN部分，会将五种不同尺寸大小的A南充人是，分别在P2~P6这五个特征图上生成，并且有一个锚点对应三种长宽比例


# 训练用权重
训练好的权重是基于coco数据集的，可以直接运行用于coco数据集的实例分割。   

# 测试训练用的数据集
数据集是用于分辨图片中的圆形、正方形、三角形的，格式已经经过了处理，可以让大家明白训练集的格式。  

# 使用方法
## 1、准备数据集
a、利用labelme标注数据集，注意标注的时候同一个类要用不同的序号，比如画面中存在**两个苹果那么一个苹果的label就是apple1另一个是apple2。**    
b、标注完成后将jpg文件和json文件放在根目录下的before里面。  
c、之后运行json_to_dataset.py就可以生成train_dataset文件夹了。  
## 2、修改训练参数
a、dataset.py内修改自己要分的类，分别是load_shapes函数和load_mask函数内和类有关的内容，即将原有的circle、square等修改成自己要分的类。    
b、在train文件夹下面修改ShapesConfig(Config)的内容，NUM_CLASS等于自己要分的类的数量+1。  
c、IMAGE_MAX_DIM、IMAGE_MIN_DIM、BATCH_SIZE和IMAGES_PER_GPU根据自己的显存情况修改。RPN_ANCHOR_SCALES根据IMAGE_MAX_DIM和IMAGE_MIN_DIM进行修改。  
d、STEPS_PER_EPOCH代表每个世代训练多少次。   
## 3、预测
a、测试时运行predict即可，img内存在测试文件street.jpg。  
b、测试自身代码时将_defaults里面的参数修改成训练时用的参数。  


# py文件
## mask_rcnn
构建MASK_RCNN文件
1. 初始化
2. 获得所有分类
3. 生成模型
4. 检测图片
5. 关闭网络

## 网络架构相关文件 (net)
### resnet101
关键点：C1-C5

### layer
将先验框转化成建议框

ProposalLayer:

将RPN的输出作为输入，先删除一部分超阶地ROI，然后进行排序，保留其中预测为前景概率大的一部分

DetectionTargetLayer:

判断正负样本，计算正样本中的anchor哪和哪一个真实框最接近，计算二者的偏移值

### mrcnn
搭建网络模型
1. 将五个不同大小的特征层传到RPN中，获得建议框
2. 建立RPN模型
3. 建立classifier模型（这个模型的预测结果会调整建议框，获得最终的预测框）
4. 建立ROIAlign模型
5. 建立预测模型
6. 建立训练模型

### mrcnn_training
训练过程
1. 损失函数
- smooth li
- rpn_class_loss
- rpn_bbox_loss
- mrcnn_class_loss
- mrcnn_bbox_loss
- mrcnn_mask_loss
2. 载入图片并处理
3. rpn目标：判断正负样本
4. 生成数据



