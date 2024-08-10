# 步骤 2: 导入必要的库
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam

# 步骤 3: 加载预训练的VGG16模型
# 加载预训练的VGG16模型（不包括顶层的全连接层）
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 冻结预训练层的权重
for layer in base_model.layers:
    layer.trainable = False


# 步骤 4: 添加自定义层
# 添加自定义层
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # 假设是二分类问题

# 创建新的模型
model = Model(inputs=base_model.input, outputs=predictions)

# 编译模型
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# 步骤 5: 数据增强和模型训练
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'E:\AI\八斗2024精品班\【12】图像识别\代码\homework\Cat&DogRecognitionByVGG',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    # 'path_to_validation_dir',
    'E:\AI\八斗2024精品班\【12】图像识别\代码\homework\Cat&DogRecognitionByVGG',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

# 训练模型
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32)