import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 定义图像尺寸
IMG_SIZE = 48  # FER-2013 数据集图像大小
BATCH_SIZE = 32  # 批次大小
EPOCHS = 20  # 最大训练轮次

# 数据目录
train_dir = 'archive/train'
test_dir = 'archive/test'

# 创建数据生成器并进行图像增强
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 归一化到[0,1]
    shear_range=0.2,  # 剪切角度范围
    zoom_range=0.2,  # 缩放范围
    horizontal_flip=True  # 随机水平翻转
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 加载训练和测试数据
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',  # 确保加载灰度图像
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # 因为是多类别分类
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',  # 确保加载灰度图像
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # 因为是多类别分类
    shuffle=False  # 测试集不需要打乱顺序
)

# 获取类别数
num_classes = len(train_generator.class_indices)

# 创建模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),  # 使用单通道输入
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # 多类别分类交叉熵
              metrics=['accuracy'])

# 设置早停（防止过拟合）和模型保存回调
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

# 训练模型
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=test_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, model_checkpoint]
)

# 评估模型
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# 保存模型
model.save('emotion_recognition_model.keras')
