#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import sys
import tensorflow as tf
import time
from tensorflow import keras
from keras.applications.vgg16 import VGG16


# 打印使用的python库的版本信息
print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

# 1,实现tensorflow动态按需分配GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# 常量的定义
train_dir = "../../newdata/train"
valid_dir = "../../newdata/test"
# label_file = "D:/ResearchSpace/TFL/newdata/labels.txt"
print(os.path.exists(train_dir))
print(os.path.exists(valid_dir))
# print(os.path.exists(label_file))

print(os.listdir(train_dir))
print(os.listdir(valid_dir))

# 查看打印出来的label值
# labels = pd.read_csv(label_file, header=0)
# print(labels)

# 定义常量
height = 128  # VGG16的处理的图片大小
width = 128  # VGG16的处理的图片大小
channels = 3
batch_size = 32  # 因为处理的图片变大,batch_size变小一点 32->24
num_classes = 4

# 一,使用keras中ImageDataGenerator读取数据
# 1,实例化ImageDataGenerator
# 对于图片数据,在keras里有更高层的封装.读取数据且做数据增强 -> Generator
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    # preprocessing_function=keras.applications.vgg16.VGG16.preprocess_input,  # 此函数是是现在keras中,而非tf.keras中,在tf中,实现数据做归一化,数据取值在-1~1之间.
    width_shift_range=0.2,  # 做水平位移 - 增加位移鲁棒性(如果0~1之间则位移比例随机选数做位移;如果大于1,则是具体的像素大小)
    height_shift_range=0.2,  # 做垂直位移 - 增加位移鲁棒性(如果0~1之间则位移比例随机选数做位移;如果大于1,则是具体的像素大小)
    shear_range=0.2,  # 剪切强度
    zoom_range=0.2,  # 缩放强度
    fill_mode='nearest',  # 填充像素规则,用离其最近的像素点做填充
)
# 2,使用ImageDataGenerator读取图片
# 从训练集的文件夹中读取图片
train_generator = train_datagen.flow_from_directory(
    train_dir,  # 图片的文件夹位置
    target_size=(height, width),  # 将图片缩放到的大小
    batch_size=batch_size,  # 多少张为一组
    seed=7,  # 随机数种子
    shuffle=True,  # 是否做混插
    class_mode="categorical")  # 控制目标值label的形式-选择onehot编码后的形式
# 从验证集的文件夹中读取图片
valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    # preprocessing_function=keras.applications.vgg16.VGG16.preprocess
    )
valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(height, width),
    batch_size=batch_size,
    seed=7,
    shuffle=False,
    class_mode="categorical")

#
VGG16 = keras.applications.vgg16.VGG16(
    include_top=False,
    pooling='avg',
    weights= 'imagenet'
)
VGG16.summary()

#
for layer in VGG16.layers[0:-3]:
    layer.trainable = False

VGG16_new = keras.models.Sequential([
    VGG16,
    keras.layers.Dense(num_classes, activation='softmax'),
])
VGG16_new.compile(
    loss="categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])
VGG16_new.summary()

train_num = train_generator.samples
valid_num = valid_generator.samples
epochs = 30
# history = VGG16_fine_tune.fit_generator(train_generator,
history = VGG16_new.fit_generator(
    train_generator,
    steps_per_epoch=train_num // batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=valid_num // batch_size)
VGG16_new.save("VGG16_complt.h5")

# 四,打印模型训练曲线
def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_'+label] = history.history['val_'+label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()


plot_learning_curves(history, 'acc', epochs, 0, 1)
plot_learning_curves(history, 'loss', epochs, 0, 2)
