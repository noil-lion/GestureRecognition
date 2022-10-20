#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import multi_gpu_model


# 常量的定义
train_dir = "../dataV6/train"
valid_dir = "../dataV6/test"
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
height = 128  # resne50的处理的图片大小
width = 128  # resne50的处理的图片大小
channels = 3
batch_size = 32  # 因为处理的图片变大,batch_size变小一点 32->24
num_classes = 5

# 一,使用keras中ImageDataGenerator读取数据
# 1,实例化ImageDataGenerator
# 对于图片数据,在keras里有更高层的封装.读取数据且做数据增强 -> Generator
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.resnet50.preprocess_input,  # 此函数是是现在keras中,而非tf.keras中,在tf中,实现数据做归一化,数据取值在-1~1之间.
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
    preprocessing_function=keras.applications.resnet50.preprocess_input)
valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(height, width),
    batch_size=batch_size,
    seed=7,
    shuffle=False,
    class_mode="categorical")
"""
resnet50_fine_tune = keras.models.Sequential()
resnet50_fine_tune.add(
    keras.applications.ResNet50(
        include_top=False,  # 网络结构的最后一层,resnet50有1000类,去掉最后一层
        pooling='avg',  # resnet50模型倒数第二层的输出是三维矩阵-卷积层的输出,做pooling或展平
        weights='D:/ResearchSpace/TFL/Fes/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    ))  # 参数有两种imagenet和None,None为从头开始训练,imagenet为从网络下载已训练好的模型开始训练
resnet50_fine_tune.add(keras.layers.Dense(
    num_classes, activation='softmax'))  # 因为include_top = False,所以需要自己定义最后一层
resnet50_fine_tune.layers[0].trainable = False  # 因为参数是从imagenet初始化的,所以我们可以只调整最后一层的参数

resnet50_fine_tune.compile(
    loss="categorical_crossentropy", optimizer="sgd",
    metrics=['accuracy'])  # 对于微调-finetune来说,优化器使用sgd来说更好一些
resnet50_fine_tune.summary()"""

#
resnet50 = keras.applications.ResNet50(
    include_top=False,
    pooling='avg',
    weights=None
)
resnet50.summary()


resnet50_new = keras.models.Sequential([
    resnet50,
    keras.layers.BatchNormalization(),
    keras.layers.Dense(num_classes, activation='softmax'),
])
resnet50_new.compile(
    loss="categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])
resnet50_new.summary()

train_num = train_generator.samples
valid_num = valid_generator.samples
epochs = 30

# history = resnet50_new.fit_generator(
history = resnet50_new.fit_generator(
    train_generator,  # 生成器函数， 生成器函数的输出应为：（inputs， tergets）的数组
    steps_per_epoch=train_num // batch_size, # int， 当生成器返回N次数据是，一个epoch结束，执行下一个epoch。
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=valid_num // batch_size)
resnet50_new = multi_gpu_model(resnet50_new, 2)  #GPU个数为2
resnet50_new.save('resnet50_complt_'+str(epochs)+'_.h5')
