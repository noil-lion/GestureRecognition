print("starting...")
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
import numpy as np
import os
import pandas as pd

import time
from tensorflow import keras
# 调用此次训练的数据集
import numpy as np
import pandas as pd
from tensorflow.keras.utils import multi_gpu_model
from keras.layers.recurrent import LSTM
# 1,实现tensorflow动态按需分配GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# 加载数据
# 常量的定义
train_dir = "../../dataV6/train"
valid_dir = "../../dataV6/test"
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
num_classes = 5  # 增加了静止态

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

# 调用模型
from keras.models import Sequential

# 用于模型初始化，Conv2D模型初始化、Activation激活函数，MaxPooling2D是池化层
# Flatten作用是将多位输入进行一维化
# Dense是全连接层
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense

# 优化方法选用Adam(其实可选项有很多，如SGD)
from keras.optimizers import Adam

# 初始化一个模型
model = Sequential()

# 模型卷积层设计
model.add(Conv2D(
    nb_filter=32,  # 第一层设置32个滤波器
    nb_row=5,
    nb_col=5,  # 设置滤波器的大小为5*5
    padding='same',  # 选择滤波器的扫描方式，即是否考虑边缘
    input_shape=(height,width,channels),  # 设置输入的形状
))

# 选择激活函数
model.add(Activation('relu'))

# 设置下采样(池化层）
model.add(MaxPool2D(
    pool_size=(2,2),  # 下采样格为2*2
    strides=(2,2),  # 向右向下的步长
    padding='same', # padding mode is 'same'
))

model.add(Conv2D(64, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(strides=(2, 2), padding='same'))
model.add(Flatten())
# 添加LSTM

# 在建设一层
model.add(Dense(5))  # 输入是个类别
model.add(Activation('softmax'))  # 用于分类的softmax函数

adam = Adam()  # 学习速率lr=0.0001

model.compile(optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model.summary()

x = model.layers[-5].output_shape
print('输出维度：'+str(x))
train_num = train_generator.samples
valid_num = valid_generator.samples
epochs = 30
# history = VGG16_fine_tune.fit_generator(train_generator,
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_num // batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=valid_num // batch_size)
model.save("CNN.h5")
pd.DataFrame(history.history).to_csv('training_history_CNN'+str(epochs)+'dataV6log.csv', index=False)
