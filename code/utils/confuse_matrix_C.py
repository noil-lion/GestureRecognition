from sklearn.metrics import confusion_matrix
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import itertools
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import confusionMatrix as cm


number = 93
data_path = r'D:/ResearchSpace/TFL/dataRaw/test'
model = load_model('../comparedMethod/model_WXY.h5')
# 模型加载
 
# 混淆矩阵绘制
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.binary,#这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
                          normalize=True):
   
 
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
	#plt.savefig('/content/drive/My Drive/Colab Notebooks/confusionmatrix32.png',dpi=350)
    plt.show()

def pridict(model, data_path):
    pridiction = []
    test_path_list = os.listdir(data_path)
    for K in test_path_list:
        test_class_path_list = os.listdir(data_path+'/'+K)
        for i in test_class_path_list[0:number]:
            test_image = image.load_img(data_path+'/'+K+'/'+i, target_size = (1156, 1), color_mode = "grayscale") 
            test_image = image.img_to_array(test_image)  # img_to_array 转换前后类型都是一样的，唯一区别是转换前元素类型是整型，转换后元素类型是浮点型(和keras等机器学习框架相适应的图像类型。
            test_image = np.expand_dims(test_image, axis = 0)
            #predict the result
            result = model.predict(test_image)
            pridiction.append(result)
        # print(pridiction)
    return pridiction


# 显示混淆矩阵
def plot_confuse(model, data_path, y_val, labels):
    predictions = pridict(model, data_path)
    truelabel = np.argmax(y_val, axis=-1)   # 将one-hot转化为label
    predictions = np.argmax(predictions, axis=-1)
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plot_confusion_matrix(conf_mat, normalize=True,target_names=labels,title='Confusion Matrix')
    return conf_mat
#=========================================================================================
#最后调用这个函数即可。 test_x是测试数据，test_y是测试标签（这里用的是One——hot向量）
#labels是一个列表，存储了你的各个类别的名字，最后会显示在横纵轴上。
#比如这里我的labels列表
labels=['action1','action2','action3','action4','action5']
test_y =      [[1.00,0.00,0.00,0.00,0.00]]*number   # 初始状态
test_y.extend([[0.00,1.00,0.00,0.00,0.00]]*number)  # qbcy
test_y.extend([[0.00,0.00,1.00,0.00,0.00]]*number)  # qbxq
test_y.extend([[0.00,0.00,0.00,1.00,0.00]]*number)  # szcs
test_y.extend([[0.00,0.00,0.00,0.00,1.00]]*number)  # szqs






conf_mat = plot_confuse(model, data_path, test_y, labels)
class_indict = {'0': 'Initial state','1':'Lumbar touch', '2': 'Forearm pronation', '3': 'Arm side extension', '4':'Arm forward extension'}
label = [label for _, label in class_indict.items()]
confusion = cm.ConfusionMatrix(num_classes=5, labels=label)
#实例化混淆矩阵，这里NUM_CLASSES = 5
confusion.update(conf_mat)
confusion.plot()
confusion.summary()

