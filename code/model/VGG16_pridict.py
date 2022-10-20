from tensorflow.keras.models import load_model
import numpy as np
model = load_model('./VGG16_pre_based_on_dataV4.h5')
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
from tensorflow.keras.preprocessing import image

test_image = image.load_img('D:\\ResearchSpace\\TFL\\data_gan\\qbcy\\L\\211113152938_gan_qbcy411.jpg', target_size = (128, 128)) 
test_image = image.img_to_array(test_image)  # img_to_array 转换前后类型都是一样的，唯一区别是转换前元素类型是整型，转换后元素类型是浮点型(和keras等机器学习框架相适应的图像类型。
test_image = np.expand_dims(test_image, axis = 0)
#predict the result
result = model.predict(test_image)
print(result)
