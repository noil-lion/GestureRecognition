
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import multi_gpu_model

dst_path = r'D:/ResearchSpace/TFL/dataV6/valid'
model_file = r"../dif_data_cmp_weights/VGG16_complt30dataV6.h5"
test_dir = os.path.join(dst_path, '')

batch_size = 32

model = load_model(model_file)

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode="categorical")
model.compile(
    loss="categorical_crossentropy", metrics=['accuracy'], optimizer='sgd')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.samples / batch_size)

# model = multi_gpu_model(model, 2)  #GPU个数为2
print('test acc: %.3f%%' % test_acc)
print('test loss: %.3f' % test_loss)