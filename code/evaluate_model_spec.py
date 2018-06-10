from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras
import numpy as np
import sys
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
from keras import applications

IMAGE_SIZE = 56
VALID_DATASET = '/home/dylan2/workspace/coding_ex/eecs349ml/final_project/round_2/data/augmented/valid_10_300'
pieceTypeList = ['b_jiang','b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu',
                'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']

def evaluate_toy_cnn_model(model_file = 'toy_cnn_mini_model_30_1800_5epo_0.97.h5'):

    model = keras.models.load_model(model_file)
    model.summary()

    rescale_ratio = 1.0/255
    shift_range = 0.2; fill = 'wrap'; zoom_l = 0.8; zoom_h = 1.2; shear = 0.2; rot = 360

    valid_gen = image.ImageDataGenerator(rescale = rescale_ratio)

    gen_data_valid = valid_gen.flow_from_directory(VALID_DATASET, shuffle=False, target_size=(IMAGE_SIZE, IMAGE_SIZE), class_mode = 'categorical')

    prediction = model.predict_generator(gen_data_valid, steps=None, max_queue_size=10, workers=1, use_multiprocessing=True, verbose=0)
    prediction = np.argmax(prediction, axis=1)

    print(gen_data_valid.classes.shape)
    print(prediction.shape)
    print('Confusion Matrix')
    print(confusion_matrix(gen_data_valid.classes, prediction))
    print('Classification Report')
    print(classification_report(gen_data_valid.classes, prediction, target_names=pieceTypeList))

def evalute_toy_cnn_bottleneck(
    wights_file = 'bottleneck_fc_model_weights_30_1800_50epo_0.92.h5'):
    model = applications.VGG16(include_top=False, weights='imagenet')

    valid_gen = image.ImageDataGenerator(rescale = rescale_ratio)
    gen_data_valid = valid_gen.flow_from_directory(VALID_DATASET, shuffle=False, target_size=(IMAGE_SIZE, IMAGE_SIZE), class_mode = 'categorical')
    prediction = model.predict_generator(gen_data_valid, steps=None, max_queue_size=10, workers=1, use_multiprocessing=True, verbose=0)

if __name__ == '__main__':
    evaluate_toy_cnn_model(
        model_file = 'toy_cnn_mini_model_30_1800_5epo_0.97.h5')
