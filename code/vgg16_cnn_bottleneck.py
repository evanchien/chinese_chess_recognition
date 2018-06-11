import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

pieceTypeList = ['b_jiang','b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu',
                'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']

# dimensions of our images.
img_width, img_height = 56, 56

top_model_weights_path = './h5_file/bottleneck_fc_model_weights.h5'
train_data_dir = '../data/augmented/train_30_1800'
validation_data_dir = '../data/augmented/valid_10_300'
train_data_quantity_per_piece = 1800
valid_data_quantity_per_piece = 300
nb_train_samples = 1800 * 14
nb_validation_samples = 300 * 14
# epochs = 50


def save_bottlebeck_features():
    '''
    here use the 'train_30_1800' training set to train the bottleneck features
    '''
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=28,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, 900)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=28,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, 150)
    np.save(open('bottleneck_features_valid.npy', 'wb'),
            bottleneck_features_validation)


def train_top_model():
    le = LabelEncoder()

    train_data = np.load(open('bottleneck_features_train.npy','rb'))
    train_labels_integers = np.array(
        [0] * train_data_quantity_per_piece +
        [1] * train_data_quantity_per_piece +
        [2] * train_data_quantity_per_piece +
        [3] * train_data_quantity_per_piece +
        [4] * train_data_quantity_per_piece +
        [5] * train_data_quantity_per_piece +
        [6] * train_data_quantity_per_piece +
        [7] * train_data_quantity_per_piece +
        [8] * train_data_quantity_per_piece +
        [9] * train_data_quantity_per_piece +
        [10] * train_data_quantity_per_piece +
        [11] * train_data_quantity_per_piece +
        [12] * train_data_quantity_per_piece +
        [13] * train_data_quantity_per_piece)
    # encode class values as integers
    encoded_train_labels = le.fit_transform(train_labels_integers)
    # convert integers to dummy variables (one hot encoding)
    train_labels = np_utils.to_categorical(encoded_train_labels)

    validation_data = np.load(open('bottleneck_features_valid.npy','rb'))
    validation_labels_integers = np.array(
        [0] * valid_data_quantity_per_piece +
        [1] * valid_data_quantity_per_piece +
        [2] * valid_data_quantity_per_piece +
        [3] * valid_data_quantity_per_piece +
        [4] * valid_data_quantity_per_piece +
        [5] * valid_data_quantity_per_piece +
        [6] * valid_data_quantity_per_piece +
        [7] * valid_data_quantity_per_piece +
        [8] * valid_data_quantity_per_piece +
        [9] * valid_data_quantity_per_piece +
        [10] * valid_data_quantity_per_piece +
        [11] * valid_data_quantity_per_piece +
        [12] * valid_data_quantity_per_piece +
        [13] * valid_data_quantity_per_piece)
    # encode class values as integers
    encoded_valid_labels = le.fit_transform(validation_labels_integers)
    # convert integers to dummy variables (one hot encoding)
    validation_labels = np_utils.to_categorical(encoded_valid_labels)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(14, activation='softmax'))

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=1000,
              batch_size=28,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

if __name__ == '__main__':
    # save_bottlebeck_features()
    train_top_model()
