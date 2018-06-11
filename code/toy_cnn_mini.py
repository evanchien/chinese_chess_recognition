from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

NUM_CLASSES = 14
valid_dir = '../data/augmented/valid_10_300'

def create_toy_cnn_model():
    model = Sequential()
    ### Conv layer 1
    model.add(Convolution2D(
        input_shape=(56, 56, 3),
        filters = 32,
        kernel_size=3,
        strides=1,
        padding='same',
        data_format='channels_last',
        activation='relu'
    ))
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',
        data_format='channels_last',
    ))
    ### Conv layer 2
    model.add(Convolution2D(32, 3, strides=1, padding='same', data_format='channels_last',     activation='relu'))
    model.add(MaxPooling2D(2, 2, padding='same', data_format='channels_last'))

    ### Conv layer 3
    model.add(Convolution2D(64, 3, strides=1, padding='same', data_format='channels_last',     activation='relu'))
    model.add(MaxPooling2D(2, 2, padding='same', data_format='channels_last'))
    model.add(Dropout(0.25))

    ### FC
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

##########################################################################
##########################################################################
##########################################################################
model = create_toy_cnn_model()

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    )

# this is the augmentation configuration we will use for testing:
# only rescaling
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    )

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    '../data/augmented/train_30_1800',  # this is the target directory
    target_size=(56, 56),  # all images will be resized to 150x150
    batch_size=28,
    # save_to_dir='temp/train',
    class_mode='categorical')

# print('#!!!!!!!!!!!!!#####')
# print(len(train_generator))

# next(train_generator)
# print(train_generator.class_indices)

# pieceTypeList = ['b_jiang','b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu',
#                 'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']

# this is a similar generator, for validation data
validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(56, 56),
    batch_size=28,
    # save_to_dir='temp/valid',
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=900,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=150,
    use_multiprocessing=True)
model.save('./h5_file/toy_cnn_mini_model.h5')
