#!/usr/bin/env python3

import pre_processing as pp
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import  Adam

def build_model():
    model = Sequential()
    model.add(Conv2D(4, kernel_size=(3, 3),activation='relu',input_shape=(150,150,3)))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(14, activation='softmax'))
    return model

source = '/home/evan/Dropbox/EECS_349_Machine_Learning/Homework/Final/Data'
dest = '/media/evan/disk/gen'
tra = 200
val = 200
prefix = 'test'
size = 150
batch = 5

train_gen, val_gen = pp.data_gen(source, dest, tra, val, prefix, size, batch)
model = build_model()
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model.fit_generator(train_gen, steps_per_epoch =50, epochs=100, validation_data = val_gen, use_multiprocessing=True)
# score = model.evaluate
# model.save_weights('test.h5')
# score = model.evaluate(train_gen, batch_size = 10)
# print('\nTrain Acc:', score[1])
# score = model.evaluate(val_gen, batch_size = 10)
# print('\nVal Acc:', score[1])
