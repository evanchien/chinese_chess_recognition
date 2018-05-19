#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import  Adam
from keras.datasets import mnist
from keras.utils import np_utils

def load_data():
    # 載入minst的資料
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 將圖片轉換為一個60000*784的向量，並且標準化
    x_train = x_train.reshape(x_train.shape[0], 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train/255
    x_test = x_test/255
    # 將y轉換成one-hot encoding
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    # 回傳處理完的資料
    return (x_train, y_train), (x_test, y_test)

def build_model():
    #建立模型
    model = Sequential()
    #將模型疊起
    model.add(Dense(input_dim=28*28,units=500,activation='relu'))
    model.add(Dense(units=500,activation='relu'))
    model.add(Dense(units=500,activation='relu'))
    model.add(Dense(units=10,activation='softmax'))
    model.summary()
    return model

(x_train,y_train),(x_test,y_test)=load_data()
model = build_model()
#開始訓練模型
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=100,epochs=20)
#顯示訓練結果
score = model.evaluate(x_train,y_train)
print ('\nTrain Acc:', score[1])
score = model.evaluate(x_test,y_test)
print ('\nTest Acc:', score[1])
