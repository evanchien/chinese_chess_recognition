#!/usr/bin/env python3

# coding: utf-8



# 將 士 象 車 馬 砲 卒 帥 仕 相 俥 傌 炮 兵

from keras.preprocessing import image
import glob




def data_gen(source, dest, tra, val, prefix, size, batch):
    '''
        source: The source of the image wish to generate
        dest:   The destination of the output images. There will be two subfolders, train/valid, with images count ratio around 7:3
        tra:    The number of training images wish to generate
        val:    The number of validation images wish to generate
        prefix:
        size:   Image size of the output images


    '''
    rescale_ratio = 1.0/255
    shift_range = 0.2
    fill = 'wrap'
    zoom_l = 0.5
    zoom_h = 1
    shear = 0.2
    rot = 30
    split = 0.3
    datagen = image.ImageDataGenerator(rescale= rescale_ratio, width_shift_range = shift_range, height_shift_range = shift_range, fill_mode=fill, zoom_range=[zoom_l, zoom_h], shear_range=shear, rotation_range=rot, validation_split = split)

    gen_data_train = datagen.flow_from_directory(source, batch_size=batch, shuffle=True,save_to_dir=dest + '/train',save_prefix=prefix,target_size=(size, size),class_mode = 'categorical', subset = 'training')
    # for i in range(tra):
    #     x, y = gen_data_train.next()
    # print(gen_data_train.filename)
    # print(len(gen_data_train.classes))

    gen_data_valid = datagen.flow_from_directory(source, batch_size=batch,shuffle=True,save_to_dir=dest + '/valid',save_prefix=prefix,target_size=(size, size),class_mode = 'categorical', subset = 'validation')
    # for i in range(val):
    #     gen_data_valid.next()


    # print(gen_data_train.classes)
    # label_map = (gen_data_train.class_indices)
    # print(label_map)
    return gen_data_train, gen_data_valid


if __name__ == '__main__':
    source = '/home/evan/Dropbox/EECS_349_Machine_Learning/Homework/Final/Data'
    dest = '/home/evan/1/gen'
    tra = 10
    val = 10
    prefix = 'test'
    size = 30
    batch = 500
    label = ['black_jiang', 'black_ju', 'black_ma', 'black_pao', 'black_shi', 'black_xiang', 'black_zu', 'red_bing', 'red_ju', 'red_ma', 'red_pao', 'red_shi', 'red_shuai', 'red_xiang']
    data_gen(source, dest, tra, val, prefix, size, batch)
