#!/usr/bin/env python3

from keras.preprocessing.image import ImageDataGenerator
import os, shutil, re

'''
designed especially for augment training data
'''
'''
[file structure]
root
----utility
    ----dataAugment.py
----data
    ----raw
        ----train_10
        ----tarin_20
        ----train_30
        ----valid_10

Instructions: before running this script, you need a folder structure like above.
'''
def augment_data(raw_data_quantity_per_piece, scale, target_size, source_dir, util='train'):
    '''
    raw_data_quantity_per_piece: the total number of raw data for each piece
    scale: the number of augmented images will be (# of raw data * scale)
    target size: augmented image size
    source_dir: directory containing raw data to be augmented
    util: choose to augment training data or validation data. Default is training data.
    '''
    pieceTypeList = ['b_jiang','b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu',
                    'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']
    # target_dir: target directory to save augmented images
    if util=='train':
        target_dir= '{}/../../augmented/train_{}_{}'.format(source_dir, raw_data_quantity_per_piece, raw_data_quantity_per_piece*scale)
    elif util=='valid':
        target_dir= '{}/../../augmented/valid_{}_{}'.format(source_dir, raw_data_quantity_per_piece, raw_data_quantity_per_piece*scale)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        shutil.rmtree(target_dir)
        os.makedirs(target_dir)

    augment_datagen = ImageDataGenerator(
        rescale= 1./255,
        rotation_range= 360,
        width_shift_range= 0.1,
        height_shift_range= 0.1,
        shear_range= 0.2,
        zoom_range= (0.8,1.1),
        fill_mode= 'nearest',
        data_format= 'channels_last'
        )

    augment_generator = augment_datagen.flow_from_directory(
        source_dir,
        target_size,  # all images will be resized to 56 * 56
        batch_size= raw_data_quantity_per_piece * 14,
        save_to_dir= target_dir,
        shuffle= False,
        class_mode= 'categorical')

    print('######')
    # print(len(augment_generator))

    print('generating augmented data...')
    for i in range(scale):
        augment_generator.next()
    print('---- generation done!')

    ## separate images to sub-folder
    print('separating images to sub-folder...')
    for i in range(len(pieceTypeList)):
        if not os.path.exists(target_dir+'/'+pieceTypeList[i]):
            os.makedirs(target_dir+'/'+pieceTypeList[i])
    file_all = os.listdir(target_dir)
    for file in file_all:
        if file.split('_')[0] == '':
            file_index = int(file.split('_')[1])
            pieceType = pieceTypeList[file_index // raw_data_quantity_per_piece]
            src_file = target_dir+'/'+file
            target_file = target_dir+'/'+pieceType+'/'+file
            shutil.move(src_file, target_file)
    print('---- separation done!')

    # next(train_generator)
    print(augment_generator.class_indices)
    print(target_dir)

if __name__ == '__main__':
    augment_data(
        raw_data_quantity_per_piece= 10,
        source_dir= '../data/raw/valid_10',
        scale= 30,
        target_size= (56,56),
        util='valid')
