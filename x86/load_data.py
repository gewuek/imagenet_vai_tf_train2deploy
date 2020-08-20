#! /usr/bin/python3
# coding=utf-8
#####################

#import argphase
import os
import glob

LABEL_FOLDER_CSV = './label_wordnetid.csv'
TRAINING_DATA_DIR = './ILSVRC2012_img_train'

# Create a dictionary dictFolderLabel like {'folder name', label}
dictFolderLabel = {}
with open(LABEL_FOLDER_CSV) as f:
    for line in f:
        label, folderName = line.rstrip().split(',')
        dictFolderLabel[folderName] = label

# print(dictFolderLabel)

# Create a file to record all the image location and the labels
paths = glob.glob(TRAINING_DATA_DIR + '/*/*')
with open('./training_image_path_label.txt',  'w') as f:
    for path in paths:
        folderName = path.rstrip().split('/')[-2:-1]
        # print(folderName[0])
        label = dictFolderLabel[folderName[0]]
        f.write('{} {}\n'.format(path, label))
