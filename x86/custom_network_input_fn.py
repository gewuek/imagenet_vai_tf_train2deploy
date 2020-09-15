#! /usr/bin/python3
# coding=utf-8
#####################

import tensorflow as tf
import numpy as np
import cv2

IMG_SIZE = 299
dim = (IMG_SIZE, IMG_SIZE)
mean = [103.939, 116.779, 123.68]
mean_array = np.zeros((IMG_SIZE, IMG_SIZE, 3))
mean_array[..., 0] = mean[0]
mean_array[..., 1] = mean[1]
mean_array[..., 2] = mean[2]

tf.enable_eager_execution()

calib_image_dir = "./ILSVRC2012_img_val/"
calib_image_list = "./calibration.txt"
calib_batch_size = 50
def calib_input(iter):
    images = []
    line = open(calib_image_list).readlines()
    for index in range(0, calib_batch_size):
        curline = line[iter * calib_batch_size + index]
        [calib_image_name, label_id] = curline.split(' ')
        image = cv2.imread(calib_image_dir + calib_image_name)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #cv2.imshow("test image", image)
        #cv2.waitKey(300)
        #image = image / 255.0
        image = image - mean_array
        images.append(image)
    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    return {"input_1": images}
