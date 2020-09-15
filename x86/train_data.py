#! /usr/bin/python3
# coding=utf-8
#####################
from __future__ import absolute_import, division, print_function
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import random
from tensorflow.keras.utils import to_categorical


print(tf.__version__)


BATCH_SIZE = 128
VAL_IMAGE_NUM = 50000
VAL_IMAGE_DIR = './ILSVRC2012_img_val/'
VAL_STEPS = int(VAL_IMAGE_NUM/BATCH_SIZE)
image_labels = []
image_paths = []
image_cnt = 0
image_shape = []

val_image_labels = []
val_image_paths = []
val_image_cnt = 0

mean = [103.939, 116.779, 123.68]

# mean_array = np.zeros((224, 224, 3))
mean_array = np.zeros((299, 299, 3))

mean_array[..., 0] = mean[2]
mean_array[..., 1] = mean[1]
mean_array[..., 2] = mean[0]

# AUTOTUNE = tf.data.experimental.AUTOTUNE
AUTOTUNE = -1

 # tf.compat.v1.enable_eager_execution()

# Open file training_image_path_label.txt
with open ('training_image_path_label.txt', 'r') as f:
    for line in f:
        image_path, image_label = line.rstrip().split(' ')
        # label_one_hot = np.zeros(1000)
        # label_one_hot[int(image_label)] = 1
        image_paths.append(image_path)
        image_labels.append(int(image_label))
        #image_labels.append(label_one_hot)
        image_cnt = image_cnt + 1
f.close()

# Print image counter        
print('Here are: ' + str(image_cnt) + ' images')

img_path_label = list(zip(image_paths, image_labels))
random.shuffle(img_path_label)
image_paths, image_labels = zip(*img_path_label)
image_paths = list(image_paths)
image_labels = list(image_labels)

# Convert to tensor
def _read_img_function(imagepath): 
    image = tf.read_file(imagepath)
    image_tensor = tf.image.decode_jpeg(image, channels=3)
    #image_tensor = image_tensor[..., ::-1] # Need to remove for self training
    #image_resize = tf.image.resize_images(image_tensor, [224, 224])
    image_resize = tf.image.resize_images(image_tensor, [299, 299])
    red, green, blue = tf.split(axis=2, num_or_size_splits=3, value=image_resize)    
    image_bgr = tf.concat(axis=2, values=[blue - mean[2], green - mean[1], red - mean[0]])
    #image_resize = image_resize - mean_array
    # image_resize = (image_resize / 127.5) - 1.0
    #image_resize = image_resize / 255.0
    # image_resize -= mean_array
    #img = image.load_img(imagepath, target_size=(224, 224))
    #x = image.img_to_array(img)
    #x = preprocess_input(x)    
    return image_bgr

def _indice_to_one_hot(indices):
    return tf.one_hot(indices, 1000)

# Dataset APIs
print("Start Training Dataset Settings")
imagepaths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
label_ds = tf.data.Dataset.from_tensor_slices(image_labels)
image_ds = imagepaths_ds.map(_read_img_function)

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

ds = image_label_ds.shuffle(buffer_size=5000)
ds = ds.repeat()

ds = ds.batch(BATCH_SIZE)

#print("After batch: ", repr(ds))
ds = ds.prefetch(buffer_size=-1)
print("Finish Training Dataset Settings")

######################## Configure the validation data ########################
with open ('validation_data_label.txt', 'r') as f:
    for line in f:
        val_image_label = line
        val_image_labels.append(int(val_image_label))
        val_image_cnt = val_image_cnt + 1
        val_image_paths.append(VAL_IMAGE_DIR + 'ILSVRC2012_val_' + str(val_image_cnt).zfill(8) + '.JPEG')

# Print validate image counter        
print('Here are: ' + str(val_image_cnt) + ' validate images')
val_imagepaths_ds = tf.data.Dataset.from_tensor_slices(val_image_paths)
val_label_ds = tf.data.Dataset.from_tensor_slices(val_image_labels)
val_image_ds = val_imagepaths_ds.map(_read_img_function)
val_image_label_ds = tf.data.Dataset.zip((val_image_ds, val_label_ds))
val_image_label_ds = val_image_label_ds.repeat()
val_image_label_ds = val_image_label_ds.batch(BATCH_SIZE)

print("======================================================")
#print(tf.compat.v1.data.get_output_shapes(dataset))

# Set the checkpoint
checkpoint_path = "./check_point/float_model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
    save_weights_only=True,
    verbose=1)


#define a model
def create_model():
    '''
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), padding="same", activation=tf.nn.relu, input_shape=(224, 224, 3)),
        keras.layers.MaxPool2D(pool_size=(2,2)),
        keras.layers.Conv2D(64, (3,3), padding="same", activation=tf.nn.relu),
        keras.layers.MaxPool2D(pool_size=(2,2)),
        keras.layers.Conv2D(128, (3,3), padding="same", activation=tf.nn.relu),
        keras.layers.MaxPool2D(pool_size=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(1000, activation=tf.nn.softmax)
    ])
    '''
    # model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top=True, input_shape= (224,224,3), classes=1000)
    # model = keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=True, input_shape= (224,224,3), classes=1000)
    model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=True, input_shape= (299,299,3), classes=1000)

    #model = keras.applications.resnet50.ResNet50(weights=None, include_top=True, input_shape= (224,224,3), classes=1000)
    #model.compile(optimizer=keras.optimizers.Adam(),
    model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


model = create_model()
model.summary()

###
latest = tf.train.latest_checkpoint(checkpoint_dir)
if (latest):
    print("Can find stored check point, load the latest one ...")
    model.load_weights(latest)



steps = int(image_cnt / BATCH_SIZE)

model.fit(ds, steps_per_epoch= steps, epochs=3, validation_data=val_image_label_ds, validation_steps=VAL_STEPS,callbacks = [cp_callback])
# model.fit(ds, steps_per_epoch= 10, epochs=1, validation_data=val_image_label_ds, validation_steps=1,callbacks = [cp_callback])

model.save("custom_network.h5")
