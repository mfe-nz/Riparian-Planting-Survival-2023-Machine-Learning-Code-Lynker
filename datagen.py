"""Riparian, 2022, Lynker, PDP, MfE"""
#---------------------------------------------------------
#  Data generators
#---------------------------------------------------------
# Author: David Knox david.knox@lynker-analytics.com
# Created: 2022 Nov
# (c) Copyright by Lynker Analytics.
#---------------------------------------------------------
# Description:
#               Used within train.py in model.fit(...) and showvalid.py
#               tensorflow/keras data loading, augmentation and generator functions
#        Loads image and segmentation mask pairs from:
#            train_image_dir + train_label_dir    (training data)
#            valid_image_dir + valid_label_dir    (validation/monitoring data)
#        see config.py for directories
#---------------------------------------------------------
# Modified:
#---------------------------------------------------------

import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from config import HEIGHT,WIDTH,BATCH_SIZE,SEED,MODELTYPE,CHANNELS
from config import TRAIN_IMAGE_DIR,TRAIN_LABEL_DIR,VALID_IMAGE_DIR,VALID_LABEL_DIR

print ( 'Data Model Type', MODELTYPE )

def pixel_rescale(img):
    """rescale pixel values to -1 to 1"""
    return (img / 127.5) -1.

label_data_gen_parms = dict(
           horizontal_flip=True
           ,vertical_flip=True
    #,rescale=1./255.
    ,dtype="uint8"
)

if MODELTYPE=='ENet':
    data_gen_parms = dict(
            horizontal_flip=True
            ,vertical_flip=True
        ,rescale=1./255.
        ,dtype="float32"
    )
elif MODELTYPE=='DeepLab':
    data_gen_parms = dict(
            horizontal_flip=True
            ,vertical_flip=True
        ,preprocessing_function=pixel_rescale
        ,dtype="float32"
    )
else:
    print ( 'Unknown modeltype', MODELTYPE )
    sys.exit()

image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_parms)
label_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**label_data_gen_parms)

if CHANNELS == 4:
    image_parms = dict(
        class_mode=None
        ,seed=SEED
        ,shuffle=True
        ,batch_size=BATCH_SIZE
        ,target_size=(HEIGHT,WIDTH)
        ,color_mode="rgba"
    )
else:
    image_parms = dict(
        class_mode=None
        ,seed=SEED
        ,shuffle=True
        ,batch_size=BATCH_SIZE
        ,target_size=(HEIGHT,WIDTH)
    )
label_parms = dict(
        class_mode=None
        ,seed=SEED
        ,shuffle=True
        ,batch_size=BATCH_SIZE
        ,target_size=(HEIGHT,WIDTH)
    ,color_mode='grayscale'
)

train_image_generator = image_datagen.flow_from_directory(TRAIN_IMAGE_DIR ,**image_parms)
train_label_generator = label_datagen.flow_from_directory(TRAIN_LABEL_DIR ,**label_parms)

valid_image_generator = image_datagen.flow_from_directory(VALID_IMAGE_DIR ,**image_parms)
valid_label_generator = label_datagen.flow_from_directory(VALID_LABEL_DIR ,**label_parms)

train_generator = zip(train_image_generator,train_label_generator)
valid_generator = zip(valid_image_generator,valid_label_generator)


if __name__ == "__main__":
    SAMPLES=0
    for X,y in train_generator:
        SAMPLES+=len(X)
        print ( SAMPLES, X.shape, y.shape, np.min(X), np.mean(X), np.max(X), end='	')
        print ( np.min(y), np.mean(y), np.max(y) )
        im=(X[0].reshape((HEIGHT,WIDTH,CHANNELS))*127.5+1).astype(np.uint8)
        lbl=(y[0].reshape((HEIGHT,WIDTH))*30).astype(np.uint8)
        for c in np.unique(y[0]):
            print ( c, np.sum(y[0]==c) )
        Image.fromarray(im[:,:,:3]).show()
        input("press enter to continue")
        if SAMPLES > 5000:
            sys.exit()
