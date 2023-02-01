"""Riparian, 2022, Lynker, PDP, MfE"""
#---------------------------------------------------------
#  CNN model training
#---------------------------------------------------------
# Author: David Knox david.knox@lynker-analytics.com
# Created: 2022 Nov
# (c) Copyright by Lynker Analytics.
#---------------------------------------------------------
# Description:
#        Trains the CNN model
#        handles either a new model or finetunes an existing model
#        Training a new model:
#            new models are defined in the model.py getmodel() function
#            for DeepLabV3+ model architecture:
#                will use pre-trained imagenet weights in the encoder part of the network
#                               and randomly initialised weights in the decoder part
#                the encoder will be frozen initially
#                               (frozen means weights will not update during training)
#            this code will train any unfrozen layers of the model for some epochs
#            then will unfreeze all layers and continue training at a lower learning rate
#        Training an existing model:
#            this code will unfreeze all layers and train at a low learning rate
#
#        callback functions handle
#                logging, model saving, learning rate adjustments and early stopping
#        parameters affecting training are set in config.py
#---------------------------------------------------------
# Modified:
#---------------------------------------------------------

import numpy as np
import tensorflow as tf
from config import O_CHANNELS, BATCH_SIZE, SAMPLES_PER_EPOCH, VALIDATION_SAMPLES_PER_EPOCH
from config import EPOCHS, INITIAL_EPOCHS, MODELFILE, WORKERS
from config import HEIGHT, WIDTH, CHANNELS
from datagen import train_generator, valid_generator
from model import getmodel
from callbacks import callbacks

def main():
    """Main function for CNN model training."""

    model,age=getmodel(MODELFILE,shape=(HEIGHT,WIDTH,CHANNELS),o_channels=O_CHANNELS)

    print ( age, flush=True )

    adam=tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(
        optimizer=adam
        ,loss='sparse_categorical_crossentropy'
        ,metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    print ( model.summary(), flush=True )

    if age != 'old':
        print ( 'starting initial training' )

        hist=model.fit(
            train_generator
            ,steps_per_epoch=int(SAMPLES_PER_EPOCH/BATCH_SIZE)
            ,validation_data=valid_generator
            ,validation_steps=int(VALIDATION_SAMPLES_PER_EPOCH/BATCH_SIZE)
            ,use_multiprocessing=False
            ,workers=WORKERS
            ,callbacks=callbacks
            ,epochs=INITIAL_EPOCHS
            ,verbose=1
            ,max_queue_size=128
        )

        print ( 'done initial training' )

        best_epoch=np.argmax(hist.history['val_loss'])
        val_acc=hist.history['val_sparse_categorical_accuracy'][best_epoch]

        print ( 'best epoch', best_epoch, 'best val_acc', val_acc )

        adam=tf.keras.optimizers.Adam(learning_rate=0.0000001)
        model.compile(
            optimizer=adam
            ,loss='sparse_categorical_crossentropy'
            ,metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        hist=model.fit(
            train_generator
            ,steps_per_epoch=int(SAMPLES_PER_EPOCH/BATCH_SIZE)
            ,validation_data=valid_generator
            ,validation_steps=int(VALIDATION_SAMPLES_PER_EPOCH/BATCH_SIZE)
            ,use_multiprocessing=False
            ,workers=WORKERS
            ,callbacks=callbacks
            ,initial_epoch=INITIAL_EPOCHS
            ,epochs=INITIAL_EPOCHS+10
            ,verbose=1
            ,max_queue_size=128
        )

    for layer in model.layers:
        layer.trainable=True

    adam=tf.keras.optimizers.Adam(learning_rate=0.00001)

    model.compile(
        optimizer=adam
        ,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        ,metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    print ( model.summary(), flush=True )

    print ( 'starting initial training' )

    hist=model.fit(
        train_generator
        ,steps_per_epoch=int(SAMPLES_PER_EPOCH/BATCH_SIZE)
        ,validation_data=valid_generator
        ,validation_steps=int(VALIDATION_SAMPLES_PER_EPOCH/BATCH_SIZE)
        ,use_multiprocessing=False
        ,workers=WORKERS
        ,callbacks=callbacks
        ,initial_epoch=INITIAL_EPOCHS+10
        ,epochs=EPOCHS-(INITIAL_EPOCHS+10)
        ,verbose=1
        ,max_queue_size=128
    )

    print ( 'done initial training' )

    best_epoch=np.argmin(hist.history['val_loss'])
    val_acc=hist.history['val_sparse_categorical_accuracy'][best_epoch]

    print ( 'best epoch', best_epoch, 'best val_acc', val_acc )

    print ( 'done.' )

if __name__ == "__main__":
    main()
