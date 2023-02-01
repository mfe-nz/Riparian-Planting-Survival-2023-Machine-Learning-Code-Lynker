"""Riparian, 2022, Lynker, PDP, MfE"""
#---------------------------------------------------------
#  Callbacks
#---------------------------------------------------------
# Author: David Knox david.knox@lynker-analytics.com
# Created: 2022 Nov
# (c) Copyright by Lynker Analytics.
#---------------------------------------------------------
# Description:
#        Used within train.py in model.fit(...)
#        tensorflow/keras callback functions
#        used in model training to
#            save model weights during training    (ModelCheckpoint)
#            monitor training            (CSVLogger, TensorBoard)
#            Lower learning rate            (ReduceLROnPlateau)
#            early stopping                (EarlyStopping)
#---------------------------------------------------------
# Modified:
#---------------------------------------------------------

import tensorflow as tf
from config import MODELFILE ,CSVFILE ,EARLYSTOP_PATIENCE ,LR_PATIENCE

MySaver=tf.keras.callbacks.ModelCheckpoint(
    MODELFILE
    , monitor='val_loss'
    , verbose=1
    , save_best_only=True
    , save_weights_only=False
    , mode='auto'
    , period=1
)
MyCSV=tf.keras.callbacks.CSVLogger(CSVFILE
    , separator=','
    , append=False
)
MyStopper=tf.keras.callbacks.EarlyStopping(monitor='val_loss'
    , min_delta=0
    , patience=EARLYSTOP_PATIENCE
    , verbose=1
    , mode='auto'
    , baseline=None
    , restore_best_weights=False
)
MyLR=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss'
    , factor=0.1
    , patience=LR_PATIENCE
    , verbose=1
    , mode='auto'
    , min_delta=0.0001
    , cooldown=0
    , min_lr=0
)
MyTB=tf.keras.callbacks.TensorBoard(log_dir="logs")

callbacks=[MySaver ,MyCSV ,MyStopper ,MyLR ,MyTB]
