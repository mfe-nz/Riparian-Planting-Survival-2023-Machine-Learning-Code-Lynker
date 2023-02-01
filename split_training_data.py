"""Riparian, 2022, Lynker, PDP, MfE"""
#---------------------------------------------------------
#  Split training data into Train and Valid sets
#---------------------------------------------------------
# Author: David Knox david.knox@lynker-analytics.com
# Created: 2022 Nov
# (c) Copyright by Lynker Analytics.
#---------------------------------------------------------
# Description:
#    given an input directory BULK_TRAINING_DATA
#    of training imagery and labels in rasterised image chips
#    in RCNN format 
#    as produced by the ArcGIS "export training data for deep learning tool"
#
#    preprocess
#    and split randomly into train and valid datasets
#---------------------------------------------------------
# Modified:
#---------------------------------------------------------

from os import listdir
from os.path import isfile
import re
import sklearn
import rasterio
from numpy.random import random as rnd
from PIL import Image
from rasterio.plot import reshape_as_image
import numpy as np
from config import SEED, BULK_TRAINING_DATA, HEIGHT, WIDTH
from config import VALID_IMAGE_DIR, VALID_LABEL_DIR, TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR

np.random.seed(SEED)

INDIR=BULK_TRAINING_DATA
FILEPAT="^.*.tif$"
shape=(HEIGHT,WIDTH)
MASKS={'labels/101':1,'labels/102':2,'labels/103':3, 'labels/104':4, 'labels/105':5, 'labels/106':6,}

PREFIX='Set1_'

VALID_FRACTION=0.3
COUNTER=0

filelist=listdir(INDIR+'/images/')
for f in sklearn.utils.shuffle(filelist):
    if rnd() < VALID_FRACTION:
        OUT_IM_DIR=VALID_IMAGE_DIR
        OUT_LB_DIR=VALID_LABEL_DIR
    else:
        OUT_IM_DIR=TRAIN_IMAGE_DIR
        OUT_LB_DIR=TRAIN_LABEL_DIR
    if re.search(FILEPAT,f):
        if not isfile(TRAIN_IMAGE_DIR+'/0/'+f) and not isfile(VALID_IMAGE_DIR+'/0/'+f):

            check=np.zeros(shape,np.float32)
            mask=np.ones(shape,np.uint8)

            iminfile=INDIR+'/images/'+f
            imoutfile=OUT_IM_DIR+'/0/'+PREFIX+f
            maskoutfile=OUT_LB_DIR+'/0/'+PREFIX+f

            print ( COUNTER, f, end=' ' )

            try:
                src=rasterio.open(iminfile)
                rgb=reshape_as_image(src.read())[:,:,:4]
                src.close()
                rgb[rgb>5000]=0
                rgb=(rgb*1.0)/np.max(rgb)
                rgb*=255
                rgb=rgb.astype(np.uint8)

                print ( np.min(rgb), np.percentile(rgb,80), np.mean(rgb), np.max(rgb) )

                if np.percentile(rgb,70) == 255:
                    print ( f, 'has blank areas' )
                else:
                    for subdir in MASKS:
                        label=MASKS[subdir]
                        sourcefile=INDIR+subdir+'/'+f
                        im=np.zeros(shape,np.float32)
                        if isfile(sourcefile):
                            with rasterio.open(sourcefile) as src:
                                for i in range(src.count):
                                    im+=src.read(i+1)

                        mask[im>0]=label
                        check+=(im>0).astype(np.float32)

                    print ( np.mean(check)  )

                    if np.abs(np.mean(check) - 1) > 0.1:
                        print ( 'doubleup annotations. exclude' )
                    else:
                        Image.fromarray(mask).save(maskoutfile.replace('.tif','.png'))
                        Image.fromarray(rgb).save(imoutfile.replace('.tif','.png'))

                    COUNTER+=1
            except Exception as e:
                print ( 'Error on ',f, ':', e )
