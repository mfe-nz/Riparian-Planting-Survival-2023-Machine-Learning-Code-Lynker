"""Riparian, 2022, Lynker, PDP, MfE"""
#---------------------------------------------------------
#  Config - Global variables
#---------------------------------------------------------
# Author: David Knox david.knox@lynker-analytics.com
# Created: 2022 Nov
# (c) Copyright by Lynker Analytics.
#---------------------------------------------------------
# Description:
#		Global variables
#			image size, channels
#			CNN model training parameters such as batch_size
#			file locations for models, logfiles, training directories
#---------------------------------------------------------
# Modified:
#---------------------------------------------------------

SEED=1
#---------------------------------------
# image parameters
#---------------------------------------
HEIGHT=256
WIDTH=256
CHANNELS=4
O_CHANNELS=7

#---------------------------------------
# inference params
#---------------------------------------
INFERENCE_BORDER=64
FEATHERED_INFERENCE_BORDER=64
INFERENCE_DIM=256
INFERENCE_BATCH_SIZE=2

INFERENCE_INDIR='indata'
INFERENCE_OUTDIR='output'

#---------------------------------------
# model parameters
#---------------------------------------
BATCH_SIZE=8
SAMPLES_PER_EPOCH=320
VALIDATION_SAMPLES_PER_EPOCH=128
INITIAL_EPOCHS=100
EPOCHS=10000
EARLYSTOP_PATIENCE=399
LR_PATIENCE=71
WORKERS=1

MODELTYPE='DeepLab'

#---------------------------------------
# file locations for CNN model
#---------------------------------------
MODELFILE=f'models/{MODELTYPE}/riparian_lkr_pdp_1.h5'
CSVFILE='logs/epochstats.csv'

BULK_TRAINING_DATA='data/MLTraining/'

TRAIN_IMAGE_DIR='data/train/image/'
TRAIN_LABEL_DIR='data/train/label/'
VALID_IMAGE_DIR='data/valid/image/'
VALID_LABEL_DIR='data/valid/label/'

#---------------------------------------
# presentation
#---------------------------------------
COLOURS={
	0:[0,0,0]		#No Data
	,1:[127,127,0]		#Tall Woody
	,2:[127,255,0]		#Medium/Low Woody
	,3:[255,255,0]		#Herbaceous
	,4:[0,255,0]		#Rank Grass
	,5:[127,255,127]	#Pasture Grass
	,6:[0,0,0]		#Unvegetated
}
