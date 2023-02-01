"""Riparian, 2022, Lynker, PDP, MfE"""
#---------------------------------------------------------
#  model code wrapper
#---------------------------------------------------------
# Author: David Knox david.knox@lynker-analytics.com
# Created: 2022 Nov
# (c) Copyright by Lynker Analytics.
#---------------------------------------------------------
# Description:
#        imports model code based on modeltype as defined in config.py
#---------------------------------------------------------
# Modified:
#---------------------------------------------------------
import sys
from config import MODELTYPE, MODELFILE, HEIGHT, WIDTH, CHANNELS, O_CHANNELS

if MODELTYPE=='DeepLab':
    from model_deeplabv3plus_rgbi import getmodel
else:
    print ( 'Model Type Unknown', MODELTYPE )
    sys.exit()

if __name__ == "__main__":
    model,age=getmodel(MODELFILE,shape=(HEIGHT,WIDTH,CHANNELS),o_channels=O_CHANNELS)
    print ( model.summary() )
