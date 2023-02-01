"""Riparian, 2022, Lynker, PDP, MfE"""
#---------------------------------------------------------
#  CNN model inference
#---------------------------------------------------------
# Author: David Knox david.knox@lynker-analytics.com
# Created: 2022 Nov
# (c) Copyright by Lynker Analytics.
#---------------------------------------------------------
# Description:
#        Runs the CNN model against input imagery
#        and produces a classified raster
#        also produces a multi-channel class probability raster
#---------------------------------------------------------
# Modified:
#---------------------------------------------------------

import argparse
from os import listdir
from os.path import isfile
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.plot import reshape_as_image
from scipy import signal
from model import getmodel
from config import INFERENCE_DIM,INFERENCE_BATCH_SIZE,FEATHERED_INFERENCE_BORDER
from config import CHANNELS,O_CHANNELS
from config import MODELFILE, INFERENCE_INDIR, INFERENCE_OUTDIR, MODELTYPE

#--------------------------------------------
# handle command line arguments
#--------------------------------------------
parser=argparse.ArgumentParser()
parser.add_argument(
    '--indir'
    , help='input folder for RGBI rasters'
    , default=INFERENCE_INDIR
    , type=str
    , required=False
)
parser.add_argument(
    '--outdir'
    , help='output folder for classified rasters'
    , default=INFERENCE_OUTDIR
    , type=str
    , required=False
)
parser.add_argument(
    '--modelfile'
    , help='binary model weights file location'
    , default=MODELFILE
    , type=str
    , required=False
)
args=parser.parse_args()

#--------------------------------------------
# globals
#--------------------------------------------
INDIR=args.indir
OUTDIR=args.outdir
MODELFILE=args.modelfile
HEIGHT=INFERENCE_DIM
WIDTH=INFERENCE_DIM
BATCH_SIZE=INFERENCE_BATCH_SIZE
BORDER=FEATHERED_INFERENCE_BORDER
STRIDE=INFERENCE_DIM-2*BORDER

infiles=[]
for f in listdir(INDIR):
    if f.split('.')[-1] in ['tif','jp2']:
        infiles.append(INDIR+'/'+f)

print ( infiles )

print ( 'INFERENCE_DIM', INFERENCE_DIM )
print ( 'BORDER', BORDER )
print ( 'Stride', STRIDE )

model,age=getmodel(MODELFILE,shape=(HEIGHT,WIDTH,CHANNELS),o_channels=O_CHANNELS)
for layer in model.layers:
    layer.trainable=False
model.compile(optimizer='sgd' ,loss='mse')

print ( model.summary() )

def main():
    """main funcion for inference on input raster using CNN model"""

    #we want the weight at the edge of the inner chip to be about 50% hence the divide by.
    lin=np.linspace(0,INFERENCE_DIM,num=INFERENCE_DIM)
    lweight=signal.gaussian(
        INFERENCE_DIM
        , std=(INFERENCE_DIM-2*BORDER)/1.68)*np.power(np.sin((lin*np.pi/INFERENCE_DIM)),1/4
    )
    lweight-=np.min(lweight)
    lweight/=np.max(lweight)

    mask=np.outer(lweight,lweight)

    print ( mask.shape )

    for inraster in infiles:
        outraster=inraster.replace(INDIR,OUTDIR).replace('.jp2','.tif')
        class_outraster=outraster.replace('.tif','_class.tif')
        if isfile(outraster):
            print ( outraster, 'already exists' )
            continue
        print ( 'Processing', outraster )

        chips=[]
        coords=[]

        def infer(dst):
            if MODELTYPE == 'DeepLab':
                Xin=np.array(chips)/127.5 -1
            else:
                Xin=np.array(chips)/255.
            print ( Xin.shape )
            p1=model.predict(Xin)*255

            for i in range(len(coords)):
                (c,r)=coords[i]
                try:
                    for o in range(O_CHANNELS):
                        print ( '    ', c, r )
                        out=p1[i,:,:,o].reshape((HEIGHT,WIDTH))*mask
                        inout=0
                        try:
                            print ( c,r,WIDTH,HEIGHT )
                            inout=dst.read(o+1,window=Window(c,r,WIDTH,HEIGHT))
                        except Exception as e:
                            print ( 'inout error', e )
                        dst.write(out+inout,window=Window(c,r,WIDTH,HEIGHT),indexes=o+1)
                except Exception as e:
                    print ( 'Exception', e )

        with rasterio.open(inraster,'r') as src:
            profile=src.profile
            meta=src.meta
            profile.update({
                'count':O_CHANNELS,
                'bigtiff':True,
                'driver': 'GTiff',
                'compress':'LZW',
                'num_threads':'ALL_CPUS',
                'nodata': 255,
                'dtype': 'float32'
            })
            del profile['nodata']
            print ( profile )
            with rasterio.open(outraster,'w+',**profile) as dst:
                #----------------------------------------------------------------
                # initialise raster with 0s
                #----------------------------------------------------------------
                dim=10
                print ( 'initialising', outraster )
                for col_offset in range(0,meta['width']-dim,dim):
                    for row_offset in range(0,meta['height']-dim,dim):
                        for o in range(O_CHANNELS):
                            dst.write(
                                np.zeros((dim,dim),dtype=np.float32)
                                ,window=Window(col_offset,row_offset,dim,dim)
                                ,indexes=o+1
                            )
                print ( 'initialised', outraster )

                b=0
                for col_offset in range(0,meta['width'],STRIDE):
                    print ( '', flush=True )
                    for row_offset in range(0,meta['height'],STRIDE):
                        print ( col_offset
                                , '/', meta['width']
                                , row_offset, '/', meta['height']
                                , flush=True
                              )
                        im=np.zeros((HEIGHT,WIDTH,CHANNELS),dtype=np.float32)
                        size_ok=True
                        for c in range(CHANNELS):
                            patch=src.read(c+1,window=Window(col_offset,row_offset,WIDTH,HEIGHT))
                            if patch.shape[:2] == (HEIGHT,WIDTH):
                                im[:,:,c]=patch
                                #print ( 'happy shape', patch.shape )
                            else:
                                size_ok=False
                                #print ( 'irregular shape', patch.shape )
                        if size_ok:
                            if np.max(im) > 0:
                                coords.append((col_offset,row_offset))
                                chips.append(adjustim(im))
                                b+=1
                                if b >= BATCH_SIZE:
                                    infer(dst)
                                    b=0
                                    chips=[]
                                    coords=[]
                            else:
                                print ( 'empty chip', np.min(im), np.max(im) )

                if len(chips) > 0:
                    infer(dst)

        #-------------------------------------------------------------
        # create classified raster
        #-------------------------------------------------------------
        print ( 'creating classified raster', outraster )
        with rasterio.open(outraster,'r+') as src:
            profile=src.profile
            meta=src.meta
            profile.update({
                'count':1,
                'bigtiff':True,
                'driver': 'GTiff',
                'compress':'LZW',
                'num_threads':'ALL_CPUS',
                'nodata': 0,
                'dtype': 'uint8'
            })
            del profile['nodata']
            print ( profile )
            with rasterio.open(class_outraster,'w+',**profile) as dst:
                dim=10
                for col_offset in range(0,meta['width'],dim):
                    for row_offset in range(0,meta['height'],dim):
                        try:
                            patch=reshape_as_image(
                                      src.read(window=Window(col_offset,row_offset,dim,dim))
                            )
                            """
                            patch[:,:,1]        #Big Trees
                            patch[:,:,2]        #Woody
                            patch[:,:,3]        #Veg
                            patch[:,:,4]        #Lush Grass
                            patch[:,:,5]        #Dry Grass
                            patch[:,:,6]        #Unveg
                            """
                            #patch[:,:,1]*=0.65                #Big Trees
                            #patch[:,:,1]+=0.2*p1[:,:,:,2]
                            #patch[:,:,1]+=0.2*p1[:,:,:,3]
                            #patch[:,:,1]*=1.2
                            #patch[:,:,2]+=0.2*p1[:,:,:,3]
                            #patch[:,:,3]*=0.9

                            patch[:,:,6]*=1.1

                            p1=np.argmax(patch,axis=-1).astype(np.uint8)+100
                            dst.write(
                                      p1.reshape(dim,dim)
                                      ,window=Window(col_offset,row_offset,dim,dim)
                                      ,indexes=1
                            )
                            weights=np.sum(patch,axis=-1)+1e-6
                            for o in range(O_CHANNELS):
                                p2=patch[:,:,o]/weights
                                src.write(
                                    p2.reshape(dim,dim)
                                    ,window=Window(col_offset,row_offset,dim,dim)
                                    ,indexes=o+1
                                )
                            print ( 'ok in classifying' )
                        except Exception as e:
                            print ( 'error in classifying', class_outraster, e )
        print ( 'done', outraster )

def adjustim(rgb):
    """adjustim, normalize pixel values"""
    rgb[rgb>5000]=0
    img=rgb*1.0
    img-=np.min(img)
    img/=np.max(img)
    img*=255
    return img

if __name__ == "__main__":
    main()
    print ( 'done.' )
