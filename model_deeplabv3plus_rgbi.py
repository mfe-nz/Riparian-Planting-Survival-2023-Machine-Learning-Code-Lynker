"""Riparian, 2022, Lynker, PDP, MfE"""
#---------------------------------------------------------
#  DeepLabV3+ model code
#---------------------------------------------------------
# retrieved from:
# https://github.com/keras-team/keras-io/blob/master/examples/vision/deeplabv3_plus.py
#    on 27 Sept 2021
#    see original header below this header
#---------------------------------------------------------
# Description:
#        DeepLabV3+ CNN model code
#        Implemented
#---------------------------------------------------------
# Modified:
#    Modifications (c) Copyright by Lynker Analytics.
#     2021 Dec    David Knox david.knox@lynker-analytics.com
#                - Extracted model code from longer script
#                - edited variable names and use of config file
#                - added getmodel() function for consistency with other models in use by Lynker
#                    handle loading existing model weights
#                - model output layer activation to 'softmax' for multiclass
#    2022 Jun    David Knox david.knox@lynker-analytics.com
#                - Added pre layers to handle RGBI input
#                - change variable and function names to keep pylint happy
#---------------------------------------------------------
from os.path import isfile
import tensorflow as tf
#---------------------------------------------------------
# original header
#---------------------------------------------------------
#Title: Multiclass semantic segmentation using DeepLabV3+
#Author: [Soumik Rakshit](http://github.com/soumik12345)
#Date created: 2021/08/31
#Last modified: 2021/09/1
#Description: Implement DeepLabV3+ architecture for Multi-class Semantic Segmentation.

## Introduction

#Semantic segmentation, with the goal to assign semantic labels to every pixel in an image,
#is an essential computer vision task. In this example, we implement
#the **DeepLabV3+** model for multi-class semantic segmentation, a fully-convolutional
#architecture that performs well on semantic segmentation benchmarks.

### References:

#- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
#- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
#- [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs]
#-   (https://arxiv.org/abs/1606.00915)

def getmodel(modelfile,shape=(512,512,3),o_channels=1):
    """function to create and return CNN model"""
    width,height,channels=shape

    ### Building the DeepLabV3+ model

    #DeepLabv3+ extends DeepLabv3 by adding an encoder-decoder structure. The encoder module
    #processes multiscale contextual information by applying dilated convolution at multiple
    #scales, while the decoder module refines the segmentation results along object boundaries.

    #![](https://github.com/lattice-ai/DeepLabV3-Plus/raw/master/assets/deeplabv3_plus_diagram.png)

    #**Dilated convolution:** With dilated convolution, as we go deeper in the network, we can keep the
    #stride constant but with larger field-of-view without increasing the number of parameters
    #or the amount of computation. Besides, it enables larger output feature maps, which is
    #useful for semantic segmentation.

    #The reason for using **Dilated Spatial Pyramid Pooling** is that it was shown that as the
    #sampling rate becomes larger, the number of valid filter weights (i.e., weights that
    #are applied to the valid feature region, instead of padded zeros) becomes smaller.

    def convolution_block(
        block_input,
        num_filters=256,
        kernel_size=3,
        dilation_rate=1,
        padding="same",
        use_bias=False,
    ):
        xlayer = tf.keras.layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=tf.keras.initializers.HeNormal(),
        )(block_input)
        xlayer = tf.keras.layers.BatchNormalization()(xlayer)
        return tf.nn.relu(xlayer)


    def dilated_spatial_pyramid_pooling(dspp_input):
        dims = dspp_input.shape
        xlayer = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
        xlayer = convolution_block(xlayer, kernel_size=1, use_bias=True)
        out_pool = tf.keras.layers.UpSampling2D(
            size=(dims[-3] // xlayer.shape[1], dims[-2] // xlayer.shape[2]), interpolation="bilinear",
        )(xlayer)

        out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
        out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
        out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
        out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

        xlayer = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
        output = convolution_block(xlayer, kernel_size=1)
        return output


    """
    The encoder features are first bilinearly upsampled by a factor 4, and then
    concatenated with the corresponding low-level features from the network backbone that
    have the same spatial resolution. For this example, we
    use a ResNet50 pretrained on ImageNet as the backbone model, and we use
    the low-level features from the `conv4_block6_2_relu` block of the backbone.
    """


    def deeplabv3plus(image_size, num_classes):
        model_input = tf.keras.Input(shape=(image_size, image_size, 3))
        basemodel = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_tensor=model_input
        )
        basemodel.trainable=False
        for layer in basemodel.layers:
            layer.trainable=False
            if layer.name=='conv1_bn':
                layer.trainable=True

        xlayer = basemodel.get_layer("conv4_block6_2_relu").output
        xlayer = dilated_spatial_pyramid_pooling(xlayer)

        input_a = tf.keras.layers.UpSampling2D(
            size=(image_size // 4 // xlayer.shape[1], image_size // 4 // xlayer.shape[2]),
            interpolation="bilinear",
        )(xlayer)
        input_b = basemodel.get_layer("conv2_block3_2_relu").output
        input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

        xlayer = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
        xlayer = convolution_block(xlayer)
        xlayer = convolution_block(xlayer)
        xlayer = tf.keras.layers.UpSampling2D(
            size=(image_size // xlayer.shape[1], image_size // xlayer.shape[2]),
            interpolation="bilinear",
        )(xlayer)
        if num_classes == 1:
            model_output = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1), activation='sigmoid', padding="same")(xlayer)
        else:
            model_output = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1), activation='softmax', padding="same")(xlayer)
        return tf.keras.Model(inputs=model_input, outputs=model_output)


    model = deeplabv3plus(image_size=width, num_classes=o_channels)
    age='new'

    newmodel_input = tf.keras.Input(shape=(height, width, channels))
    xlayer=tf.keras.layers.Conv2D(filters=128,kernel_size=(1,1),padding='same',activation='linear')(newmodel_input)
    xlayer=tf.keras.layers.BatchNormalization()(xlayer)
    xlayer=tf.keras.layers.Activation('relu')(xlayer)
    xlayer=tf.keras.layers.Conv2D(filters=3,kernel_size=(1,1),padding='same',activation='linear')(xlayer)
    newmodel=tf.keras.Model(inputs=newmodel_input,outputs=model(xlayer))
    model=newmodel

    if isfile(modelfile):
        print ( model.summary(), modelfile )
        oldmodel=tf.keras.models.load_model(modelfile, compile=False)
        print ( oldmodel.summary(), modelfile )
        age='old'

        for layer in oldmodel.layers:
            name=layer.name
            print ( 'setting weights for layer', name, 'from old model' )
            model.get_layer(name).set_weights(layer.get_weights())

    return model, age
