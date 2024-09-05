
#!/usr/bin/env python3
"""
Project GAN
By Ced+
"""

import numpy as np
# load the pictures
import matplotlib.pyplot as plt
import tensorflow.keras as K


def convolutional_GenDiscr() :
    #generator model and discriminator model

    def get_generator() :
        inputs     = K.Input(shape=( 16 , ))

        hidden     = K.layers.Dense( 2048, K.layers.Activation("tanh"))(inputs)
        hidden = K.layers.Reshape((2, 2, 512))(hidden)
        hidden = K.layers.UpSampling2D((2, 2))(hidden)


        hidden = K.layers.Conv2D(filters= 64 ,kernel_size=3, padding="same" )(hidden)
        hidden = K.layers.BatchNormalization()(hidden)
        hidden = K.layers.Activation("tanh")(hidden)
        hidden = K.layers.UpSampling2D((2, 2))(hidden)

        hidden = K.layers.Conv2D(filters= 16 ,kernel_size=3, padding="same" )(hidden)
        hidden = K.layers.BatchNormalization()(hidden)
        hidden = K.layers.Activation("tanh")(hidden)
        hidden = K.layers.UpSampling2D((2, 2))(hidden)
        
        hidden = K.layers.Conv2D(filters= 1 ,kernel_size=3, padding="same")(hidden)
        hidden = K.layers.BatchNormalization()(hidden)
        outputs = K.layers.Activation("tanh")(hidden)
    
        generator  = K.Model(inputs, outputs, name="generator")
        return generator
    def get_discriminator():
        inputs        = K.Input(shape=(16, 16, 1))
        # premiere couche de convolution
        hidden        = K.layers.Conv2D(filters= 32 ,kernel_size=3, padding="same" )(inputs)
        hidden        = K.layers.MaxPooling2D((2, 2))(hidden)
        hidden = K.layers.Activation("tanh")(hidden)
        
        # deuxieme couche de convolution
        hidden        = K.layers.Conv2D(filters= 64 ,kernel_size=3, padding="same" )(hidden)  
        hidden        = K.layers.MaxPooling2D((2, 2))(hidden)
        hidden = K.layers.Activation("tanh")(hidden)

        # troisieme couche de convolution
        hidden        = K.layers.Conv2D(filters= 128 ,kernel_size=3, padding="same" )(hidden)  
        hidden        = K.layers.MaxPooling2D((2, 2))(hidden)
        hidden = K.layers.Activation("tanh")(hidden)

        # quatrieme couche de convolution
        hidden        = K.layers.Conv2D(filters= 256 ,kernel_size=3, padding="same" )(hidden)  
        hidden        = K.layers.MaxPooling2D((2, 2))(hidden)
        hidden = K.layers.Activation("tanh")(hidden)

        # flatten
        hidden = K.layers.Flatten()(hidden)
        outputs       = K.layers.Dense( 1,K.layers.Activation("tanh") )(hidden)
        discriminator = K.Model(inputs, outputs, name="discriminator")
        return discriminator
    

    
    return get_generator() , get_discriminator()

gen, discr = convolutional_GenDiscr( ) 
print( gen.summary(line_length = 100)   )
print( discr.summary(line_length = 100) )