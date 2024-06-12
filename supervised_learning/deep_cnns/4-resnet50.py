#!/usr/bin/env python3
"""
Deep Convolutional Architectures project
by Ced
"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    builds the ResNet-50 architecture 
    """
    
    # layer conv1
    input0 = K.Input(shape=(224, 224, 3))
    
    conv_0 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        kernel_initializer=K.initializers.he_normal(seed=0),
        padding="same")(input0)
    BN0 = K.layers.BatchNormalization()(conv_0)
    ReLU0 = K.layers.Activation(activation='relu')(BN0)
    POOL0 = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=2, padding="same")(ReLU0)
        
        
    # layer conv2
    ID2_0 = projection_block(POOL0, (64,64,256), 1)
    ID2_1 = identity_block(ID2_0, (64,64,256))
    ID2_2 = identity_block(ID2_1, (64,64,256))
   

    # layer conv3
    ID3_0 =  projection_block(ID2_2, (128,128,512),2)
    ID3_1 =  identity_block(ID3_0, (128,128,512))
    ID3_2 =  identity_block(ID3_1, (128,128,512))
    ID3_3 =  identity_block(ID3_2, (128,128,512))
    
    # layer conv4
    ID4_0 =  projection_block(ID3_3, (256, 256, 1024), 2)
    ID4_1 =  identity_block(ID4_0, (256, 256, 1024))
    ID4_2 =  identity_block(ID4_1, (256, 256, 1024))
    ID4_3 =  identity_block(ID4_2, (256, 256, 1024))
    ID4_4 =  identity_block(ID4_3, (256, 256, 1024))
    ID4_5 =  identity_block(ID4_4, (256, 256, 1024))
    
    # layer con5
    ID5_0 =  projection_block(ID4_5, (512, 512, 2048), 2)
    ID5_1 =  identity_block(ID5_0, (512, 512, 2048))
    ID5_2 =  identity_block(ID5_1, (512, 512, 2048))
    
    # average pooling2d and dense 1000
    avg_pooling = K.layers.AveragePooling2D(pool_size=(1, 1), padding="same")(ID5_2)
    dense = K.layers.Dense(units=1000)(avg_pooling)
    model = K.Model(input0, dense)
    return model