import numpy as np
# load the pictures
import matplotlib.pyplot as plt
import tensorflow.keras as K


# array_of_pictures=np.load("small_res_faces_10000.npy")
# array_of_pictures=array_of_pictures.astype("float32")/255

# fig,axes=plt.subplots(10,10,figsize=(10,10))
# fig.suptitle("real faces")
# for i in range(100) :
#     axes[i//10,i%10].imshow(array_of_pictures[i,:,:])
#     axes[i//10,i%10].axis("off")
#plt.show()

# mean_face=array_of_pictures.mean(axis=0)
# centered_array = array_of_pictures - mean_face
# multiplier=np.max(np.abs(array_of_pictures),axis=0)
# normalized_array = centered_array/multiplier

# plt.imshow(normalized_array[1])
# plt.show()

def recover(normalized) :
    return normalized*multiplier+mean_face


def convolutional_GenDiscr() :
    #generator model and discriminator model

    def get_generator() :
        inputs     = K.Input(shape=( 16 , ))
        hidden     = K.layers.Dense( 2048 , activation = 'tanh'    )(inputs)
        hidden = K.layers.Reshape((2, 2, 512))(hidden)
        hidden = K.layers.UpSampling2D((2, 2))(hidden)
        hidden = K.layers.Conv2D(filters= 64 ,kernel_size=(3,3), padding="same",   activation = 'tanh' )(hidden)
        hidden = K.layers.BatchNormalization()(hidden)
        hidden = K.layers.Activation("tanh")(hidden)
        hidden = K.layers.UpSampling2D((2, 2))(hidden)
        hidden = K.layers.Conv2D(filters= 16 ,kernel_size=(3,3), padding="same",   activation = 'tanh' )(hidden)
        hidden = K.layers.BatchNormalization()(hidden)
        hidden = K.layers.Activation("tanh")(hidden)
        hidden = K.layers.UpSampling2D((2, 2))(hidden)
        hidden = K.layers.Conv2D(filters= 1 ,kernel_size=(3,3), padding="same",   activation = 'tanh' )(hidden)
        hidden = K.layers.BatchNormalization()(hidden)
        outputs = K.layers.Activation("tanh")(hidden)
    
        generator  = K.Model(inputs, outputs, name="generator")
        return generator
    def get_discriminator():
        inputs        = K.Input(shape=(16, 16, 1))
        # premiere couche de convolution
        hidden        = K.layers.Conv2D(filters= 32 ,kernel_size=(3,3), padding="same",   activation = 'tanh' )(inputs)
        hidden        = K.layers.MaxPooling2D((2, 2))(hidden)
        hidden = K.layers.Activation("tanh")(hidden)
        
        # deuxieme couche de convolution
        hidden        = K.layers.Conv2D(filters= 64 ,kernel_size=(3,3), padding="same",   activation = 'tanh' )(hidden)  
        hidden        = K.layers.MaxPooling2D((2, 2))(hidden)
        hidden = K.layers.Activation("tanh")(hidden)

        # troisieme couche de convolution
        hidden        = K.layers.Conv2D(filters= 128 ,kernel_size=(3,3), padding="same",   activation = 'tanh' )(hidden)  
        hidden        = K.layers.MaxPooling2D((2, 2))(hidden)
        hidden = K.layers.Activation("tanh")(hidden)

        # quatrieme couche de convolution
        hidden        = K.layers.Conv2D(filters= 256 ,kernel_size=(3,3), padding="same",   activation = 'tanh' )(hidden)  
        hidden        = K.layers.MaxPooling2D((2, 2))(hidden)
        hidden = K.layers.Activation("tanh")(hidden)

        # flatten
        hidden = K.layers.Flatten()(hidden)
        outputs       = K.layers.Dense( 1 , activation = 'tanh' )(hidden)
        discriminator = K.Model(inputs, outputs, name="discriminator")
        return discriminator
            
    return get_generator() , get_discriminator()

gen, discr = convolutional_GenDiscr( ) 
print( gen.summary(line_length = 100)   )
print( discr.summary(line_length = 100) )