"""
making a transfer learning
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import Input
from keras.applications import InceptionResNetV2
from keras.utils.np_utils import to_categorical




prepa = __import__("0_cifar_loadData").prepare_cifar
augmentation = __import__("0_cifar_loadData").augmentation
(x_train,y_train_ohe), (x_test, y_test_ohe), (x_val, y_val_ohe) = prepa()
train_generator, val_generator = augmentation(x_train,y_train_ohe, x_val, y_val_ohe)


def resize_image(tensor):
    return tf.image.resize(tensor, (299, 299))

input_tensor = Input(shape=(32, 32, 3))  # Supposons que les images ont 3 canaux (par exemple, RGB)
resize_layer = Lambda(resize_image)(input_tensor)
# Get the InceptionV3 model so we can do transfer learning
base_inception = InceptionResNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=resize_layer,
    classifier_activation="softmax")


#InceptionV3(weights='imagenet', include_top=False, input_tensor=resize_layer,
#                             input_shape=(299, 299, 3))



# Add a global spatial average pooling layer

out = base_inception.output
out = GlobalAveragePooling2D()(out)
out = Dense(2048, activation='relu')(out)
out = Dropout(rate=0.3)(out)
out = Dense(2048, activation='relu')(out)
out = Dropout(rate=0.3)(out)

total_classes = y_train_ohe.shape[1]
predictions = Dense(total_classes, activation='softmax')(out)

model = Model(inputs=input_tensor, outputs=predictions)  # Supposons que les images ont 3 canaux (par exemple, RGB)


# only if we want to freeze layers
for layer in base_inception.layers:
    layer.trainable = False

for layer in base_inception.layers[-300:]:
    layer.trainable = True


# Compile 
model.compile(Adam(lr=.000005), loss='categorical_crossentropy', metrics=['accuracy']) 
model.summary()



# Train the model
batch_size = 512
train_steps_per_epoch = x_train.shape[0] // batch_size
val_steps_per_epoch = x_val.shape[0] // batch_size

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_steps_per_epoch,
                              validation_data=val_generator,
                              validation_steps=val_steps_per_epoch,
                              epochs=120, verbose=1)

model.save('cifar10_resnnetv2_3')
