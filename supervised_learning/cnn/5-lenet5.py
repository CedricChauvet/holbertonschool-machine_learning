#!/usr/bin/env python3
"""
Convolution and pooling project
by Ced
"""
from tensorflow import keras as K


def lenet5(X):
    """
    to be documented
    """
    # print(X.shape)
    C1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation=K.layers.ReLU(),
        kernel_initializer=K.initializers.he_normal(seed=0),
        name="C1",
        )(X)

    S2 = K.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        name="S2",
        )(C1)

    C3 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=K.layers.ReLU(),
        kernel_initializer=K.initializers.he_normal(seed=0),
        name="C3",
        )(S2)

    S4 = K.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        name="S4",
        )(C3)

    S4_flat = K.layers.Flatten()(S4)

    C5 = K.layers.Dense(
        units=120,
        activation=K.layers.ReLU(),
        kernel_initializer=K.initializers.he_normal(seed=0)
        )(S4_flat)

    F6 = K.layers.Dense(
        units=84,
        activation=K.layers.ReLU(),
        kernel_initializer=K.initializers.he_normal(seed=0),
        name="F6",)(C5)

    OUT = K.layers.Dense(
        units=10,
        activation=K.layers.Softmax(),
        kernel_initializer=K.initializers.he_normal(seed=0),
        name="OUT")(F6)

    model = K.Model(inputs=X, outputs=OUT)
    optim = K.optimizers.Adam()
    model.compile(optimizer=optim,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model
