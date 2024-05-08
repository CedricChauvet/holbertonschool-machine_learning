#!/usr/bin/env python3
"""
this projet is about keras, learning with exercices
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    ok, this save a model
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    this load a model
    """
    return K.models.load_model(filename)


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    this is the task 8, select the best model"
    """

    if save_best:

        callback_check = K.callbacks.ModelCheckpoint(
                         filepath=filepath,
                         monitor='val_loss',
                         verbose=0,
                         save_best_only=True,
                         save_weights_only=False,
                         mode='auto',
                         save_freq='epoch',
                         initial_value_threshold=None)

        history = network.fit(data, labels,
                              epochs=epochs, batch_size=batch_size,
                              verbose=verbose, shuffle=shuffle,
                              validation_data=validation_data,
                              callbacks=[callback_early,
                                         callback_invertime_decay,
                                         callback_check])

    if validation_data is None:
        history = network.fit(data, labels, epochs=epochs,
                              batch_size=batch_size,
                              verbose=False, shuffle=shuffle,
                              validation_data=validation_data)

    else:

        # Define your learning rate schedule function with inverse time decay
        def lr_schedule(epoch):
            """
            Inverse Time Decay Learning Rate Schedule
            """
            learning_rate = alpha / (1 + decay_rate * epoch)

            return learning_rate

        # callback early Stopping when back propgation is null
        callback_early = K.callbacks.EarlyStopping(patience=patience)
        # callback decay inverse time
        callback_invertime_decay = K.callbacks.\
            LearningRateScheduler(lr_schedule, verbose=False)
        callback_check = None

        history = network.fit(data, labels,
                              epochs=epochs, batch_size=batch_size,
                              verbose=verbose, shuffle=shuffle,
                              validation_data=validation_data,
                              callbacks=[callback_early,
                                         callback_invertime_decay,
                                         callback_check])

    return history
