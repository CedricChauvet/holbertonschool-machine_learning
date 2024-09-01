#!/usr/bin/env python3
"""
Project GAN
By Ced+
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model) :    
    def __init__( self, generator , discriminator , latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005,lambda_gp=10):
        super().__init__()                         # run the __init__ of keras.Model first. 
        self.latent_generator = latent_generator
        self.real_examples    = real_examples
        self.generator        = generator
        self.discriminator    = discriminator
        self.batch_size       = batch_size
        self.disc_iter        = disc_iter
        
        self.learning_rate    = learning_rate
        self.beta_1=.3                              # standard value, but can be changed if necessary
        self.beta_2=.9                              # standard value, but can be changed if necessary
        
        self.lambda_gp        = lambda_gp                                # <---- New !
        self.dims = self.real_examples.shape                             # <---- New !
        self.len_dims=tf.size(self.dims)                                 # <---- New !
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')   # <---- New !
        self.scal_shape=self.dims.as_list()                              # <---- New !
        self.scal_shape[0]=self.batch_size                               # <---- New !
        for i in range(1,self.len_dims):                                 # <---- New !
            self.scal_shape[i]=1                                         # <---- New !
        self.scal_shape=tf.convert_to_tensor(self.scal_shape)            # <---- New !
        
        # define the generator loss and optimizer:
        self.generator.loss = lambda x: \
            - tf.math.reduce_mean(input_tensor=x)

        
        
        self.generator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer , loss=generator.loss )
        
        # define the discriminator loss and optimizer:
        self.discriminator.loss =\
            lambda x, y: - tf.math.reduce_mean(input_tensor=x)\
            + tf.math.reduce_mean(input_tensor=y)
       
       
        self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer , loss=discriminator.loss )

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        if not size :
            size= self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        if not size :
            size= self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices  = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)
    
    # generator of interpolating samples of size batch_size              # <---- New !
    def get_interpolated_sample(self,real_sample,fake_sample):
        u = tf.random.uniform(self.scal_shape)
        v=tf.ones(self.scal_shape)-u
        return u*real_sample+v*fake_sample
    
    # computing the gradient penalty                                     # <---- New !
    def gradient_penalty(self,interpolated_sample):
        with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated_sample)
                pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)      
     
    def train_step(self, useless_argument):
        """
        use to  train gen and discriminator
        """
        # 1. Entraînement du discriminateur
        for _ in range(self.disc_iter):

            # compute the loss for the discriminator
            # in a tape watching the discriminator's weights
            with tf.GradientTape() as tape:

                # get a real sample
                real = self.get_real_sample(size=None)

                # get a fake sample
                fake = self.get_fake_sample(size=None)

                # compute the loss discr_loss
                # of the discriminator on real and fake samples
                pred_real = self.discriminator(real, training=True)
                pred_fake = self.discriminator(fake, training=True)
                discr_loss = self.discriminator.loss(pred_real, pred_fake)

            # apply gradient descent once to the discriminator
            discr_grads = tape.gradient(discr_loss,
                                        self.discriminator.trainable_variables)
            self.discriminator.optimizer.\
                apply_gradients(zip(discr_grads,
                                    self.discriminator.trainable_variables))

            for w in self.discriminator.trainable_weights:
                w.assign(tf.clip_by_value(w, -1, 1))

        # 2. Entraînement du générateur
        with tf.GradientTape() as tape:
            # get a fake sample
            fake = self.get_fake_sample(size=None)

            # compute the loss gen_loss of the generator on this sample
            pred_fake = self.discriminator(fake, training=True)
            gen_loss = self.generator.loss(pred_fake)

        # apply gradient descent to the generator
        gen_grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.\
            apply_gradients(zip(gen_grads, self.generator.trainable_variables))

        # return {"discr_loss": discr_loss, "gen_loss": gen_loss}
        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
