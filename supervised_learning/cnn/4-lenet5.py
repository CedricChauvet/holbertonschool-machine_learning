#!/usr/bin/env python3
"""
Convolution and pooling project
by Ced
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def model_lenet5(x):
    """
    Write a function def lenet5(x): that builds a modified version of the LeNet-5 architecture using tensorflow:
    """

    # Convolutional Layer 1
    C1 = tf.layers.conv2d(
        inputs=x,
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )

    # Subsampling Layer 1 (Max Pooling)
    S2 = tf.layers.max_pooling2d(
        inputs=C1,
        pool_size=(2, 2),
        strides=(2, 2)       
    )

    # Convolutional Layer 2
    C3 = tf.layers.conv2d(
        inputs=S2,
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )

    # Subsampling Layer 2 (Max Pooling)
    S4 = tf.layers.max_pooling2d(
        inputs=C3,
        pool_size=(2, 2),
        strides=(2, 2),
    )

    # Flatten the output for the fully connected layer
    S4_flat = tf.layers.flatten(S4)

    # Fully Connected Layer 1
    C5 = tf.layers.dense(
        inputs=S4_flat,
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )
    
    # Fully Connected Layer 2
    F6 = tf.layers.dense(
        inputs=C5,
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )
    
    # Output Layer
    OUT = tf.layers.dense(
        inputs=F6,
        units=10,
       # No activation function here because we will use softmax_cross_entropy_with_logits_v2
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )

    return OUT

def lenet5(x, y_oh):
    """
    I have to return logits, train op, loss and accuracy
    """
    # Build the LeNet-5 model
    logits = model_lenet5(x)

    # Softmax activated output
    softmax_output = tf.nn.softmax(logits)

    # Define loss
    loss = tf.compat.v1.losses.softmax_cross_entropy(
           onehot_labels=y_oh, logits=logits)

    # Define optimizer
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Define accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_oh, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
   
    return softmax_output, optimizer, loss, accuracy