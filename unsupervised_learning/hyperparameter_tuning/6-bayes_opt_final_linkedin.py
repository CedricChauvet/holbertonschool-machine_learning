import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Conv2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from GPyOpt.methods import BayesianOptimization
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

import tensorflow as tf

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape the images to have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def create_model(learning_rate, momentum, l2_weight, Drop, dense):
    # Define the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
               kernel_regularizer=L2(l2_weight), input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(dense, activation='relu', kernel_initializer='he_uniform'),
        Dropout(rate=Drop),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    opt = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def objective_function(params):
    learning_rate = params[0][0]
    momentum = params[0][1]
    l2_weight = params[0][2]
    Drop = params[0][3]
    dense = int(params[0][4])
    
    model = create_model(learning_rate, momentum, l2_weight, Drop, dense)
    early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=3,          # Number of epochs with no improvement after which training will be stopped
    verbose=1,           # Verbosity mode
    )
    history = model.fit(x_train, y_train, epochs=20, batch_size=256, validation_split=0.2, callbacks=[early_stopping], verbose=True)
    plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)
    validation_accuracy = history.history['val_accuracy'][-1]
    return -validation_accuracy

# Define the bounds of the search space
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.03)},
    {'name': 'momentum', 'type': 'continuous', 'domain': (0.7, 0.95)},
    {'name': 'l2_weight', 'type': 'continuous', 'domain': (1e-8, 1e-5)},
    {'name': 'Drop', 'type': 'continuous', 'domain': (0.2, 0.8)},
    {'name': 'Dense', 'type': 'discrete', 'domain': (100,125,150,175,200, 225)},  # Discrete hyperparameter
    
]

# Create the Bayesian optimization object
optimizer = BayesianOptimization(f=objective_function, domain=bounds)

# Run the optimization
num_iterations = 30
optimizer.run_optimization(max_iter=num_iterations)

# Plot the convergence
optimizer.plot_convergence()

X = optimizer.X # Points évalués
Y = optimizer.Y  # Valeurs de la fonction objectif (négatif de l'accuracy dans notre cas)

best_params = optimizer.X[np.argmin(optimizer.Y)]
# Save the point of the plt into a file
with open('bayes_opt.txt', 'w') as report_file:
    report_file.write("Bayesian Optimization Report\n")
    report_file.write("===========================\n")
    report_file.write(f"Best learning rate: {best_params[0]:.4f}\n")
    report_file.write("===========================\n")
    report_file.write(f"Best momentum: {best_params[1]}\n")
    report_file.write("===========================\n")
    report_file.write(f"Best l2_weight: {best_params[2]}\n")
    report_file.write("===========================\n")
    report_file.write(f"Best Dropout: {best_params[3]}\n")
    report_file.write("===========================\n")
    report_file.write(f"Best Dense width: {best_params[4]}\n")
    report_file.write("xxxxx\n")
    report_file.write(f"Best accuracy: { -np.min(optimizer.Y):.4f}\n\n")
    report_file.write("Optimization History:\n")
    report_file.write("---------------------\n")
    for i,  acc in enumerate(optimizer.Y):
        report_file.write(f"Iteration {i+1}, : Accuracy = {-acc[0]:.5f} \n")
print("Optimization report saved to 'bayes_opt.txt'")


# Display the results
print("Optimal parameters:", best_params)
print("Best validation accuracy:", -np.min(optimizer.Y))


