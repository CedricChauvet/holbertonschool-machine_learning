
import keras
from matplotlib import pyplot as plt

# last instruction to fill
# history = model1.fit(train_x, train_y,validation_split = 0.1, epochs=50, batch_size=4)
# print(dir(history))

def plot_history(history):
    # model should be fitted in history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()