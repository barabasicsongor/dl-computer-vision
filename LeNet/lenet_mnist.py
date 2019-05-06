import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

class LeNet:

    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (width, height, depth)

        model = Sequential()

        model.add(Conv2D(20, (5,5), padding='same', activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        model.add(Conv2D(50, (5,5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(500, activation='relu'))

        model.add(Dense(classes, activation='softmax'))

        return model

def plot_history(history):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0,20), history.history['loss'], label='train_loss')
    plt.plot(np.arange(0,20), history.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0,20), history.history['acc'], label='train_acc')
    plt.plot(np.arange(0,20), history.history['val_acc'], label='val_acc')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()

def main():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='MNIST Data')

    # Reshape image data, to add the depth
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Scale the data to [0,1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # One-Hot Encode Labels
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    # LeNet Model
    model = LeNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

    print("[INFO] TRAINING NETWORK ...")
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=20, verbose=1)

    print("[INFO] EVALUATING NETWORK ...")
    predictions = model.predict(x_test, batch_size=128)
    print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1),
                                target_names=[str(x) for x in lb.classes_]))

    # Plot results
    plot_history(history)

if __name__ == '__main__':
    main()