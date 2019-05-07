import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

class MiniVGG:

    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (width, height, depth)

        model = Sequential()

        model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

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
    print("[INFO] Loading CIFAR-10 ...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Scale the data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # One-Hot Encode Labels
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform((y_test))

    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
                   'horse', 'ship', 'truck']

    # Init and optimize model
    print("[INFO] Optimizing model ...")
    # Decay reduces the learning rate over time, good setting is lr / nr epochs
    optimizer = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
    model = MiniVGG.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Training network
    print("[INFO] Training network ...")
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64,
                        epochs=40, verbose=1)

    # Evaluation
    print("[INFO] Evaluating network ...")
    predictions = model.predict(x_test, batch_size=64)
    print(classification_report(x_test.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

    plot_history(history)


if __name__ == '__main__':
    main()