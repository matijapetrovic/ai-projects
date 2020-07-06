import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
import os

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, MaxPool2D
from keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.applications.vgg16 import VGG16

BATCH_SIZE = 32
IMAGE_SIZE = (70, 70)
EPOCHS = 25
COLOR_MODE = "grayscale"
CHANNELS = 1 if COLOR_MODE == "grayscale" else 3
INPUT_SHAPE = IMAGE_SIZE + (CHANNELS,)

data_dir = './chest_xray_data_set'
test_dir = './chest-xray-dataset-test/test'

classes = ['Normal', 'Virus', 'bacteria']


def read_data(data_dir_path):
    data = []
    labels = []
    for class_num, label in enumerate(classes):
        print("READING ", label)
        path = os.path.join(data_dir_path, label)
        for f in os.listdir(path):
            image = tf.keras.preprocessing.image.load_img(os.path.join(path, f),
                                                          color_mode=COLOR_MODE,
                                                          target_size=IMAGE_SIZE,
                                                          interpolation="bilinear")
            data.append(tf.keras.preprocessing.image.img_to_array(image))
            lab = [0, 0, 0]
            lab[class_num] = 1
            labels.append(lab)
    return np.array(data), np.array(labels)


def load_train_data():
    x, y = read_data(data_dir)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

    datagen_kwargs = dict(rescale=1. / 255)
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        **datagen_kwargs)

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs)

    dataflow_kwargs = dict(batch_size=BATCH_SIZE)

    train_generator = train_datagen.flow(
        x_train, y_train, shuffle=True, seed=123, **dataflow_kwargs)

    valid_generator = validation_datagen.flow(
        x_train, y_train, shuffle=False, seed=123, **dataflow_kwargs)

    return train_generator, valid_generator


def load_test_data():
    x_test, y_test = read_data(test_dir)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow(
        x_test,
        y_test,
        shuffle=False,
        batch_size=1)
    return test_generator


def load_model(path):
    return tf.keras.models.load_model(path)


def save_model(model, path):
    return tf.keras.models.save_model(model, path)


def build_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), strides=1, padding='same', activation='relu',
                     input_shape=INPUT_SHAPE, data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))

    model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))

    model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))

    model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))

    model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units=3, activation='softmax'))

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def build_transfer_model():
    base_model = VGG16(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE, pooling='avg')
    base_model.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def train_model(model, train_data, valid_data):

    lrr = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3, min_lr=0.000001)
    return model.fit(train_data, epochs=EPOCHS, validation_data=valid_data, callbacks=[lrr])


def plot_history(history):
    epochs = [i for i in range(EPOCHS)]
    fig, ax = plt.subplots(1, 2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    fig.set_size_inches(20, 10)

    ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
    ax[0].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    ax[0].set_title('Training & Validation Accuracy')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
    ax[1].plot(epochs, val_loss, 'r-o', label='Validation Loss')
    ax[1].set_title('Testing Accuracy & Loss')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Training & Validation Loss")
    plt.show()


def test_model(model, test_data):
    loss, accuracy = model.evaluate(test_data)
    return loss, accuracy


TRAIN = False
SAVE = False


def main():
    if TRAIN:
        train_data, valid_data = load_test_data()
        model = build_model()
        history = train_model(model, train_data, valid_data)
        plot_history(history)
    else:
        model = load_model('model_mani.h5')
    if SAVE:
        save_model(model, 'model_mani.h5')
    test_data = load_test_data()
    loss, accuracy = test_model(model, test_data)
    print("Loss of the model is - ", loss)
    print("Accuracy of the model is - ", accuracy * 100, "%")


if __name__ == '__main__':
    main()
