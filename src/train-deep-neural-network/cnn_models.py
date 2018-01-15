from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, LeakyReLU, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D


def alexnet(img_height, img_width, img_depth, dim_choice):
    """Uses the AlexNet network topology for the cnn model

    Arguments:
        img_height: int, input image height
        img_width: int, input image width
        img_depth: int, numer of input image channels
        dim_choice: int, number of output dimensions"""
    model = Sequential()
    model.add(
        Conv2D(
            96, kernel_size=(11, 11), strides=(4, 4), padding="same",
            input_shape=(img_height, img_width, img_depth)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Conv2D(256, kernel_size=(5, 5), strides=(4, 4)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(384, kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(384, kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    if 2 == dim_choice:
        model.add(Dense(2, activation='linear'))
    else:
        model.add(Dense(1, activation='linear'))
    return model


def alexnet_no_dropout(img_height, img_width, img_depth, dim_choice):
    """Uses the AlexNet network topology without dropout layers for the cnn model

    Arguments:
        img_height: int, input image height
        img_width: int, input image width
        img_depth: int, numer of input image channels
        dim_choice: int, number of output dimensions"""
    model = Sequential()
    model.add(
        Conv2D(
            96, kernel_size=(11, 11), strides=(4, 4), padding="same",
            input_shape=(img_height, img_width, img_depth)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Conv2D(256, kernel_size=(5, 5), strides=(4, 4)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(384, kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(384, kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation("relu"))
    model.add(Dense(4096))
    model.add(Activation("relu"))
    if 2 == dim_choice:
        model.add(Dense(2, activation='linear'))
    else:
        model.add(Dense(1, activation='linear'))
    return model


def tensorkart(img_height, img_width, img_depth, dim_choice):
    """Uses NeuralKart/TensorKart model architecture

    Arguments:
        img_height: int, input image height
        img_width: int, input image width
        img_depth: int, numer of input image channels
        dim_choice: int, number of output dimensions"""
    model = Sequential()
    model.add(BatchNormalization(
        input_shape=(img_height, img_width, img_depth)))
    model.add(
        Conv2D(
            24,
            kernel_size=(5, 5),
            strides=(2, 2),
            activation='relu'))
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            36,
            kernel_size=(5, 5),
            strides=(2, 2),
            activation='relu'))
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            48,
            kernel_size=(5, 5),
            strides=(2, 2),
            activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    #model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    drop_out = 0.4
    model.add(Dropout(drop_out))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(drop_out))
    if 2 == dim_choice:
        model.add(Dense(2, activation='linear'))
    else:
        model.add(Dense(1, activation='linear'))
    return model


def simple_min1d(img_height, img_width, img_depth, dim_choice):
    """Uses own simple model as cnn topology

    Arguments:
        img_height: int, input image height
        img_width: int, input image width
        img_depth: int, numer of input image channels
        dim_choice: int, number of output dimensions"""
    model = Sequential()
    model.add(
        Conv2D(
            256,
            input_shape=(img_height, img_width, img_depth),
            kernel_size=(7, 7),
            padding='same',
            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(
            256,
            kernel_size=(5, 5),
            activation='relu',
            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(
            128,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    if 2 == dim_choice:
        model.add(Dense(2, activation='linear'))
    else:
        model.add(Dense(1, activation='linear'))
    return model


def simple_min2d(img_height, img_width, img_depth, dim_choice):
    """Uses own simple model as cnn topology

    Arguments:
        img_height: int, input image height
        img_width: int, input image width
        img_depth: int, numer of input image channels
        dim_choice: int, number of output dimensions"""
    model = Sequential()
    model.add(Conv2D(256,
        input_shape=(img_height, img_width, img_depth),
        kernel_size=(7, 7),
        padding='same',
        activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256,
                            kernel_size=(5, 5),
                            activation='relu',
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128,
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    if 2 == dim_choice:
        model.add(Dense(2, activation='linear'))
    else:
        model.add(Dense(1, activation='linear'))
    return model


def simple_plu1d(img_height, img_width, img_depth, dim_choice):
    """Uses own simple model as cnn topology

    Arguments:
        img_height: int, input image height
        img_width: int, input image width
        img_depth: int, numer of input image channels
        dim_choice: int, number of output dimensions"""
    model = Sequential()
    model.add(Conv2D(256,
        input_shape=(img_height, img_width, img_depth),
        kernel_size=(7, 7),
        padding='same',
        activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256,
                            kernel_size=(5, 5),
                            activation='relu',
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128,
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    if 2 == dim_choice:
        model.add(Dense(2, activation='linear'))
    else:
        model.add(Dense(1, activation='linear'))
    return model


def simple_plu2d(img_height, img_width, img_depth, dim_choice):
    """Uses own simple model as cnn topology

    Arguments:
        img_height: int, input image height
        img_width: int, input image width
        img_depth: int, numer of input image channels
        dim_choice: int, number of output dimensions"""
    model = Sequential()
    model.add(Conv2D(256,
        input_shape=(img_height, img_width, img_depth),
        kernel_size=(7, 7),
        padding='same',
        activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256,
                            kernel_size=(5, 5),
                            activation='relu',
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128,
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    if 2 == dim_choice:
        model.add(Dense(2, activation='linear'))
    else:
        model.add(Dense(1, activation='linear'))
    return model


def simple(img_height, img_width, img_depth, dim_choice):
    """Uses own simple model as cnn topology

    Arguments:
        img_height: int, input image height
        img_width: int, input image width
        img_depth: int, numer of input image channels
        dim_choice: int, number of output dimensions"""
    model = Sequential()
    model.add(Conv2D(256,
        input_shape=(img_height, img_width, img_depth),
        kernel_size=(7, 7),
        padding='same',
        activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256,
                            kernel_size=(5, 5),
                            activation='relu',
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128,
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    if 2 == dim_choice:
        model.add(Dense(2, activation='linear'))
    else:
        model.add(Dense(1, activation='linear'))
    return model


def simple_invcnv(img_height, img_width, img_depth, dim_choice):
    """Uses own simple model as cnn topology

    Arguments:
        img_height: int, input image height
        img_width: int, input image width
        img_depth: int, numer of input image channels
        dim_choice: int, number of output dimensions"""
    model = Sequential()
    model.add(Conv2D(128,
        input_shape=(img_height, img_width, img_depth),
        kernel_size=(7, 7),
        padding='same',
        activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256,
                            kernel_size=(5, 5),
                            activation='relu',
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(384,
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    if 2 == dim_choice:
        model.add(Dense(2, activation='linear'))
    else:
        model.add(Dense(1, activation='linear'))
    return model


def simple_invcnv_adv(img_height, img_width, img_depth, dim_choice):
    """Uses own simple model as cnn topology

    Arguments:
        img_height: int, input image height
        img_width: int, input image width
        img_depth: int, numer of input image channels
        dim_choice: int, number of output dimensions"""
    model = Sequential()
    model.add(Conv2D(128,
        input_shape=(img_height, img_width, img_depth),
        kernel_size=(7, 7),
        padding='same',
        activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256,
                            kernel_size=(5, 5),
                            activation='relu',
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(384,
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256,
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    if 2 == dim_choice:
        model.add(Dense(2, activation='linear'))
    else:
        model.add(Dense(1, activation='linear'))
    return model


def simple_small_leaky_relu(img_height, img_width, img_depth, dim_choice):
    """Simple model with LeakyReLU as activation function

    Arguments:
        img_height: int, input image height
        img_width: int, input image width
        img_depth: int, numer of input image channels
        dim_choice: int, number of output dimensions"""
    model = Sequential()
    model.add(
        Conv2D(
            96,
            input_shape=(img_height, img_width, img_depth),
            kernel_size=(7, 7),
            padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(7, 7), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.3))
    if 2 == dim_choice:
        model.add(Dense(2, activation='linear'))
    else:
        model.add(Dense(1, activation='linear'))
    return model


def simple_small(img_height, img_width, img_depth, dim_choice):
    """Simple model smaller...

    Arguments:
        img_height: int, input image height
        img_width: int, input image width
        img_depth: int, numer of input image channels
        dim_choice: int, number of output dimensions"""
    model = Sequential()
    model.add(
        Conv2D(
            96,
            input_shape=(img_height, img_width, img_depth),
            kernel_size=(7, 7),
            padding='same',
            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(7, 7), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    if 2 == dim_choice:
        model.add(Dense(2, activation='linear'))
    else:
        model.add(Dense(1, activation='linear'))
    return model


def simple_very_small(img_height, img_width, img_depth, dim_choice):
    """Simple model very small

    Arguments:
        img_height: int, input image height
        img_width: int, input image width
        img_depth: int, numer of input image channels
        dim_choice: int, number of output dimensions"""
    model = Sequential()
    model.add(
        Conv2D(
            96,
            input_shape=(img_height, img_width, img_depth),
            kernel_size=(3, 3),
            padding='same',
            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(7, 7), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    if 2 == dim_choice:
        model.add(Dense(2, activation='linear'))
    else:
        model.add(Dense(1, activation='linear'))
    return model


def simple_very_small_3l(img_height, img_width, img_depth, dim_choice):
    """Simple model very small with only 3+1 layers containing neurons

    Arguments:
        img_height: int, input image height
        img_width: int, input image width
        img_depth: int, numer of input image channels
        dim_choice: int, number of output dimensions"""
    model = Sequential()
    model.add(
        Conv2D(
            96,
            input_shape=(img_height, img_width, img_depth),
            kernel_size=(3, 3),
            padding='same',
            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    if 2 == dim_choice:
        model.add(Dense(2, activation='linear'))
    else:
        model.add(Dense(1, activation='linear'))
    return model

