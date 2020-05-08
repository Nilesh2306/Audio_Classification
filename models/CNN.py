from tensorflow.keras import datasets, layers, models, Sequential
from tensorflow.keras.layers import BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dense, Input, Activation
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def get_CNN_model(input):
    """
    :param input:  Input features to the model
    :return: Returns the 2-D convolution model
    Architecture is inspired by the paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6479959/
    """

    model = Sequential()
    model.add(Conv2D(48, kernel_size=(3, 3), input_shape=input.shape[1:], activation='relu', padding='same',
                     bias_regularizer=l2(0.001), kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', bias_regularizer=l2(0.001),
                     kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Conv2D(192, kernel_size=(11, 11), activation='relu', padding='same', bias_regularizer=l2(0.001),
                     kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Conv2D(192, kernel_size=(5, 5), activation='relu', padding='same', bias_regularizer=l2(0.001),
                     kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', bias_regularizer=l2(0.001),
                     kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(512, activation='sigmoid', bias_regularizer=l2(0.001), kernel_regularizer=l2(0.001)))
    model.add(Dense(41, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001),
                  metrics=['accuracy'])
    return model
