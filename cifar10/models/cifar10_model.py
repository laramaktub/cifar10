import os
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
K.set_image_dim_ordering('th')


labels=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def train_nn(epochs, lrate, outputpath):


    K.clear_session()
    #Training parameters
    epochs = epochs
    lrate = lrate
    decay = lrate/epochs

    # load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Transform to hot vectors

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    print("We are running on %i classes" % num_classes)

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(3, 32, 32)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())


    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

    #Evaluate the model on the test dataset
    scores = model.evaluate(X_test, y_test, verbose=10)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    #Save the model
    model.save(os.path.join(outputpath,"model.h5"))

def predict_nn(image):
    K.clear_session()
    model=load_model("model.h5")
    image = numpy.expand_dims(image, axis=0)
    index_prediction=numpy.argmax(model.predict(image))
    message= "The image is a " + labels[index_prediction]
    if labels[index_prediction]=="airplane" or labels[index_prediction]=="automobile":
        message= "The image is an " + labels[index_prediction]

    return message
