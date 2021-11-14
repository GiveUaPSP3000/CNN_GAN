import matplotlib
import sklearn
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
from skimage import io
from numpy import fliplr

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import BatchNormalization
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Design the model layers
def createModel():
    model = Sequential()
    input_data = (512, 512, 1)
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_data, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='mse',
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.9, nesterov=True),
        metrics=['accuracy'])
    return model


def generate_for_kp_test(file_list, batch_size):
    while True:
        count = 0
        x= []
        for path in file_list:
            if path.startswith('*'):
                path = path.strip('*#')
                img=io.imread('E:/cat/original_images/'+path, as_gray=True)
                img = fliplr(img)
            else:
                path = path.strip('#')
                img=io.imread('E:/cat/original_images/'+path, as_gray=True)
            img = np.array(img)
            count += 1
            x.append(img)
            if count % batch_size == 0 and count != 0:
                x = np.array(x)
                x = x.reshape(batch_size, 512, 512, 1).astype("float32")
                yield x
                x = []


model = []
filepath = [
    'CNN_Left_Eye_(Left).hdf5',
    'CNN_Left_Eye(Top).hdf5',
    'CNN_Left_Eye_(Right).hdf5',
    'CNN_Left_Eye(Bottom).hdf5',
    'CNN_Right_Eye_(Left).hdf5',
    'CNN_Right_Eye(Top).hdf5',
    'CNN_Right_Eye_(Right).hdf5',
    'CNN_Right_Eye(Bottom).hdf5',
    'CNN_Nose.hdf5',
    'CNN_Lip_Left.hdf5',
    'CNN_Upper_Lip.hdf5',
    'CNN_Lip_Right.hdf5',
    'CNN_Lower_Lip.hdf5']
for i in range(0, 13):
    model.append(createModel())
    model[i].load_weights(filepath)
    model[i].compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])


def get_result(file_test):
    result = [[], []]
    for cu_model in model:
        predicts = cu_model.predict_generator(generate_for_kp_test(file_test, 1), steps=int(len(file_test)/2)+1, verbose=1)
        for r in [0, 1]:
            result[r].append([predicts[r]])
    return result
