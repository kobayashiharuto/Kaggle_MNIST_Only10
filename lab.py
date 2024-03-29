from sys import prefix
from matplotlib import image
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.models import Sequential


# CNN でMNISTを分類するモデルを構築
model = Sequential([
    Conv2D(64, (3, 3), padding='Same',
           activation='relu',
           input_shape=(28, 28, 1),
           kernel_initializer='he_normal'
           ),
    Conv2D(64, (3, 3),
           padding='Same', activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    MaxPooling2D((3, 3)),
    Dropout(0.3),
    Conv2D(128, (3, 3),
           padding='Same', activation='relu', kernel_initializer='he_normal'),
    Conv2D(128, (3, 3),
           padding='Same', activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    MaxPooling2D((3, 3)),
    Dropout(0.3),
    Conv2D(256, (3, 3),
           padding='Same', activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Conv2D(256, (3, 3),
           padding='Same', activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax'),
])
model.summary()
