from sys import prefix
from matplotlib import image
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


# CNN でMNISTを分類するモデルを構築
model = tf.keras.models.Sequential([
    Conv2D(64, (5, 5),
           activation='relu',
           padding='same',
           input_shape=(28, 28, 1)
           ),
    MaxPooling2D((3, 3)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    Conv2D(256, (3, 3), activation='relu'),
    Dropout(0.5),
    Flatten(),
    Dense(units=128, activation='relu'),
    Dropout(0.5),
    Dense(units=10, activation='softmax')
])


model.summary()
