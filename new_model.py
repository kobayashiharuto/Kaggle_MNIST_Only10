from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import L2
from tensorflow.keras import Input
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from data_controller import get_image_and_labels
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Concatenate
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from module.residual import ResidualBlock
from module.seblock import se_block


input_x = Input((28, 28, 1))
x = Conv2D(64, (3, 3), padding="same")(input_x)
x = se_block(x, 64)
x = ResidualBlock(64, 64)(x)
x = ResidualBlock(64, 64)(x)
x = Conv2D(128, (3, 3), strides=2, padding="same")(x)
x = se_block(x, 128)
x = ResidualBlock(128, 128)(x)
x = ResidualBlock(128, 128)(x)
x = Conv2D(256, (3, 3), strides=2, padding="same")(x)
x = MaxPooling2D(2, 2)(x)
x = se_block(x, 256)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output_x = Dense(10, activation="softmax")(x)

model = Model(input_x, output_x)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
