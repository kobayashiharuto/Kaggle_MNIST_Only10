from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Concatenate, ReLU
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from module.residual import ResBlock
from module.seblock import SEBlock


model = Sequential([
    Conv2D(64, (3, 3), padding='same', activation='relu',
           input_shape=(28, 28, 1),
           kernel_initializer='he_normal'),
    Conv2D(64, (3, 3), padding='same', activation='relu',
           kernel_initializer='he_normal'),
    SEBlock(64),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    ResBlock(64, 128),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Conv2D(128, (3, 3), padding='same', activation='relu',
           input_shape=(28, 28, 1),
           kernel_initializer='he_normal'),
    Conv2D(128, (3, 3), padding='same',  activation='relu',
           kernel_initializer='he_normal'),
    SEBlock(128),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(256, activation="relu"),
    Dropout(0.4),
    Dense(10, activation="softmax")
])

model.summary()
