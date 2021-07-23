from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Concatenate, ReLU
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from module.residual import ResBlock
from module.seblock import SEBlock


model = Sequential([
    Conv2D(16, (3, 3), padding='same',
           input_shape=(28, 28, 1),
           kernel_initializer='he_normal'),
    BatchNormalization(),
    ReLU(),
    SEBlock(16),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    ResBlock(32, 32),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    ResBlock(32, 64),
    MaxPooling2D(),
    Dropout(0.3),
    Conv2D(64, (3, 3),
           padding='same',
           activation='relu',
           kernel_initializer='he_normal'),
    MaxPooling2D(2, 2),
    SEBlock(64),
    Dropout(0.4),
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(256, activation="relu"),
    Dropout(0.4),
    Dense(10, activation="softmax")
])

model.summary()
