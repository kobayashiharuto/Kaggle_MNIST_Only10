from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Concatenate, GlobalAveragePooling2D, Multiply


def se_block(input, channels, rate=8):
    x = GlobalAveragePooling2D()(input)
    x = Dense(channels//rate, activation="relu")(x)
    x = Dense(channels, activation="sigmoid")(x)
    return Multiply()([input, x])
