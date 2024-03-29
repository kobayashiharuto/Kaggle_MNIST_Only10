from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Multiply


class SEBlock(Layer):
    def __init__(self, channels, rate=8, name=None, **kwargs):
        super().__init__()
        self.channels = channels
        self.rate = rate

        self.gap = GlobalAveragePooling2D()
        self.relu = Dense(channels//rate, activation="relu")
        self.sigmoid = Dense(channels, activation="sigmoid")
        self.multi = Multiply()

    def call(self, input):
        x = self.gap(input)
        x = self.relu(x)
        x = self.sigmoid(x)
        x = self.multi([input, x])
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'channels': self.channels,
            'rate': self.rate,
        })
        return config
