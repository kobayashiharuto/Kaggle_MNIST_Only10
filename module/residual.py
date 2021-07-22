from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import Add, Layer
from tensorflow.keras.layers import ReLU


class ResidualBlock(Layer):
    def __init__(self, channel_in=64, channel_out=256):
        super().__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out

        channel = channel_out // 4

        self.conv1 = Conv2D(channel, kernel_size=(1, 1), padding="same")
        self.bn1 = BatchNormalization()
        self.av1 = ReLU()
        self.conv2 = Conv2D(channel, kernel_size=(3, 3), padding="same")
        self.bn2 = BatchNormalization()
        self.av2 = ReLU()
        self.conv3 = Conv2D(channel_out, kernel_size=(1, 1), padding="same")
        self.bn3 = BatchNormalization()
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.add = Add()
        self.av3 = ReLU()

    def call(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.av1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.av2(h)
        h = self.conv3(h)
        h = self.bn3(h)
        shortcut = self.shortcut(x)
        h = self.add([h, shortcut])
        y = self.av3(h)
        return y

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'channel_in': self.channel_in,
            'channel_out': self.channel_out,
        })
        return config

    def _shortcut(self, channel_in, channel_out):
        if channel_in == channel_out:
            return lambda x: x
        else:
            return self._projection(channel_out)

    def _projection(self, channel_out):
        return Conv2D(channel_out, kernel_size=(1, 1), padding="same")
