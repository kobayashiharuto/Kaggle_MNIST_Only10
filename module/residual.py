from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import Add, Layer
from tensorflow.keras.layers import ReLU


class ResBlock(Layer):
    def __init__(self, channel_in=64, channel_out=256, name=None, **kwargs):
        super().__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out

        channel = channel_out // 4

        self.conv1 = Conv2D(channel, (1, 1),
                            padding='same',
                            kernel_initializer='he_normal')
        self.bn1 = BatchNormalization()
        self.av1 = ReLU()
        self.conv2 = Conv2D(channel, (3, 3),
                            padding='same',
                            kernel_initializer='he_normal')
        self.bn2 = BatchNormalization()
        self.av2 = ReLU()
        self.conv3 = Conv2D(channel_out, (1, 1),
                            padding='same',
                            kernel_initializer='he_normal')
        self.bn3 = BatchNormalization()
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.add = Add()
        self.av3 = ReLU()

    def call(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.av1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.av2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        shortcut = self.shortcut(x)
        x = self.add([x, shortcut])
        x = self.av3(x)
        return x

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


class WideResBlock(Layer):
    def __init__(self, channel_in=64, channel_out=256, name=None, **kwargs):
        super().__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out

        channel = channel_out // 4

        self.conv1 = Conv2D(channel, (1, 1),
                            padding='same',
                            kernel_initializer='he_normal')
        self.bn1 = BatchNormalization()
        self.av1 = ReLU()
        self.conv2 = Conv2D(channel, (3, 3),
                            padding='same',
                            kernel_initializer='he_normal')
        self.bn2 = BatchNormalization()
        self.av2 = ReLU()
        self.conv3 = Conv2D(channel_out, (1, 1),
                            padding='same',
                            kernel_initializer='he_normal')
        self.bn3 = BatchNormalization()
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.add = Add()
        self.av3 = ReLU()

    def call(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.av1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.av2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        shortcut = self.shortcut(x)
        x = self.add([x, shortcut])
        x = self.av3(x)
        return x

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
