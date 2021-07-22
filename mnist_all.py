from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import L2
from data_controller import get_image_and_labels
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Concatenate
from tensorflow.keras.layers import Layer


# 学習結果をpltで表示
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


train_df = pd.read_csv("data/train.csv")

# 訓練用データを読み込む
train_images, train_labels = get_image_and_labels(
    path=r'C:\Users\owner\Desktop\Image_tool\image_randomizer\out\mnist_data2')
train_images = train_images.reshape(train_images.shape + (1,))

# テスト用データを読み込む
test_data = pd.read_csv("data/train.csv")

# test_data から、画像とラベルを取り出し、numpyの配列に変換
test_images = test_data.iloc[:, 1:].values
test_images = test_images.astype(np.float)
test_labels = test_data.iloc[:, 0].values
test_labels = test_labels.astype(np.float)

# CNN で扱えるように次元を変換
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# 正規化
test_images = test_images / 255
train_images = train_images / 255

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# CNN でMNISTを分類するモデルを構築
model = Sequential([
    Conv2D(64, (3, 3), padding='Same',
           activation='relu',
           input_shape=(28, 28, 1),
           kernel_initializer='he_normal',
           ),
    Conv2D(64, (3, 3),
           padding='Same', activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Conv2D(64, (3, 3),
           padding='Same', activation='relu', kernel_initializer='he_normal'),
    Conv2D(64, (3, 3),
           padding='Same', activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(128, (3, 3),
           padding='Same', activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Conv2D(128, (3, 3),
           padding='Same', activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu', kernel_initializer='he_normal'
          ),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax'),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.003),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

history = model.fit(
    train_images, train_labels, epochs=500,
    batch_size=512,
    validation_data=(test_images, test_labels),
    callbacks=[
        EarlyStopping(monitor='loss', min_delta=0,
                      patience=15, verbose=1),
        ReduceLROnPlateau(monitor='val_acc',
                          patience=3,
                          verbose=1,
                          factor=0.5,
                          min_lr=0.00001),
        ModelCheckpoint('models/best_v2.h5', save_best_only=True)
    ],
)

show_train_history(history, 'acc', 'val_acc')

model.save('models/model.h5')

predict_df = pd.read_csv("data/test.csv")
predict_images = predict_df.values
predict_images = predict_images.astype(np.float)
predict_images = predict_images.reshape(predict_images.shape[0], 28, 28, 1)
predict_images = predict_images / 255

# test
model = tf.keras.models.load_model('models/model.h5')
predict = model.predict(predict_images)
predict = np.argmax(predict, axis=1)
predict = predict.astype(np.int32)

# 結果をcsvで出力
predict_df = pd.DataFrame(
    {"ImageId": range(1, len(predict)+1), "Label": predict})
predict_df.to_csv("result/result.csv", index=False)

# 結果を64枚一覧で表示
plt.figure(figsize=(10, 10))
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(f'{predict[i]}')
plt.show()


class Inception(Layer):
    def __init__(self, output_filter=64, **kwargs):
        super(Inception, self).__init__(output_filter, **kwargs)

        self.c1_conv1 = Conv2D(output_filter//4, 1, padding="same")
        self.c1_conv2 = Conv2D(output_filter//4, 3, padding="same")
        self.c1_conv3 = Conv2D(output_filter//4, 3, padding="same")

        self.c2_conv1 = Conv2D(output_filter//4, 1, padding="same")
        self.c2_conv2 = Conv2D(output_filter//4, 3, padding="same")

        self.c3_MaxPool = MaxPooling2D(pool_size=(2, 2), padding="same")
        self.c3_conv = Conv2D(output_filter//4, 1, padding="same")

        self.c4_conv = Conv2D(output_filter//4, 1, padding="same")

        self.concat = Concatenate()

    def call(self, input_x, training=False):
        x1 = self.c1_conv1(input_x)
        x1 = self.c1_conv2(x1)
        cell1 = self.c1_conv3(x1)

        x2 = self.c2_conv1(input_x)
        cell2 = self.c2_conv2(x2)

        x2 = self.c3_MaxPool(input_x)
        cell3 = self.c3_conv(x2)

        cell4 = self.c4_conv(input_x)

        return self.concat([cell1, cell2, cell3, cell4])
