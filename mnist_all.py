from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import L2
from data_controller import get_image_and_labels
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Concatenate, ReLU
from tensorflow.keras.layers import Layer
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from data_controller import get_image_and_labels
from tensorflow.keras import Model
from module.residual import ResBlock
from module.seblock import SEBlock
from tensorflow.keras.datasets import mnist


# 学習結果をpltで表示
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# データをロード
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 正規化
train_images = train_images / 255
test_images = test_images / 255

# CNN で扱えるように次元を変換
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# 画像をランダム化する処理を追加
image_generater = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=10,
    zoom_range=0.1,  # Randomly zoom image
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False
)

# 訓練用画像にランダム化を適用
image_generater.fit(train_images)

# モデルを設計
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

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.003),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

history = model.fit(
    image_generater.flow(train_images, train_labels, batch_size=128),
    epochs=500,
    validation_data=(test_images, test_labels),
    callbacks=[
        EarlyStopping(monitor='loss', min_delta=0,
                      patience=5, verbose=1),
        ReduceLROnPlateau(monitor='val_acc',
                          patience=3,
                          verbose=1,
                          factor=0.5,
                          min_lr=0.00001),
        ModelCheckpoint('models/best_v7.h5', save_best_only=True)
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
model = tf.keras.models.load_model(
    'models/best_v7.h5', custom_objects={'SEBlock': SEBlock, 'ResBlock': ResBlock})
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
