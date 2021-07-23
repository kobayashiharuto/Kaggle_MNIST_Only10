from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Concatenate, ReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from module.residual import ResBlock
from module.seblock import SEBlock
from image_loader import image_load, target_data_load


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
train_images, train_labels, test_images, test_labels = image_load()

# 正規化
train_images = train_images / 255
test_images = test_images / 255

# CNN で扱えるように次元を変換
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# 画像をランダム化する処理を追加
image_generater = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
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
    ResBlock(128, 128),
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
    ResBlock(128, 128),
    ResBlock(128, 128),
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

# 推論対象をロードする
target_images = target_data_load()
target_images = target_images / 255
target_images = target_images.reshape(target_images[0].shape, 28, 28, 1)

# 推論
model = tf.keras.models.load_model(
    'models/best_v7.h5', custom_objects={'SEBlock': SEBlock, 'ResBlock': ResBlock})
predict = model.predict(target_images)
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
