from re import A
from sys import prefix
from matplotlib import image
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


# 学習結果をpltで表示
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


test_df = pd.read_csv("data/test.csv")
train_df = pd.read_csv("data/train.csv")

# train_df から、画像とラベルを取り出し、numpyの配列に変換
train_images = train_df.iloc[:, 1:].values
train_images = train_images.astype(np.float)
train_labels = train_df.iloc[:, 0].values
train_labels = train_labels.astype(np.float)

# 一片のサイズを取得
size, = (train_images[0].shape)
side = int(size**0.5)

# CNN で扱えるように次元を変換
train_images = train_images.reshape(train_images.shape[0], side, side, 1)
# 正規化
train_images = train_images / 255

print(train_images.shape)
print(train_labels.shape)

# CNN でMNISTを分類するモデルを構築
model = tf.keras.models.Sequential([
    Conv2D(64, (5, 5),
           activation='relu',
           padding='same',
           input_shape=(28, 28, 1)
           ),
    MaxPooling2D((3, 3)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    Conv2D(256, (3, 3), activation='relu'),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_images, train_labels, epochs=500,
    batch_size=32, validation_split=0.1,
    callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
)

show_train_history(history, 'acc', 'val_acc')

model.save('models/model.h5')

# test_df から、画像を取り出し、numpyの配列に変換
test_images = test_df.values
test_images = test_images.astype(np.float)
test_images = test_images.reshape(test_images.shape[0], side, side, 1)
test_images = test_images / 255

# test
model = tf.keras.models.load_model('models/model.h5')
predict = model.predict(test_images)
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
