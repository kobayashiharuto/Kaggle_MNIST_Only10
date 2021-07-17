from sys import prefix
from matplotlib import image
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# 学習結果をpltで表示
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# test_df から、画像を取り出し、numpyの配列に変換
test_df = pd.read_csv("data/test.csv")
test_images = test_df.values
test_images = test_images.astype(np.float)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
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

# 結果を画像で出力
for i in range(10):
    img = test_images[i].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title('predict: ' + str(predict[i]))
    plt.show()
