import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def image_load():
    # 訓練用CSVから読み込み
    data_csv = pd.read_csv("data/train.csv")

    # 訓練用CSVから、画像とラベルを取り出し、numpyの配列に変換
    images = data_csv.iloc[:, 1:].values
    images = images.astype(np.float)
    labels = data_csv.iloc[:, 0].values
    labels = labels.astype(np.float)

    # データを分割
    train_images, test_images, train_labels, test_labels =\
        train_test_split(images, labels, test_size=0.1)

    return train_images, test_images, train_labels, test_labels


def target_data_load():
    target_data = pd.read_csv("data/test.csv")

    # target_data から、画像を取り出し、numpyの配列に変換
    target_images = target_data.iloc[:, 1:].values
    target_images = target_images.astype(np.float)

    return target_images
