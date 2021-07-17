import os
from re import A
from sys import prefix
from matplotlib import image
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


test_df = pd.read_csv("data/test.csv")
train_df = pd.read_csv("data/train.csv")

# train_df から、画像とラベルを取り出し、numpyの配列に変換
train_images = train_df.iloc[:, 1:].values
train_images = train_images.astype(np.float)
train_labels = train_df.iloc[:, 0].values
train_labels = train_labels.astype(np.float)


# train_images をラベルごとに配列として分割した配列を作る
train_images = np.array(train_images)
train_images = train_images.reshape(
    train_images.shape[0], 28, 28).astype(np.uint8)

# ラベル別に配列に分ける
train_images_list = [train_images[train_labels == i] for i in range(10)]
train_images_list = np.array(train_images_list)

for label, train_images in enumerate(train_images_list):
    for index, image in enumerate(train_images):
        image_data = Image.fromarray(image, mode='L')
        image_data.save(f'images/{label}_{index}.png')
