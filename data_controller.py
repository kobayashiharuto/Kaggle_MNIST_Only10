
import glob
from PIL import Image
import numpy as np


# ディレクトリパスから画像とラベルを取得
def get_image_and_labels(path, resize=0):
    paths = glob.glob(path + '/*')
    images = np.array([image_to_str_array(path, resize) for path in paths])
    labels = np.array([int(path.split('\\')[-1].split('_')[0])
                       for path in paths], dtype='u2')
    return images, labels


# 画像データを2次元配列にする
def image_to_str_array(img_path, resize):
    image = Image.open(img_path)
    image_gray = image.convert('L')
    if resize:
        image_gray = image_gray.resize((resize, resize))

    return np.array(image_gray, dtype='u1')
