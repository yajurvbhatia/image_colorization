import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
import numpy as np

def get_dataset(directory):
    """
    Returns preprocessed data as train and test sets
    """
    X = []
    Y = []
    for filename in os.listdir(directory):
        img = resize(img_to_array(load_img(directory+filename)), (256, 256))
        Y.append(img)
        X.append(gray2rgb(rgb2gray(img)))
    X = np.array(X, dtype=float)
    X = 1.0/255*X
    Y = np.array(Y, dtype=float)
    Y = 1.0/255*Y

    return train_test_split(X, Y, test_size=0.1, train_size=0.9, random_state=2, shuffle=True, stratify=None)

    # print(X.shape)
    # print(Y.shape)
    # print(X_train.shape)
    # print(Y_train.shape)
    # print(X_test.shape)
    # print(Y_test.shape)