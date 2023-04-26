
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
from keras.applications.inception_resnet_v2 import preprocess_input
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
import numpy as np

def create_inception_embedding(grayscaled_rgb, inception):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    embed = inception.predict(grayscaled_rgb_resized)
    return embed

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

# Generate training data

def image_a_b_gen(X_train, batch_size, inception):
    for batch in datagen.flow(X_train, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([X_batch, create_inception_embedding(grayscaled_rgb, inception)], Y_batch)

