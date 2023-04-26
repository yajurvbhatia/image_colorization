
import tensorflow as tf
from keras.applications import InceptionResNetV2
from keras.models import load_model
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, concatenate #,merge
from keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization
from keras.callbacks import TensorBoard 
from keras.models import Sequential, Model, load_model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from utils import create_inception_embedding, image_a_b_gen
from skimage.io import imsave
import numpy as np
import matplotlib.pyplot as plt

# import Image

def get_model():
    #Load weights
    inception = InceptionResNetV2(weights='imagenet', include_top=True)
    # inception.graph = tf.compat.v1.get_default_graph()
    embed_input = Input(shape=(1000,))

    #Encoder
    encoder_input = Input(shape=(256, 256, 1,))
    encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

    #Fusion
    fusion_output = RepeatVector(32 * 32)(embed_input) 
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output) 

    #Decoder
    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)

    return Model(inputs=[encoder_input, embed_input], outputs=decoder_output), inception

def train_model(model, inception, X_train):
    # Train model     
    tensorboard = TensorBoard(log_dir="output/sixth_run")
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(image_a_b_gen(X_train, 64, inception), epochs=70, steps_per_epoch=16)
    model.save('final_model.h5')
    return model

def loadmodel(model_name):
    # Load model
    return load_model(model_name)

def test_model(model, inception, X_test):
    embedded_gray_image = create_inception_embedding(X_test, inception)
    gray_image = rgb2lab(1.0/255*X_test)[:,:,:,0]
    gray_image = gray_image.reshape(gray_image.shape+(1,))

    # Test model
    output = model.predict([gray_image, embedded_gray_image])
    output = (output + 1)/2 #* 128

    # Output colorizations
    for i in range(len(output)):
        cur = np.zeros((256, 256, 3))
        cur[:,:,0] = gray_image[i][:,:,0]
        cur[:,:,1:] = output[i]
        cur = lab2rgb(cur)
        print(cur)
        plt.imsave("Results/semmedimg_"+str(i)+".png", cur)
        # imsave("Results/semmedimg_"+str(i)+".png", cur.astype(np.uint8)) # * 255).astype(np.uint8))