import tensorflow as tf
import os
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

#Tensorflow dependencies (for Siamese neural network)
#Allows to pass through two images and get a similarity score
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
#Models are used to create the neural network (Models are made up of multiple layers)
#Layers are used to create the layers of the neural network (the following are types of layers)
#Conv2D is used to create a convolutional layer (A convolutional layer is a layer that applies a 
# convolution operation to the input) convolution measures the overlap between two images (calculus)
#Dense is used to create a fully connected layer (A fully connected layer is a layer that connects different neurons)
#MaxPooling2D shrinks information to take the most important features (reduces the size of the image)
#Input is used to create the input layer of the neural network
#Flatten is used to flatten the input to a 1D array

#Set GPU growth limit so that it doesn't use infinite RAM
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#Paths to the folders containing the images
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

#Make above directories if they don't exist
os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)