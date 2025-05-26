import tensorflow as tf
import os
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import uuid #Allows to create a unique ID for each image

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

#Make above directories if they don't exist (uncomment the lines below on first run)
#os.makedirs(POS_PATH)
#os.makedirs(NEG_PATH)
#os.makedirs(ANC_PATH)

#Code to move images to the created folders
#Images are taken from Kaggle dataset (https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?resource=download)
#Uncomment for first run and potentially change the paths

# for directory in os.listdir('faces//lfw-deepfunneled//lfw-deepfunneled'):
#     for file in os.listdir(os.path.join('faces//lfw-deepfunneled//lfw-deepfunneled', directory)):
#         EX_PATH = os.path.join('faces', 'lfw-deepfunneled', 'lfw-deepfunneled', directory, file)
#         NEW_PATH = os.path.join(NEG_PATH, file)
#         os.replace(EX_PATH, NEW_PATH)
        
#Access webcamera 
#Collect positive and anchor images (get about 300 of each)

# cap = cv2.VideoCapture(0)
# while cap.isOpened(): #Loops through every frame of webcam
#     ret, frame = cap.read() #Read capture at that point in time
#     frame = frame[120:120+250, 200:200+250, :] #Crop image
    

#     #Collect Anchor Images
#     if cv2.waitKey(1) & 0xFF == ord('a'): #Waits one millisecond for pressing 'a' so it can capture image
#         imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1())) #Creates uniwque name for image in anchor folder
#         cv2.imwrite(imgname, frame) #Writes the image to the anchor folder

#     #Collect Positive Images
#     if cv2.waitKey(1) & 0xFF == ord('p'): #Waits one millisecond for pressing 'p' so it can capture image
#         imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1())) #Creates uniwque name for image in anchor folder
#         cv2.imwrite(imgname, frame) #Writes the image to the anchor folder

#     cv2.imshow('Image Collection', frame) #Display the frame

#     #Breaking gracefully
#     if cv2.waitKey(1) & 0xFF == ord('q'): #Waits one millisecond for pressing 'q' so it can close frame
#         break
# cap.release()
# cv2.destroyAllWindows()

#Collect data

anchor = tf.data.Dataset.list_files(ANC_PATH + '/*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH + '/*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH + '/*.jpg').take(300)

#Changes the values of the image to be between 0-1 instead of 0-255
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path) #read in image from file path
    img = tf.io.decode_jpeg(byte_img) #Load in image
    img = tf.image.resize(img, (100, 100)) #Resizing image to be 100x100x3 (lowers quality a bit but improves runtime)
    img = img/255.0 #changes each pixel value from 0-255 to 0-1
    return img

#Convert our data list to an array of ones and zeros
#(anchor, positive) => 1, 1, 1, 1, 1
#(anchor, negative) => 0, 0, 0, 0, 0

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)
samples = data.as_numpy_iterator()


def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

#Dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)
samples = data.as_numpy_iterator()
