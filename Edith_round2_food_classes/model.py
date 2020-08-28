from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class SmallerVGGNet:
	@staticmethod
	def build(width, height, depth, classes, finalAct="softmax"):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
    # CONV => RELU => POOL
		model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3)))
		model.add(Dropout(0.25))
    	# (CONV => RELU) * 2 => POOL
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		# (CONV => RELU) * 2 => POOL
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
    # first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		# use a *softmax* activation for single-label classification
		# and *sigmoid* activation for multi-label classification
		model.add(Dense(classes))
		model.add(Activation(finalAct))
		# return the constructed network architecture
		return model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
# from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 75
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

# grab the image paths and randomly shuffle them
# print("[INFO] loading images...")
# imagePaths = sorted(list(paths.list_images(args["dataset"])))
# random.seed(42)
# random.shuffle(imagePaths)
# initialize the data and labels
data = []
labels = []

path='/content/drive/My Drive/recognizance/recognizance1/train/0'
for i in os.listdir(path):

    # print(path)
    try:
        image = cv2.imread(path+'/'+str(i))

        image = cv2.resize(image, (IMAGE_DIMS[0],IMAGE_DIMS[1]))
        image = img_to_array(image)
        data.append(image)
        labels.append('0')
    except:

        continue


path='/content/drive/My Drive/recognizance/recognizance1/train/1'
for i in os.listdir(path):

    # print(path)
    try:
        image = cv2.imread(path+'/'+str(i))

        image = cv2.resize(image, (IMAGE_DIMS[0],IMAGE_DIMS[1]))
        image = img_to_array(image)
        data.append(image)
        labels.append('1')
    except:

        continue

path='/content/drive/My Drive/recognizance/recognizance1/train/2'
for i in os.listdir(path):

    # print(path)
    try:
        image = cv2.imread(path+'/'+str(i))

        image = cv2.resize(image, (IMAGE_DIMS[0],IMAGE_DIMS[1]))
        image = img_to_array(image)
        data.append(image)
        labels.append('2')
    except:

        continue

path='/content/drive/My Drive/recognizance/recognizance1/train/3'
for i in os.listdir(path):

    # print(path)
    try:
        image = cv2.imread(path+'/'+str(i))

        image = cv2.resize(image, (IMAGE_DIMS[0],IMAGE_DIMS[1]))
        image = img_to_array(image)
        data.append(image)
        labels.append('3')
    except:

        continue


path='/content/drive/My Drive/recognizance/recognizance1/train/4'
for i in os.listdir(path):

    # print(path)
    try:
        image = cv2.imread(path+'/'+str(i))

        image = cv2.resize(image, (IMAGE_DIMS[0],IMAGE_DIMS[1]))
        image = img_to_array(image)
        data.append(image)
        labels.append('4')
    except:

        continue

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print(labels)

print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.2, random_state=42)
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

EPOCHS = 300

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")
# initialize the optimizer
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

print("[INFO] serializing network...")
model.save("model")
# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open("tags", "wb")
f.write(pickle.dumps(mlb))
f.close()
