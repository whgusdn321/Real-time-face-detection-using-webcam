from imutils import paths
from keras.preprocessing.image import img_to_array
import keras.utils
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import cv2
import os
import numpy

from GNet import GNet

trainingImages = [] # Place to store all our prepared images
trainingLabels = [] # Place to store all labels for images

# First, read the image and convert it to grayscale and the same width
for pathToImage in paths.list_images('data/'):
    print('pathToImage : ',pathToImage)
    image = cv2.imread(pathToImage)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    image = cv2.resize(image, (70, 70)) # Use resize image into 70 by 70. I chose 70 by 70 because I thought average faces appeared over the window was 70 by 70
    image = img_to_array(image) # Convert image into an array that keras can use
    trainingImages.append(image)

    label = pathToImage.split(os.path.sep)[-2][5:] # Get directory that each image is in
    print('label :',label)
    if label == 'head':
        label = 'head'
    else:
        label = 'not_head'
    trainingLabels.append(label)

trainingImages = numpy.array(trainingImages, dtype='float') / 255.0 # Convert matrix into array of floats between 0 and 1
trainingLabels = numpy.array(trainingLabels) # Convert to numpy array

le = sklearn.preprocessing.LabelEncoder().fit(trainingLabels) # Initialize LabelEncoder
trainingLabels = keras.utils.to_categorical(le.transform(trainingLabels), 2) # Convert into a binary class matrix

print('trainingLabels ',trainingLabels)
classTotals = trainingLabels.sum(axis=0) # Find whichever category has more entries
print('classTotals : ',classTotals)
classWeight = classTotals.max() / classTotals # Create a weight matrix to balance this out

(x, X, y, Y) = train_test_split(trainingImages, trainingLabels, test_size=0.20, stratify=trainingLabels,
    random_state=1) # Split the data into training and testing sets

epochs = 300
gNetModel = GNet.build(width = 70, height=70, depth=1, classes=2)
gNetModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
H = gNetModel.fit(x, y, validation_data=(X, Y), class_weight=classWeight,
    batch_size=64, epochs=epochs, verbose=1)

gNetModel.save('Gnet2.h5py')
