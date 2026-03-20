# Question 6:
# Convolutional Neural Networks (CNNs) are widely used for image classification
# because they can automatically learn spatial patterns from pixel data.
# In this question, I will train a basic CNN using the Fashion MNIST dataset.

# Tasks:
# • Load the Fashion MNIST dataset
# • Normalize the pixel values to the range [0,1]
# • Reshape the images to include the channel dimension
# • Build a CNN that includes at least:
#   - one Conv2D layer
#   - one MaxPooling2D layer
#   - one Dense output layer
# • Train the model for at least 15 epochs
# • Report the test accuracy
# In Python comments, briefly explain:
# • why CNNs are generally preferred over fully connected networks for image data
# • what the convolution layer is learning in this task


# Import the required libraries

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


import tensorflow as tf


from tensorflow.keras.datasets import fashion_mnist


from tensorflow.keras.models import Sequential


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense



# Load the Fashion MNIST dataset


# X_train and X_test contain the image data
# y_train and y_test contain the class labels
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# print the original shapes just to better understand the data
print("Original X_train shape:", X_train.shape)
print("Original X_test shape:", X_test.shape)

# Fashion MNIST contains grayscale images of size 28 x 28
# There are 60,000 training images and 10,000 test images



# Normalize the pixel values

# pixel values originally go from 0 to 255
# dividing by 255.0 rescales them to the range [0, 1]
# this helps the neural network train more effectively
X_train = X_train / 255.0
X_test = X_test / 255.0



# Reshape the images to include the channel dimension


# CNNs expect image data in the format:
# (number of samples, height, width, channels)

# Since Fashion MNIST images are grayscale, each image has 1 channel
# so I reshape from (28, 28) to (28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# print the new shapes after reshaping
print("Reshaped X_train shape:", X_train.shape)
print("Reshaped X_test shape:", X_test.shape)



# Build the CNN model


# I am creating a simple CNN with:
# • one convolutional layer
# • one max pooling layer
# • one flatten layer
# • one hidden dense layer
# • one output dense layer

cnn_model = Sequential()

# Conv2D applies filters to the image to detect useful patterns
# such as edges, curves, and textures
# 32 means there are 32 different filters
# kernel_size=(3,3) means each filter is 3x3
# relu is used as the activation function

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

cnn_model = Sequential()

# add input layer first
cnn_model.add(Input(shape=(28, 28, 1)))

# convolution layer
cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

# max pooling layer
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten layer
cnn_model.add(Flatten())

# hidden dense layer
cnn_model.add(Dense(128, activation='relu'))

# output layer
cnn_model.add(Dense(10, activation='softmax'))




# Compile the model


# sparse_categorical_crossentropy is used because:
# - this is a multi-class classification problem
# - the target labels are integer encoded, not one-hot encoded

cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)



# Train the model

# the question asks for at least 15 epochs
# I will train for exactly 15 epochs
history = cnn_model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    verbose=1
)



# Evaluate the model on the test data


# evaluate returns loss and accuracy on the test set
test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)

# print the final test accuracy
print("Test Accuracy:", test_accuracy)



# Interpretation


# CNNs are generally preferred over fully connected networks for image data
# because they are able to preserve and learn spatial relationships between pixels.
# Nearby pixels in an image are related to each other, and CNNs can capture
# patterns such as edges, shapes, and textures much better than a fully connected network.

# A fully connected network would treat every pixel more independently after flattening,
# which loses important spatial structure in the image.

# In this task, the convolution layer is learning useful visual patterns from the clothing images.
# For example, it may learn to detect edges, corners, outlines, textures,
# and simple shapes that help distinguish items such as shirts, shoes, bags, and dresses.

# The CNN performed well on this dataset, achieving a test accuracy
# of about 0.9148 after 15 epochs of training.