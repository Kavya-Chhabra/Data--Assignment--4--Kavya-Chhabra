# Q7 — CNN Error Analysis and Misclassification Study (10 points)
# In practical machine learning systems, analyzing model errors is just as
# important as reporting accuracy.
#
# Using the trained CNN from Question 6, I will:
# • generate predictions on the test set
# • compute and display the confusion matrix
# • identify and visualize at least three misclassified images
# • clearly show the true label and predicted label for each one
#
# In Python comments, I will also briefly discuss:
# • one pattern observed in the misclassifications
# • one realistic method to improve the CNN performance


# import the required libraries


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

from sklearn.metrics import confusion_matrix


# load the Fashion MNIST dataset


# load the training and testing data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# normalize pixel values to the range [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# reshape the images so they include the channel dimension
# grayscale images have 1 channel
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


# build the CNN model


# create a simple CNN similar to Question 6
cnn_model = Sequential()

# input layer
cnn_model.add(Input(shape=(28, 28, 1)))

# convolution layer
cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

# max pooling layer
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten layer to turn feature maps into a 1D vector
cnn_model.add(Flatten())

# hidden dense layer
cnn_model.add(Dense(128, activation='relu'))

# output layer for 10 classes
cnn_model.add(Dense(10, activation='softmax'))

# compile the model
cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# train the CNN model


# train for at least 15 epochs as required in Q6
cnn_model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    verbose=1
)


#generate predictions on the test set

# predict class probabilities for the test images
test_probabilities = cnn_model.predict(X_test, verbose=0)

# convert the probabilities into predicted class labels
y_pred = np.argmax(test_probabilities, axis=1)


# compute and display the confusion matrix


cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# identify misclassified images


# misclassified images are the ones where predicted label
# does not match the true label
misclassified_indices = np.where(y_pred != y_test)[0]

print("\nNumber of misclassified images:", len(misclassified_indices))


#define label names for Fashion MNIST

# these are the class names for labels 0 through 9
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# visualize at least 3 misclassified images

# display the first 3 misclassified images
plt.figure(figsize=(12, 4))

for i in range(3):
    index = misclassified_indices[i]

    # get the image, true label, and predicted label
    image = X_test[index].reshape(28, 28)
    true_label = y_test[index]
    predicted_label = y_pred[index]

    # create a subplot for each misclassified image
    plt.subplot(1, 3, i + 1)
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.title(
        f"True: {class_names[true_label]}\nPred: {class_names[predicted_label]}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()


# Discussion / Interpretation


# One pattern I observe in the misclassifications is that the model may confuse
# clothing items that have similar overall shapes or visual features.
# For example, items such as shirts, pullovers, coats, and dresses can look
# somewhat similar in small grayscale images, especially when details are limited.

# Another example from the misclassified images is that some footwear items
# can be confused with other categories when their shapes appear unusual
# or when important visual details are less obvious.


# One realistic way to improve CNN performance would be to use a deeper CNN
# with more convolution layers so the model can learn more detailed image features.
# Another helpful improvement could be data augmentation, which creates slightly
# modified training images and helps the model generalize better.