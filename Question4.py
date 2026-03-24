#Question 4:
#Neural networks approach learning differently, relying on optimization rather than rule-based split-ting.
#Using the same training and testing data:
#• Standardize the input features.
#• Train a neural network with at least one hidden layer and a sigmoid output unit.
#• Report training accuracy and test accuracy.
#In Python comments, explain:
#• why feature scaling is necessary for neural networks, and
#• what an epoch represents during neural network training.

#First we will import the required libraries

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Input



#Loading the breast cancer dataset

#load the dataset into a variable called data
data = load_breast_cancer()

# create the feature matrix X
# this contains the 30 numeric tumor measurements
X = data.data

# create the target vector y
# 0 = malignant
# 1 = benign
y = data.target


#split the data into training and testing

# I am using the same kind of 80/20 split as before
# stratify=y keeps the class proportions similar in both sets
# random_state=42 makes the results reproducible

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Step 4: standardize the input features

# Neural networks usually perform much better when features are scaled.
# This is because the model learns by updating weights using optimization,
# and very different feature ranges can make training unstable or slower.

# StandardScaler transforms each feature so that it has:
# mean = 0
# standard deviation = 1

scaler = StandardScaler()

# fit the scaler only on the training data
# this learns the mean and standard deviation from the training set
X_train_scaled = scaler.fit_transform(X_train)

# use the same scaling transformation on the test data
# I do NOT fit again on the test set, because that would leak information
X_test_scaled = scaler.transform(X_test)


#Build the neural network

# I am creating a simple neural network with:
# • one hidden layer
# • a sigmoid output layer

# The hidden layer has 16 neurons and uses ReLU activation.
# ReLU helps the network learn non-linear patterns.

# The output layer has 1 neuron with sigmoid activation.
# Sigmoid is used because this is a binary classification problem,
# so the output will be a value between 0 and 1.
neural_network_model = Sequential()

# add an input layer first
neural_network_model.add(Input(shape=(30,)))

# hidden layer
neural_network_model.add(Dense(16, activation='relu'))

# output layer
neural_network_model.add(Dense(1, activation='sigmoid'))


#compiling the model

# compile tells the model how it should learn
# binary_crossentropy is the correct loss function for binary classification
# adam is a commonly used optimizer that adjusts the weights efficiently
# accuracy is included so that I can monitor classification performance

neural_network_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#Training the neural network

# epochs tells the model how many times to go through the full training dataset
# batch_size=32 means the model updates its weights after looking at
# 32 training examples at a time

# verbose=0 keeps the output clean
history = neural_network_model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=32,
    verbose=0
)


#Making predictions

# predict() returns probabilities between 0 and 1
train_probabilities = neural_network_model.predict(X_train_scaled, verbose=0)
test_probabilities = neural_network_model.predict(X_test_scaled, verbose=0)

# convert probabilities into class labels
# if the probability is 0.5 or more, classify as 1
# otherwise classify as 0
train_predictions = (train_probabilities >= 0.5).astype(int)
test_predictions = (test_probabilities >= 0.5).astype(int)


#calculate accuracy

# compare the predicted labels to the true labels
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

# print the results
print("Neural Network Training Accuracy:", train_accuracy)
print("Neural Network Test Accuracy:", test_accuracy)


#Interpretation

# Feature scaling is necessary for neural networks because the model learns
# by adjusting weights using optimization algorithms such as gradient descent.
# If some features have much larger values than others, they can dominate
# the learning process and make training slower or less effective.

# Standardizing the features helps the neural network train more smoothly,
# converge faster, and often achieve better performance.

# An epoch represents one complete pass through the entire training dataset.
# For example, if the model is trained for 50 epochs, that means it sees
# the whole training set 50 times and updates its weights repeatedly
# in order to improve its predictions.

# The neural network performed well on this dataset.
# The training accuracy is about 0.9890 and the test accuracy is about 0.9561.
# Since these two values are fairly close, the model appears to generalize
# well to unseen data, although there may be a small amount of overfitting.




