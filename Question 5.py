# Q5 — Model Evaluation and Comparison (5 points)
# Different models may perform similarly in terms of accuracy,
# yet behave very differently in practice.
# For the constrained Decision Tree and the Neural Network:
# • Compute and display the confusion matrix for each model.
# In Python comments, provide a concise comparison addressing:
# • which model you would prefer for this task, and
# • one advantage and one limitation of each model.


#import the required libraries

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


# Step 2: load the dataset

# load the breast cancer dataset
data = load_breast_cancer()

# create the feature matrix X and target vector y
X = data.data
y = data.target


#create the same train-test split


# using the same 80/20 split with stratification
# this keeps the class proportions similar in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


#train the constrained Decision Tree


# use the same constrained tree idea from Q3
decision_tree_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)

# train the tree
decision_tree_model.fit(X_train, y_train)

# make predictions on the test set
tree_predictions = decision_tree_model.predict(X_test)

# compute confusion matrix for the decision tree
tree_cm = confusion_matrix(y_test, tree_predictions)

print("Confusion Matrix for Constrained Decision Tree:")
print(tree_cm)


#Standardize the data for the Neural Network

# neural networks work better when input features are scaled
scaler = StandardScaler()

# fit on training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# build and train the Neural Network


# create a simple neural network with one hidden layer
neural_network_model = Sequential()

# add input layer
neural_network_model.add(Input(shape=(30,)))

# hidden layer
neural_network_model.add(Dense(16, activation='relu'))

# output layer for binary classification
neural_network_model.add(Dense(1, activation='sigmoid'))

# compile the model
neural_network_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# train the model
neural_network_model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=32,
    verbose=0
)

# make predictions on the test set
nn_probabilities = neural_network_model.predict(X_test_scaled, verbose=0)

# convert probabilities into class labels
nn_predictions = (nn_probabilities >= 0.5).astype(int).flatten()

# compute confusion matrix for the neural network
nn_cm = confusion_matrix(y_test, nn_predictions)

print("\nConfusion Matrix for Neural Network:")
print(nn_cm)



# Interpretation / Comparison

# I would prefer the neural network for this task because it achieved
# slightly better test performance and may capture more complex patterns
# in the data.

# One advantage of the constrained decision tree is interpretability.
# It is easier to understand because we can see which features are used
# to make decisions and examine the feature importance values.

# One limitation of the constrained decision tree is that it may be less
# flexible than a neural network and may miss more complex relationships
# in the data.

# One advantage of the neural network is that it can learn more complex
# patterns and may achieve better predictive performance.

# One limitation of the neural network is that it is less interpretable.
# It is harder to clearly explain exactly how it reached a prediction
# compared with a decision tree.
