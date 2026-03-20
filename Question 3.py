#Question 3:
#Controlling Tree Complexity and Interpretability (10 points)
#Unconstrained decision trees can easily overfit training data.
#Modify the Decision Tree model by introducing at least one constraint (e.g., max depth, min samples split,
#or a similar parameter):
#• Train the constrained model and report training and test accuracy.
#• Display the top five most important features according to the model.
#In Python comments, briefly discuss:
#• how controlling model complexity affects overfitting, and
#• how feature importance contributes to the interpretability of decision trees.

#First we have to import the required libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

#load the breast cancer dataset
data = load_breast_cancer()

#Create the feature matrix X
#this contains the 30 numeric input features for each sample
X = data.data

#create the target vector y
#This contains the class labels:
# 0 = malignant
#1 = benign
y = data.target

# save the feature names so that later I can display
# which features were the most important in the decision tree
feature_names = data.feature_names

#split the data into training and testing sets
#stratify=y keeps the class distribution similar in both sets

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,  #20% of data used for testing
    stratify=y,     #keeps the class distribution consistent
    random_state=42  #ensures reproducibility
)

#create a constrained decision tree model

# In Q2, the tree had no real constraint, so it could grow freely
# and this often leads to overfitting

# Here I add max_depth=3 as a constraint
# This means the tree can only grow to a depth of 3 levels

# Limiting the depth helps prevent the model from becoming too complex
# It also makes the final tree easier to understand and interpret

# I am still using criterion="entropy" because the question is still
# about decision trees that split using information gain

constrained_tree_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)

#train the constrained model

# fit() trains the model using the training data
# the model learns patterns that connect the feature values to the class labels
constrained_tree_model.fit(X_train, y_train)

#make predictions on the training data

# predict the class labels for the test set
# this helps me evaluate how well the model generalizes to unseen data
train_predictions = constrained_tree_model.predict(X_train)

# predict the class labels for the test set
# this helps me evaluate how well the model generalizes to unseen data
test_predictions = constrained_tree_model.predict(X_test)

#Calaculate the training and testing accuracy

# training accuracy tells me the proportion of training examples
# the model classified correctly
train_accuracy = accuracy_score(y_train, train_predictions)

# test accuracy tells me the proportion of test examples
# the model classified correctly
test_accuracy = accuracy_score(y_test, test_predictions)

#print the accuracy results
print("Constrained Model Training Accuracy:", train_accuracy)
print("Constrained Model Test Accuracy:", test_accuracy)

# get the feature importance values from the trained tree

# feature_importances_ gives an importance score for every feature
# higher values mean that feature played a bigger role in making splits
importances = constrained_tree_model.feature_importances_

#create a dataframe to pair each feature name with its importance score
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# sort the features from most important to the least important
feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

# display the top 5 most important features
print("\nTop 5 Most Important Features:")
print(feature_importance_df.head(5))


# Interpretation

# Controlling model complexity helps reduce overfitting.
# If a decision tree is allowed to grow without limits, it can memorize
# the training data too closely, which usually leads to very high training
# accuracy but worse performance on unseen test data.

# By adding a constraint such as max_depth, the tree is forced to stay simpler.
# A simpler tree usually generalizes better because it focuses on the most
# important patterns instead of fitting every small detail in the training set.

# Feature importance helps make decision trees interpretable because it shows
# which input variables had the biggest influence on the model's decisions.

# In this model, the top features are the ones the tree relied on the most
# when making splits. This is useful because we can better understand which
# tumor measurements were most important for predicting whether a tumor is
# malignant or benign.







