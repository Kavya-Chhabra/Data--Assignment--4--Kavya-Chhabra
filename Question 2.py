#Q2 — Decision Tree Model Using Entropy (10 points)
#Decision trees aim to reduce uncertainty in the target variable by selecting splits that maximize
#information gain.
#Using an 80/20 train–test split with stratification:
#• Train a Decision Tree classifier using entropy as the splitting criterion.
#• Report the training accuracy and test accuracy of the model.
#In Python comments, explain:
#• what entropy represents in the context of decision trees, and
#• whether the observed results suggest overfitting or good generalization.

#first I will import the required libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# load the dataset
data = load_breast_cancer()

# create feature matrix X and target vector y
X = data.data
y = data.target


#Train-test split (80/20) with stratification

#stratification ensures that the proportion of malignant and benign
# samples remains roughly the same in both the training and testing sets.
#This is important because if one class become overrepresented in either
#split, the model evaluation might become misleading.

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,  #20% of data used for testing
    stratify=y,     #keeps the class distribution consistent
    random_state=42  #ensures reproducibility
)

#Training the model using entropy

#criterion="entropy" tells the decision tree to use
#information gain to decide the best splits.

decision_tree_model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)
#Fit(train) the model on the training data
decision_tree_model.fit(X_train, y_train)

# make predictions on the training data
train_predictions= decision_tree_model.predict(X_train)

#Predictions on testing data
test_predictions = decision_tree_model.predict(X_test)

#Calculate accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

#print out the accuracies
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

#Question explanation

# Entropy measures the amount of uncertainty or impurity in a dataset.
# In a decision tree, entropy tells us how mixed the classes are in a node.

# If all samples in a node belong to one class, the entropy is 0,
# which means the node is perfectly pure.

# If the samples are more mixed between classes, the entropy is higher.

# The decision tree chooses splits that reduce entropy the most.
# This reduction in entropy is called information gain.

# The results suggest some overfitting rather than perfect generalization.
# The training accuracy is 1.0, while the test accuracy is about 0.9123.
# This means the model fits the training data extremely well, but its
# performance drops on unseen test data.
# That pattern suggests the tree may have memorized the training set too
# closely, which is a common problem with unpruned decision trees.

