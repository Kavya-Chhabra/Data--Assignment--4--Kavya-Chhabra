#Q1 — Dataset Exploration and Understanding (5 points)
#A correct understanding of the dataset is essential before applying any machine learning model.
#Using the provided dataset:
#• Construct the feature matrix X and target vector y.
#• Report the shape of X and y.
#• Report the number of samples belonging to each class.
#In Python comments, provide a brief discussion addressing:
#• whether the dataset is balanced or imbalanced, and
#• why class balance is an important consideration for classification models.

#Importing the dataset from sklearn
from sklearn.datasets import load_breast_cancer

# Load the dataset into a variable called "data"
data = load_breast_cancer()

#Load the dataset into a variable called "Data"
data = load_breast_cancer()

#Construct feature matrix X and target vector y

#X contains all the feature values(30 numeric measurements for each tumor)
# these include things like radius, texture, smoothness,etc.
X = data.data

# y contains the target labels(classification)
# 0 = malignant(cancerous)
# 1 = benign (non-cancerous)
y = data.target

#==========================================================
# report the shapes of X and y
#==========================================================

#Shape tells us the dimensions of the dataset
# For X: (number_of_samples, number_of_features)
# For y: (number_of_samples,)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# count the number of samples in each class
import numpy as np

#count how many samples belong to each class
class_counts = np.bincount(y)

print("Number of malignant samples (class 0):", class_counts[0])
print("Number of benign samples (class 1):", class_counts[1])


#========================================================
#interpretation
#=========================================================

#the dataset contains 569 total samples and 30 features.
# Each sample represents measurements of a tumor taken from
# digitized images of a breast mass.

# Class distribution:
# Malignant (0) ≈ 212 samples
# Benign (1) ≈ 357 samples

# This means the dataset is somewhat imbalanced because
# there are noticeably more benign cases than malignant ones.

# Class balance is important for classification models because
# if one class dominates the dataset, the model may learn to
# predict that class more often and ignore the minority class.

# In medical datasets, this is especially important because
# the minority class (malignant tumors) is actually the most
# important to detect correctly. A model that predicts benign
# too often could miss real cancer cases, which would be a
# serious medical risk.



