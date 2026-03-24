# Data Assignment- By: Kavya Chhabra

## Questions Completed

### Q1 – Dataset Exploration and Understanding
- Loaded the Breast Cancer dataset
- Constructed the feature matrix `X` and target vector `y`
- Reported dataset shape and class counts
- Discussed class balance and why it matters

### Q2 – Decision Tree Model Using Entropy
- Used an 80/20 train-test split with stratification
- Trained a Decision Tree classifier using `entropy`
- Reported training and test accuracy
- Explained entropy and discussed overfitting vs generalization

### Q3 – Controlling Tree Complexity and Interpretability
- Added a constraint to the Decision Tree model using `max_depth`
- Reported training and test accuracy
- Displayed the top 5 most important features
- Discussed overfitting and interpretability

### Q4 – Neural Network on the Breast Cancer Dataset
- Standardized the input features
- Built and trained a neural network with at least one hidden layer
- Used a sigmoid output layer for binary classification
- Reported training and test accuracy
- Explained feature scaling and epochs

### Q5 – Model Evaluation and Comparison
- Computed confusion matrices for:
  - constrained Decision Tree
  - Neural Network
- Compared both models
- Discussed one advantage and one limitation of each model

### Q6 – CNN on Fashion MNIST
- Loaded the Fashion MNIST dataset
- Normalized pixel values to the range `[0,1]`
- Reshaped images to include the channel dimension
- Built and trained a CNN with:
  - Conv2D layer
  - MaxPooling2D layer
  - Dense output layer
- Trained for at least 15 epochs
- Reported test accuracy
- Explained why CNNs are preferred for image data

### Q7 – CNN Error Analysis and Misclassification Study
- Generated predictions on the test set
- Computed the confusion matrix
- Displayed misclassified images
- Showed the true and predicted labels
- Discussed a pattern in the errors and one improvement method

## Libraries Used

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow / keras`

## Notes

- The Breast Cancer dataset was loaded from `sklearn.datasets`.
- The Fashion MNIST dataset was loaded from `tensorflow.keras.datasets`.
- Some TensorFlow warnings may appear during execution, but the code still runs correctly.
- Fashion MNIST images are low-resolution grayscale images, so misclassified examples may appear blurry or pixelated.

