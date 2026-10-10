# LAB-7: Naive Bayes Classification

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Create and train the Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("NAIVE BAYES CLASSIFICATION")
print("--------------------------------")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))
print("Algorithm       : Gaussian Naive Bayes")
print(f"Test Accuracy   : {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Plot confusion matrix for the lab output screenshot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Gaussian Naive Bayes - Confusion Matrix")
plt.tight_layout()
plt.savefig("Lab_7_Naive_Bayes_Confusion_Matrix.png", dpi=150)
plt.show()
