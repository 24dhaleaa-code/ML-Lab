# LAB-4: Decision Tree Classification on the Iris Dataset

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("----- DATASET -----")
print(X.head())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\n----- MODEL ACCURACY -----")
print("Accuracy:", accuracy)

# Classification report
print("\n----- CLASSIFICATION REPORT -----")
print(classification_report(
    y_test, y_pred, target_names=iris.target_names
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\n----- CONFUSION MATRIX -----")
print(cm)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)
disp.plot(ax=ax, values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# Decision tree visualization
plt.figure(figsize=(20, 12))
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Classification")
plt.tight_layout()
plt.savefig("decision_tree_graph.png", dpi=300, bbox_inches="tight")
plt.show()
