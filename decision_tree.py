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



# LOAD DATASET
iris = load_iris()

# Create DataFrame
X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

print("----- DATASET -----")
print(X.head())

# SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# CREATE DECISION TREE MODEL
model = DecisionTreeClassifier(
    random_state=42
)

# TRAIN MODEL
model.fit(X_train, y_train)

# MAKE PREDICTIONS
y_pred = model.predict(X_test)

# MODEL ACCURACY
accuracy = accuracy_score(y_test, y_pred)

print("\n----- MODEL ACCURACY -----")
print("Accuracy:", accuracy)


# CLASSIFICATION REPORT

print("\n----- CLASSIFICATION REPORT -----")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# CONFUSION MATRIX
cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n----- CONFUSION MATRIX -----")
print(cm)

# Display Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot(ax=ax)

plt.title("Confusion Matrix")

# Save Confusion Matrix
plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# DECISION TREE VISUALIZATION

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

# Save Decision Tree Graph
plt.savefig(
    "decision_tree_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()