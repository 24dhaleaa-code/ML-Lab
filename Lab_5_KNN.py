# LAB-5: KNN Classification and Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score
)

# PART A: KNN CLASSIFICATION
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k = 5
knn_clf = KNeighborsClassifier(n_neighbors=k)
knn_clf.fit(X_train_scaled, y_train)
y_pred = knn_clf.predict(X_test_scaled)

print("KNN CLASSIFICATION")
print("K =", k)
print("Accuracy =", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# PART B: KNN REGRESSION
X_reg, y_reg = make_regression(
    n_samples=150, n_features=1, noise=15, random_state=42
)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

reg_scaler = StandardScaler()
Xr_train_scaled = reg_scaler.fit_transform(Xr_train)
Xr_test_scaled = reg_scaler.transform(Xr_test)

k_reg = 5
knn_reg = KNeighborsRegressor(n_neighbors=k_reg)
knn_reg.fit(Xr_train_scaled, yr_train)
yr_pred = knn_reg.predict(Xr_test_scaled)

print("\nKNN REGRESSION")
print("K =", k_reg)
print("Mean Squared Error =", mean_squared_error(yr_test, yr_pred))
print("R2 Score =", r2_score(yr_test, yr_pred))

# Regression plot
order = np.argsort(Xr_test[:, 0])
plt.figure(figsize=(8, 5))
plt.scatter(Xr_test[:, 0], yr_test, label="Actual")
plt.plot(Xr_test[order, 0], yr_pred[order], label="Predicted", linewidth=2)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("KNN Regression: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Lab_5_KNN_Regression_Plot.png", dpi=150)
plt.show()
