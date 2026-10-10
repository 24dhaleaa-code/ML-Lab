# LAB-1: Titanic Dataset Data Preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
# Place titanic.csv in the same folder as this script.
df = pd.read_csv("titanic.csv")

# 2. Basic inspection
print("First five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values before preprocessing:")
print(df.isnull().sum())

# 3. Remove duplicate rows
df = df.drop_duplicates()

# 4. Drop columns with excessive missing values / unsuitable raw text
df = df.drop(columns=["Cabin"], errors="ignore")

# 5. Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# 6. Feature engineering
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# 7. Remove unnecessary columns
drop_cols = ["PassengerId", "Name", "Ticket"]
df = df.drop(columns=drop_cols, errors="ignore")

# 8. Separate target and features
X = df.drop("Survived", axis=1)
y = df["Survived"]

# 9. One-hot encode categorical variables
X = pd.get_dummies(X, columns=["Sex", "Embarked"], drop_first=True, dtype=int)

# 10. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 11. Scale continuous numerical features
# Fit the scaler only on training data to avoid test-data leakage.
scale_cols = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
scaler = StandardScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

# 12. Final output
print("\nProcessed feature columns:")
print(X_train.columns.tolist())

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

print("\nProcessed training data sample:")
print(X_train.head())

print("\nMissing values after preprocessing (training set):")
print(X_train.isnull().sum())

# Optional visualisation
missing_after = X_train.isnull().sum()
missing_after.plot(kind="bar", figsize=(10, 4))
plt.title("Missing Values After Preprocessing")
plt.ylabel("Number of Missing Values")
plt.tight_layout()
plt.savefig("Lab_1_Missing_Values_After_Preprocessing.png", dpi=150)
plt.show()
