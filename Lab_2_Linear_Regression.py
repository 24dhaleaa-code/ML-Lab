# LAB-2: Linear Regression for House Price Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

 
df = pd.read_csv("C:\Users\ASUS\OneDrive\Documents\RBU\3rd Year\5th Sem\ML  Lab\Codes\house_prices.csv")

print("First five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

features = ["Area_sqft", "Bedrooms", "Bathrooms", "Age_years", "Distance_km"]
target = "Price"
data = df[features + [target]].dropna()

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel coefficients:")
for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: {coefficient:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

result = pd.DataFrame({"Actual_Price": y_test.values, "Predicted_Price": y_pred})
print("\nActual vs Predicted (first 10 rows):")
print(result.head(10))

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("Lab_2_Actual_vs_Predicted.png", dpi=150)
plt.show()
