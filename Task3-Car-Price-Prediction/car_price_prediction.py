# ============================================================
#  OASIS INFOBYTE -- Data Science Internship | Task 3
#  Car Price Prediction with Machine Learning
#  Author : Shinjini
# ============================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 55)
print("  CAR PRICE PREDICTION - OASIS INFOBYTE TASK 3")
print("=" * 55)

# -- 1. Load Dataset
df = pd.read_csv(r"C:\Users\SHINJINI\Downloads\car\car data.csv")

print("\nShape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# -- 2. Preprocessing
print("\nMissing values:", df.isnull().sum().sum())

# Feature Engineering: Car Age
current_year = 2024
df["Car_Age"] = current_year - df["Year"]

# Encode categorical columns
le = LabelEncoder()
cat_cols = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]
for col in cat_cols:
    df[col + "_Enc"] = le.fit_transform(df[col])

# Features & Target
feature_cols = [
    "Car_Age", "Present_Price", "Driven_kms", "Owner",
    "Fuel_Type_Enc", "Selling_type_Enc", "Transmission_Enc"
]
X = df[feature_cols]
y = df["Selling_Price"]

print(f"\nFeatures used: {feature_cols}")

# -- 3. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# -- 4. Train Multiple Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "preds": preds, "model": model}
    print(f"\n{name}:")
    print(f"   MAE  : {mae:.4f} lakhs")
    print(f"   RMSE : {rmse:.4f} lakhs")
    print(f"   R2   : {r2:.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]["R2"])
best_model = results[best_name]["model"]
best_preds = results[best_name]["preds"]
print(f"\nBest Model: {best_name} (R2 = {results[best_name]['R2']:.4f})")

# -- 5. Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Car Price Prediction - Oasis Infobyte Task 3",
             fontsize=15, fontweight="bold")

# Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, best_preds, alpha=0.6, color="#4C72B0", edgecolors="white")
ax.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()], "r--", linewidth=2, label="Perfect Fit")
ax.set_xlabel("Actual Price (Lakhs)")
ax.set_ylabel("Predicted Price (Lakhs)")
ax.set_title(f"Actual vs Predicted\n({best_name})")
ax.legend()

# Feature Importances
ax = axes[0, 1]
rf_model = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
colors = sns.color_palette("Blues_r", len(feature_cols))
ax.bar(range(len(feature_cols)), importances[indices], color=colors, edgecolor="white")
ax.set_xticks(range(len(feature_cols)))
ax.set_xticklabels([feature_cols[i] for i in indices], rotation=25, ha="right", fontsize=8)
ax.set_title("Feature Importances\n(Random Forest)")
ax.set_ylabel("Importance Score")

# Model Comparison
ax = axes[1, 0]
model_names = list(results.keys())
r2_scores = [results[m]["R2"] for m in model_names]
bar_colors = ["#55A868" if m == best_name else "#4C72B0" for m in model_names]
ax.bar(model_names, r2_scores, color=bar_colors, edgecolor="white")
ax.set_title("Model Comparison - R2 Score")
ax.set_ylabel("R2 Score")
ax.set_ylim(0, 1)
for i, v in enumerate(r2_scores):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

# Selling Price Distribution
ax = axes[1, 1]
ax.hist(df["Selling_Price"], bins=25, color="#DD8452", edgecolor="white", alpha=0.85)
ax.set_title("Distribution of Car Selling Prices")
ax.set_xlabel("Selling Price (Lakhs)")
ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig(r"C:\Users\SHINJINI\Downloads\car_price_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved as car_price_results.png in Downloads folder")

# -- 6. Sample Prediction
print("\nSample Prediction (using best model):")
sample = pd.DataFrame([[3, 8.5, 25000, 0, 1, 0, 1]], columns=feature_cols)
pred_price = best_model.predict(sample)[0]
print(f"   Car Age       : 3 years")
print(f"   Present Price : 8.5 Lakhs")
print(f"   Driven kms    : 25,000")
print(f"   Predicted Selling Price: {pred_price:.2f} Lakhs")

print("\nTask 3 Complete! Push this file and car_price_results.png to your OIBSIP repo.")
