# ============================================================
#  OASIS INFOBYTE -- Data Science Internship | Task 4
#  Sales Prediction with Machine Learning
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 55)
print("  SALES PREDICTION - OASIS INFOBYTE TASK 4")
print("=" * 55)

# -- 1. Load Dataset
df = pd.read_csv(r"C:\Users\SHINJINI\Downloads\advertise\Advertising.csv")
df = df.drop(columns=[df.columns[0]])  # Drop index column

print("\nShape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing values:", df.isnull().sum().sum())

# -- 2. Features & Target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# -- 3. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples : {len(X_train)}")
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
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "preds": preds, "model": model}
    print(f"\n{name}:")
    print(f"   MAE  : {mae:.4f}")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   R2   : {r2:.4f}")

best_name  = max(results, key=lambda k: results[k]["R2"])
best_preds = results[best_name]["preds"]
best_model = results[best_name]["model"]
print(f"\nBest Model: {best_name} (R2 = {results[best_name]['R2']:.4f})")

# -- 5. Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Sales Prediction - Oasis Infobyte Task 4",
             fontsize=15, fontweight="bold")

# Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, best_preds, alpha=0.7, color="#4C72B0", edgecolors="white")
ax.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()], "r--", linewidth=2, label="Perfect Fit")
ax.set_xlabel("Actual Sales")
ax.set_ylabel("Predicted Sales")
ax.set_title(f"Actual vs Predicted\n({best_name})")
ax.legend()

# Correlation Heatmap
ax = axes[0, 1]
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax, linewidths=0.5)
ax.set_title("Feature Correlation Heatmap")

# Model Comparison
ax = axes[1, 0]
model_names = list(results.keys())
r2_scores   = [results[m]["R2"] for m in model_names]
bar_colors  = ["#55A868" if m == best_name else "#4C72B0" for m in model_names]
ax.bar(model_names, r2_scores, color=bar_colors, edgecolor="white")
ax.set_title("Model Comparison - R2 Score")
ax.set_ylabel("R2 Score")
ax.set_ylim(0, 1)
for i, v in enumerate(r2_scores):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

# Ad Spend vs Sales scatter
ax = axes[1, 1]
ax.scatter(df["TV"],        df["Sales"], alpha=0.5, label="TV",        color="#4C72B0")
ax.scatter(df["Radio"],     df["Sales"], alpha=0.5, label="Radio",     color="#DD8452")
ax.scatter(df["Newspaper"], df["Sales"], alpha=0.5, label="Newspaper", color="#55A868")
ax.set_xlabel("Ad Spend")
ax.set_ylabel("Sales")
ax.set_title("Ad Spend vs Sales")
ax.legend()

plt.tight_layout()
plt.savefig(r"C:\Users\SHINJINI\Downloads\sales_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved as sales_results.png in Downloads folder")

# -- 6. Sample Prediction
print("\nSample Prediction:")
sample = pd.DataFrame([[200.0, 30.0, 50.0]], columns=["TV", "Radio", "Newspaper"])
pred_sales = best_model.predict(sample)[0]
print(f"   TV=200, Radio=30, Newspaper=50")
print(f"   Predicted Sales: {pred_sales:.2f}")

print("\nTask 4 Complete! Push this file and sales_results.png to your OIBSIP repo.")
