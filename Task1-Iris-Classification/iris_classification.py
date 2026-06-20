# ============================================================
#  OASIS INFOBYTE -- Data Science Internship | Task 1
#  Iris Flower Classification
#  Author : Shinjini
# ============================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -- 1. Load Dataset
df = pd.read_csv(r"C:\Users\SHINJINI\Downloads\iris\Iris.csv")

print("=" * 55)
print("  IRIS FLOWER CLASSIFICATION - OASIS INFOBYTE TASK 1")
print("=" * 55)

print("\nDataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nSpecies Distribution:")
print(df["Species"].value_counts())

# -- 2. Data Preprocessing
df.drop(columns=["Id"], inplace=True)

le = LabelEncoder()
df["Species_Encoded"] = le.fit_transform(df["Species"])

X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species_Encoded"]

# -- 3. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# -- 4. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -- 5. Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy : {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# -- 6. Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Iris Flower Classification - Oasis Infobyte Task 1",
             fontsize=15, fontweight="bold", y=1.01)

# Sepal scatter
ax = axes[0, 0]
for species, color in zip(le.classes_, ["#4C72B0", "#DD8452", "#55A868"]):
    subset = df[df["Species"] == species]
    ax.scatter(subset["SepalLengthCm"], subset["SepalWidthCm"],
               label=species, alpha=0.7, edgecolors="white", color=color)
ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Sepal Width (cm)")
ax.set_title("Sepal: Length vs Width")
ax.legend()

# Petal scatter
ax = axes[0, 1]
for species, color in zip(le.classes_, ["#4C72B0", "#DD8452", "#55A868"]):
    subset = df[df["Species"] == species]
    ax.scatter(subset["PetalLengthCm"], subset["PetalWidthCm"],
               label=species, alpha=0.7, edgecolors="white", color=color)
ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
ax.set_title("Petal: Length vs Width")
ax.legend()

# Confusion Matrix
ax = axes[1, 0]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

# Feature Importances
ax = axes[1, 1]
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]
ax.bar(range(len(feature_names)), importances[indices],
       color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"], edgecolor="white")
ax.set_xticks(range(len(feature_names)))
ax.set_xticklabels([feature_names[i] for i in indices], rotation=15, ha="right")
ax.set_title("Feature Importances")
ax.set_ylabel("Importance Score")

plt.tight_layout()
plt.savefig(r"C:\Users\SHINJINI\Downloads\iris_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved as iris_results.png in Downloads folder")

# -- 7. Sample Prediction
print("\nSample Prediction:")
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
    columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
pred_label = le.inverse_transform(model.predict(sample))[0]
print(f"   Input             : {sample.values[0]}")
print(f"   Predicted Species : {pred_label}")

print("\nTask 1 Complete! Push this file and iris_results.png to your OIBSIP repo.")
