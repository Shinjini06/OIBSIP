# ============================================================
#  OASIS INFOBYTE -- Data Science Internship | Task 5
#  Email Spam Detection with Machine Learning
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 55)
print("  EMAIL SPAM DETECTION - OASIS INFOBYTE TASK 5")
print("=" * 55)

# -- 1. Load Dataset
df = pd.read_csv(r"C:\Users\SHINJINI\Downloads\spam\spam.csv",
                 encoding="latin-1", usecols=[0, 1])
df.columns = ["Label", "Message"]

print("\nShape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nLabel Distribution:")
print(df["Label"].value_counts())
print("\nMissing values:", df.isnull().sum().sum())

# -- 2. Preprocessing
df["Label_Enc"] = df["Label"].map({"ham": 0, "spam": 1})

X = df["Message"]
y = df["Label_Enc"]

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_tfidf = tfidf.fit_transform(X)

# -- 3. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

# -- 4. Train Multiple Models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    results[name] = {"accuracy": acc, "preds": preds, "model": model}
    print(f"\n{name}:")
    print(f"   Accuracy : {acc * 100:.2f}%")
    print(classification_report(y_test, preds, target_names=["Ham", "Spam"]))

best_name  = max(results, key=lambda k: results[k]["accuracy"])
best_preds = results[best_name]["preds"]
best_model = results[best_name]["model"]
print(f"\nBest Model: {best_name} (Accuracy = {results[best_name]['accuracy']*100:.2f}%)")

# -- 5. Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Email Spam Detection - Oasis Infobyte Task 5",
             fontsize=15, fontweight="bold")

# Spam vs Ham Distribution
ax = axes[0, 0]
counts = df["Label"].value_counts()
ax.bar(counts.index, counts.values, color=["#55A868", "#DD8452"], edgecolor="white", width=0.4)
ax.set_title("Spam vs Ham Distribution")
ax.set_ylabel("Count")
for i, v in enumerate(counts.values):
    ax.text(i, v + 10, str(v), ha="center", fontweight="bold")

# Confusion Matrix (best model)
ax = axes[0, 1]
cm = confusion_matrix(y_test, best_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"], ax=ax)
ax.set_title(f"Confusion Matrix\n({best_name})")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

# Model Accuracy Comparison
ax = axes[1, 0]
model_names = list(results.keys())
accuracies  = [results[m]["accuracy"] * 100 for m in model_names]
bar_colors  = ["#55A868" if m == best_name else "#4C72B0" for m in model_names]
ax.bar(model_names, accuracies, color=bar_colors, edgecolor="white")
ax.set_title("Model Accuracy Comparison")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(80, 100)
for i, v in enumerate(accuracies):
    ax.text(i, v + 0.1, f"{v:.2f}%", ha="center", fontsize=10, fontweight="bold")

# Message Length Distribution
ax = axes[1, 1]
df["Message_Length"] = df["Message"].apply(len)
df[df["Label"] == "ham"]["Message_Length"].hist(
    bins=40, ax=ax, alpha=0.6, color="#55A868", label="Ham")
df[df["Label"] == "spam"]["Message_Length"].hist(
    bins=40, ax=ax, alpha=0.6, color="#DD8452", label="Spam")
ax.set_title("Message Length Distribution")
ax.set_xlabel("Message Length")
ax.set_ylabel("Count")
ax.legend()

plt.tight_layout()
plt.savefig(r"C:\Users\SHINJINI\Downloads\spam_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved as spam_results.png in Downloads folder")

# -- 6. Sample Prediction
print("\nSample Predictions:")
samples = [
    "Congratulations! You have won a free iPhone. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?"
]
for msg in samples:
    vec  = tfidf.transform([msg])
    pred = best_model.predict(vec)[0]
    label = "SPAM" if pred == 1 else "HAM"
    print(f"   '{msg[:50]}...' --> {label}")

print("\nTask 5 Complete! Push this file and spam_results.png to your OIBSIP repo.")
