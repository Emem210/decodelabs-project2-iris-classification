"""
DecodeLabs - Project 2: Data Classification Using AI
IPO Architecture: Input -> Process -> Output

Goal: Build a basic classification model (KNN) on the Iris dataset.
Pipeline: Load -> Scale -> Split -> Train -> Predict -> Evaluate
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

# ─────────────────────────────────────────────
#  PHASE 1 - INPUT: Load & Understand the Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("  PROJECT 2: DATA CLASSIFICATION USING AI (KNN)")
print("  Dataset: Iris Benchmark")
print("=" * 60)

iris = load_iris()
X = iris.data            # features: sepal length/width, petal length/width
y = iris.target          # labels: 0=setosa, 1=versicolor, 2=virginica
feature_names = iris.feature_names
target_names = iris.target_names

print(f"\nSamples       : {X.shape[0]}")
print(f"Features      : {X.shape[1]} -> {feature_names}")
print(f"Classes       : {len(target_names)} -> {list(target_names)}")
print(f"Class balance : {np.bincount(y)}  (perfectly balanced: 50/50/50)")

# ─────────────────────────────────────────────
#  PHASE 2 - PROCESS (a): Feature Scaling
#  "The Gatekeeper Rule" - normalize so no feature dominates
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\n[Scaling] Mean after scaling : {X_scaled.mean(axis=0).round(2)}")
print(f"[Scaling] Std  after scaling : {X_scaled.std(axis=0).round(2)}")

# ─────────────────────────────────────────────
#  PHASE 2 - PROCESS (b): Train/Test Split
#  Shuffle first to remove order bias, 80/20 split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    shuffle=True,
    random_state=42,
    stratify=y,           # keep class balance in both splits
)
print(f"\n[Split] Training samples : {len(X_train)}")
print(f"[Split] Testing samples  : {len(X_test)}")

# ─────────────────────────────────────────────
#  PHASE 2 - PROCESS (c): KNN Algorithm
#  Instantiate -> Fit -> Predict
# ─────────────────────────────────────────────
K = 5
model = KNeighborsClassifier(n_neighbors=K)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"\n[Model] Algorithm    : K-Nearest Neighbors (k={K})")
print(f"[Model] Status       : Trained successfully")

# ─────────────────────────────────────────────
#  PHASE 3 - OUTPUT: Validation
# ─────────────────────────────────────────────
acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")
cm = confusion_matrix(y_test, predictions)

print("\n" + "=" * 60)
print("  OUTPUT VALIDATION")
print("=" * 60)
print(f"\nAccuracy : {acc * 100:.2f}%")
print(f"F1 Score : {f1:.4f}  (weighted average across classes)")

print("\nConfusion Matrix:")
print("                 Predicted")
print("              " + "  ".join(f"{n[:6]:>8}" for n in target_names))
for i, row in enumerate(cm):
    print(f"Actual {target_names[i][:6]:>8} " + "  ".join(f"{v:>8}" for v in row))

print("\nFull Classification Report:")
print(classification_report(y_test, predictions, target_names=target_names))

# ─────────────────────────────────────────────
#  BONUS: Predict a brand-new, unseen flower
# ─────────────────────────────────────────────
print("=" * 60)
print("  BONUS: Predicting a new, unseen sample")
print("=" * 60)
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # looks like a Setosa
new_sample_scaled = scaler.transform(new_sample)
new_pred = model.predict(new_sample_scaled)[0]
new_proba = model.predict_proba(new_sample_scaled)[0]

print(f"\nInput  : {new_sample[0]} (sepal_len, sepal_wid, petal_len, petal_wid)")
print(f"Predicted class : {target_names[new_pred]}")
print(f"Confidence      : {dict(zip(target_names, new_proba.round(2)))}")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE.")
print("=" * 60)
