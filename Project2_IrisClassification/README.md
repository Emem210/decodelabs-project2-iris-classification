# Iris Flower Classifier — Data Classification Using AI

**DecodeLabs Industrial Training Kit — Artificial Intelligence Track — Project 2**

A supervised learning pipeline that classifies Iris flowers into one of three species (Setosa, Versicolor, Virginica) using the K-Nearest Neighbors (KNN) algorithm, following the full Input → Process → Output (IPO) model taught in the training material.

## Pipeline (IPO Architecture)

| Phase | Step | Implementation |
|---|---|---|
| **Input** | Load dataset | `sklearn.datasets.load_iris()` — 150 samples, 4 features, 3 balanced classes |
| **Process** | Feature scaling | `StandardScaler` (mean = 0, variance = 1) so no feature dominates |
| **Process** | Train/test split | `train_test_split` — 80/20, shuffled, stratified |
| **Process** | Model training | `KNeighborsClassifier(n_neighbors=5)` — instantiate → fit → predict |
| **Output** | Validation | Accuracy, F1 score (weighted), full confusion matrix, classification report |
| **Bonus** | Live prediction | Classifies one brand-new, unseen flower sample with confidence scores |

## How to run

```bash
pip install scikit-learn numpy
python iris_classifier.py
```

No external dataset file needed — the Iris dataset ships with scikit-learn.

## Example output

```
Accuracy : 93.33%
F1 Score : 0.9327  (weighted average across classes)

Confusion Matrix:
                 Predicted
                setosa    versic    virgin
Actual   setosa       10         0         0
Actual   versic        0        10         0
Actual   virgin        0         2         8
```

## Why F1 score, not just accuracy?

As the training material notes, accuracy alone can be an "accuracy mirage" on imbalanced data. The F1 score (harmonic mean of precision and recall) gives a more honest picture per class, especially valuable when one class is harder to separate (here, Virginica vs. Versicolor overlap slightly in feature space).

## Why scale before splitting?

Iris features are all in centimeters with similar ranges, but in general, unscaled features with larger numeric ranges can dominate distance-based algorithms like KNN. `StandardScaler` keeps every feature on equal footing.

## Skills demonstrated

- Data loading and exploration
- Feature scaling / normalization
- Train/test split methodology (with stratification)
- Supervised learning with KNN
- Model evaluation beyond accuracy (F1, confusion matrix)

## Author

Amin — DecodeLabs AI Track, Batch 2026
