# 📊 ML Workflow: Detecting Data Leakage and Fixing It

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# =========================
# Task 1 — Flawed Approach (Data Leakage)
# =========================

# Generating synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# ❌ WRONG: Scaling entire dataset before split (causes leakage)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting AFTER scaling
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("Task 1 — Flawed Approach")
print("Train Accuracy:", round(train_acc, 3))
print("Test Accuracy:", round(test_acc, 3))

# Explanation:
# Here scaling was done before splitting the data.
# This means information from the test set influenced the scaler.
# This is called DATA LEAKAGE and leads to overly optimistic results.


# =========================
# Task 2 — Correct Approach (Pipeline + CV)
# =========================

# Creating a pipeline to avoid leakage
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# Cross-validation (proper way)
scores = cross_val_score(pipeline, X, y, cv=5)

print("\nTask 2 — Pipeline Approach")
print("Cross-validation scores:", scores)
print("Mean accuracy:", round(np.mean(scores), 3))
print("Std deviation:", round(np.std(scores), 3))

# Explanation:
# Scaling happens inside each fold, so test data is never leaked.
# This gives a more reliable estimate of performance.


# =========================
# Task 3 — Decision Tree Depth Experiment
# =========================

# Splitting data (correct way: split BEFORE preprocessing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

depths = [1, 5, 20]

print("\nTask 3 — Decision Tree Comparison")

results = []

for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    tree.fit(X_train, y_train)

    train_score = accuracy_score(y_train, tree.predict(X_train))
    test_score = accuracy_score(y_test, tree.predict(X_test))

    results.append((d, train_score, test_score))

    print(f"Depth={d} | Train={round(train_score,3)} | Test={round(test_score,3)}")

# Explanation:
# depth=1 → underfitting (low accuracy)
# depth=20 → overfitting (high train, lower test)
# depth=5 → balanced performance (best generalization)


# Optional: Convert results to DataFrame (nice for readability)
df_results = pd.DataFrame(results, columns=["max_depth", "train_acc", "test_acc"])
print("\nSummary Table:")
print(df_results)