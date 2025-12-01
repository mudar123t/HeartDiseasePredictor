"""
models.py
---------
Reusable training functions for various ML models:

- KNN
- Decision Tree
- SVM
- Gradient Boosting
- AdaBoost
- XGBoost
- Balanced Random Forest

Also includes:
- model evaluation
- threshold tuning
"""

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


# ---------------------------------------------------------
# MODEL TRAINERS
# ---------------------------------------------------------

def train_knn(X_train, y_train, n_neighbors=7):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, random_state=42):
    model = DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, kernel="rbf", C=2.0, random_state=42):
    model = SVC(
        kernel=kernel,
        C=C,
        probability=True,
        class_weight="balanced",
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42):
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_adaboost(X_train, y_train, n_estimators=200, learning_rate=0.05, random_state=42):
    base_tree = DecisionTreeClassifier(
        max_depth=4,
        class_weight="balanced",
        min_samples_leaf=5
    )

    model = AdaBoostClassifier(
        estimator=base_tree,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )

    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, n_estimators=400, learning_rate=0.05, random_state=42):
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        # n_estimators=n_estimators,
        # learning_rate=learning_rate,
        # max_depth=3,
        # scale_pos_weight=pos_weight * 1.2,
        # subsample=0.8,
        # colsample_bytree=0.8,
        # random_state=random_state,
        # eval_metric="logloss",
        # reg_alpha=0.5,
        # reg_lambda=1.0,
        max_depth=3,
        n_estimators=150,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"

    )
    # use a small validation split from training (do not use test)
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model

# ---------------------------------------------------------
# EVALUATION HELPERS
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)

    print(f"\n===== {model_name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("========================\n")

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }


def evaluate_thresholds(model, X_test, y_test, model_name="Model"):
    """Evaluate performance at thresholds from 0.05 to 0.95."""
    if not hasattr(model, "predict_proba"):
        print(f"{model_name} does not support predict_proba. Skipping threshold tuning.")
        return None

    probs = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 19)
    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)

        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f = f1_score(y_test, preds, zero_division=0)

        results.append((t, p, r, f))

    return pd.DataFrame(results, columns=["threshold", "precision", "recall", "f1"])
