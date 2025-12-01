"""
models.py
---------
Reusable training functions for imbalanced datasets:
- KNN
- Decision Tree
- SVM
- Gradient Boosting
- AdaBoost
- XGBoost

All tree-based models handle class imbalance via `class_weight` or `scale_pos_weight`.
"""

from xgboost import XGBClassifier
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


# ---------------------------------------------------------
# TRAINING FUNCTIONS
# ---------------------------------------------------------

def train_knn(X_train, y_train, n_neighbors=7):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, random_state=42):
    # Use class_weight to handle imbalance
    model = DecisionTreeClassifier(class_weight="balanced", random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, kernel="rbf", C=2.0, random_state=42):
    # Option: weight classes manually
    model = SVC(kernel=kernel, C=C, probability=True, class_weight="balanced", random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, n_estimators=400, learning_rate=0.1, random_state=42):
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
    # GradientBoostingClassifier does not support class_weight, so we rely on SMOTE
    model.fit(X_train, y_train)
    return model


def train_adaboost(X_train, y_train, n_estimators=200, learning_rate=0.05, random_state=42):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier

    base_est = DecisionTreeClassifier(max_depth=3, class_weight="balanced")

    model = AdaBoostClassifier(
        estimator=base_est,          # ← FIX HERE
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )

    model.fit(X_train, y_train)
    return model



def train_xgboost(X_train, y_train, n_estimators=400, learning_rate=0.05, random_state=42):
    # Handle imbalance with scale_pos_weight
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=pos_weight * 1.2,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        # use_label_encoder=False
    )
    model.fit(X_train, y_train)
    return model

def train_balanced_rf(X_train, y_train, random_state=42):
    model = BalancedRandomForestClassifier(
        n_estimators=400,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------
# PREDICTION & EVALUATION HELPERS
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    print(f"===== {model_name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("========================\n")
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }


def get_probabilities(model, X_test):
    return model.predict_proba(X_test)[:, 1]


# Threshold Tuning funciton
def evaluate_thresholds(model, X_test, y_test, model_name="Model"):
    """Evaluate performance across thresholds 0.1–0.9."""
    if not hasattr(model, "predict_proba"):
        print(f"{model_name} does not support predict_proba. Skipping threshold tuning.")
        return None

    probs = model.predict_proba(X_test)[:, 1]  # positive class probability

    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []

    for t in thresholds:
        preds = (probs >= t).astype(int)

        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f = f1_score(y_test, preds, zero_division=0)

        rows.append((t, p, r, f))

    return pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1"])
