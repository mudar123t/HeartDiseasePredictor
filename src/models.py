"""
models.py
---------
Reusable training functions for:
- KNN
- Decision Tree
- SVM

Also includes:
- Unified prediction function
- Model metrics function (optional)
- Functions designed to keep Jupyter notebooks clean

Author: (Your Name)
"""

from xgboost import XGBClassifier

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)


# ---------------------------------------------------------
# TRAINING FUNCTIONS
# ---------------------------------------------------------

def train_knn(X_train, y_train, n_neighbors=5):
    """
    Train and return a K-Nearest Neighbors classifier.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, random_state=42):
    """
    Train and return a Decision Tree classifier.
    """
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, kernel="rbf", random_state=42):
    """
    Train and return an SVM classifier with probability enabled.
    """
    model = SVC(kernel=kernel, probability=True, random_state=random_state)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------
# PREDICTION & EVALUATION HELPERS
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Prints standard metrics for model evaluation.
    """
    y_pred = model.predict(X_test)

    print(f"===== {model_name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("========================\n")

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }


def get_probabilities(model, X_test):
    """
    Returns predicted probabilities for ROC/AUC curves.
    Works only if model supports predict_proba().
    """
    return model.predict_proba(X_test)[:, 1]



def train_gradient_boosting(X_train, y_train, random_state=42):
    """
    Train and return a Gradient Boosting Classifier.
    """
    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_adaboost(X_train, y_train, random_state=42, n_estimators=100):
    """
    Train and return an AdaBoost Classifier.
    """
    model = AdaBoostClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state=42):
    """
    Train and return an XGBoost Classifier.
    """
    xgb_model = XGBClassifier(
    random_state=random_state,
    eval_metric="logloss",
    scale_pos_weight=pos_weight
)

    model.fit(X_train, y_train)
    return model


