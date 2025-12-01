"""
feature_selection.py
---------------------
This module provides reusable feature selection functions for:
- Chi-Square Test
- ANOVA F-test
- Mutual Information
- Recursive Feature Elimination (RFE)
- Decision Tree Feature Importance

These functions are used in notebooks for visual evaluation and
support the modeling pipeline.
"""

import pandas as pd
import numpy as np

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------
# 1. CHI-SQUARE TEST
# ---------------------------------------------------------

def chi_square_test(X, y):
    """
    Computes Chi-Square scores for categorical features.
    Works on non-negative data; uses MinMaxScaler before chi2.

    Returns:
        pd.DataFrame: ranked Chi-Square scores and p-values.
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    chi_scores, p_values = chi2(X_scaled, y)

    results = pd.DataFrame({
        "Feature": X.columns,
        "Chi2 Score": chi_scores,
        "p-value": p_values
    }).sort_values("Chi2 Score", ascending=False)

    return results


# ---------------------------------------------------------
# 2. ANOVA F-TEST (Numerical vs Target)
# ---------------------------------------------------------

def anova_test(X, y):
    """
    Computes ANOVA F-test scores for numerical features.

    Returns:
        pd.DataFrame: ranked ANOVA scores and p-values.
    """
    scores, p_values = f_classif(X, y)

    results = pd.DataFrame({
        "Feature": X.columns,
        "ANOVA Score": scores,
        "p-value": p_values
    }).sort_values("ANOVA Score", ascending=False)

    return results


# ---------------------------------------------------------
# 3. MUTUAL INFORMATION (captures nonlinear relations)
# ---------------------------------------------------------

def mutual_information_test(X, y):
    """
    Computes Mutual Information scores for all features.

    Returns:
        pd.DataFrame: ranked MI scores.
    """
    mi_scores = mutual_info_classif(X, y, random_state=42)

    results = pd.DataFrame({
        "Feature": X.columns,
        "MI Score": mi_scores
    }).sort_values("MI Score", ascending=False)

    return results


# ---------------------------------------------------------
# 4. RFE (Recursive Feature Elimination)
# ---------------------------------------------------------

def rfe_selection(X, y, n_features=10):
    """
    Runs Recursive Feature Elimination using Logistic Regression.

    Args:
        n_features (int): number of features to keep.

    Returns:
        pd.DataFrame: RFE ranking + selection status.
    """
    model = LogisticRegression(max_iter=2000)
    rfe = RFE(model, n_features_to_select=n_features)
    rfe.fit(X, y)

    results = pd.DataFrame({
        "Feature": X.columns,
        "Selected": rfe.support_,
        "Rank": rfe.ranking_
    }).sort_values("Rank")

    return results


# ---------------------------------------------------------
# 5. Decision Tree Feature Importance
# ---------------------------------------------------------

def tree_feature_importance(X, y):
    """
    Computes feature importance using a Decision Tree classifier.

    Returns:
        pd.DataFrame: importance scores sorted descending.
    """
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": tree.feature_importances_
    }).sort_values("Importance", ascending=False)

    return importance


# ---------------------------------------------------------
# 6. Combined Summary (Optional helper)
# ---------------------------------------------------------

def summarize_all_methods(X, y, n_features=10):
    """
    Runs all feature selection methods and returns a dictionary
    containing their results.

    Returns:
        dict of DataFrames
    """
    return {
        "Chi-Square": chi_square_test(X, y),
        "ANOVA": anova_test(X, y),
        "Mutual Information": mutual_information_test(X, y),
        "RFE": rfe_selection(X, y, n_features=n_features),
        "Decision Tree Importance": tree_feature_importance(X, y)
    }



def vote_feature_selection(X, y, top_n=10, min_votes=2):
    chi = chi_square_test(X, y).head(top_n)["Feature"].tolist()
    anova = anova_test(X, y).head(top_n)["Feature"].tolist()
    mi = mutual_information_test(X, y).head(top_n)["Feature"].tolist()
    rfe = rfe_selection(X, y, n_features=top_n)
    rfe = rfe[rfe["Selected"] == True]["Feature"].tolist()
    tree = tree_feature_importance(X, y).head(top_n)["Feature"].tolist()

    all_features = chi + anova + mi + rfe + tree
    votes = pd.Series(all_features).value_counts()

    selected = votes[votes >= min_votes].index.tolist()
    return selected, votes
