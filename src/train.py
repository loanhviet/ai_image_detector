"""
Module training & evaluation cho bài toán phân loại Real vs Fake.

Hỗ trợ:
  - Random Forest
  - XGBoost
  - Evaluation metrics: accuracy, precision, recall, F1, ROC-AUC
  - Feature importance visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score
)
import joblib


def train_random_forest(X_train, y_train, cv=5):
    """
    Train Random Forest với GridSearchCV.
    
    Returns: (best_model, cv_results)
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid = GridSearchCV(
        rf, param_grid, cv=cv, scoring='accuracy',
        n_jobs=-1, verbose=1, refit=True
    )
    grid.fit(X_train, y_train)
    
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    
    return grid.best_estimator_, grid


def train_xgboost(X_train, y_train, cv=5):
    """
    Train XGBoost với GridSearchCV.
    
    Returns: (best_model, cv_results)
    """
    from xgboost import XGBClassifier
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
    }
    
    xgb = XGBClassifier(
        random_state=42, n_jobs=-1,
        eval_metric='logloss', use_label_encoder=False
    )
    grid = GridSearchCV(
        xgb, param_grid, cv=cv, scoring='accuracy',
        n_jobs=-1, verbose=1, refit=True
    )
    grid.fit(X_train, y_train)
    
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    
    return grid.best_estimator_, grid


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Đánh giá model: classification report, confusion matrix, ROC-AUC.
    
    Returns: dict metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n{'=' * 50}")
    print(f"  {model_name} — Evaluation")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    
    return {
        "accuracy": acc,
        "roc_auc": auc,
        "y_pred": y_pred,
        "y_proba": y_proba
    }


def plot_confusion_matrix(y_test, y_pred, model_name="Model", ax=None):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — Confusion Matrix")


def plot_roc_curve(y_test, y_proba, model_name="Model", ax=None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)


def plot_feature_importance(model, feature_names, top_n=20, ax=None):
    """Plot top N feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(range(len(indices)), importances[indices], color='steelblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Features")
    ax.grid(axis='x', alpha=0.3)
