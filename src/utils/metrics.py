"""
Evaluation metrics for protein binding site prediction.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, precision_score,
    f1_score, matthews_corrcoef, average_precision_score
)
from typing import List, Tuple


def metrics(y_true: List, y_pred: List, y_scores: List) -> Tuple[float, ...]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        y_scores: Prediction scores (probabilities)
    
    Returns:
        Tuple of (ACC, AUC, Recall, Precision, F1, MCC, PRC)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    # Basic metrics
    ACC = accuracy_score(y_true, y_pred)
    
    # Handle edge cases for AUC
    try:
        if len(np.unique(y_true)) < 2:
            AUC = 0.5  # Default for single class
        else:
            AUC = roc_auc_score(y_true, y_scores)
    except ValueError:
        AUC = 0.5  # Default for edge cases
    
    # Classification metrics
    Recall = recall_score(y_true, y_pred, zero_division=0)
    Precision = precision_score(y_true, y_pred, zero_division=0)
    F1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Matthews correlation coefficient
    try:
        MCC = matthews_corrcoef(y_true, y_pred)
    except ValueError:
        MCC = 0.0
    
    # Precision-Recall curve AUC
    try:
        if len(np.unique(y_true)) < 2:
            PRC = 0.0  # Default for single class
        else:
            PRC = average_precision_score(y_true, y_scores)
    except ValueError:
        PRC = 0.0
    
    return ACC, AUC, Recall, Precision, F1, MCC, PRC


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate binary classification metrics."""
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'sensitivity': TP / (TP + FN) if (TP + FN) > 0 else 0,
        'specificity': TN / (TN + FP) if (TN + FP) > 0 else 0,
        'precision': TP / (TP + FP) if (TP + FP) > 0 else 0,
        'recall': TP / (TP + FN) if (TP + FN) > 0 else 0
    }


def calculate_class_weights(labels: List[int]) -> dict:
    """Calculate class weights for imbalanced datasets."""
    labels = np.array(labels)
    unique_classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    weights = {}
    for cls, count in zip(unique_classes, counts):
        weights[cls] = total / (len(unique_classes) * count)
    
    return weights