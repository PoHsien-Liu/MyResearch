import os
import json
from datetime import datetime
from sklearn.metrics import (
    matthews_corrcoef, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix
)
import numpy as np

def calculate_metrics(preds, labels):
    """
    Args:
        preds: List[str], like ["Positive", "Negative", "Unknown", ...]
        labels: List[str], like ["Positive", "Negative", ...]
    
    Returns:
        dict: {
            'accuracy': float,
            'mcc': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'confusion_matrix': List[List[int]]
        }
    """
    label_map = {'Positive': 1, 'Negative': 0}

    preds_mapped = []
    labels_mapped = []

    for p, l in zip(preds, labels):
        true_label = label_map.get(l, -1)
        pred_val = label_map.get(p, -1)

        if pred_val == -1:
            pred_val = 1 - true_label

        preds_mapped.append(pred_val)
        labels_mapped.append(true_label)

    if len(labels_mapped) == 0:
        acc = 0.0
        mcc = 0.0
        prec = 0.0
        rec = 0.0
        f1 = 0.0
        conf_matrix = [[0, 0], [0, 0]]
    else:
        acc = accuracy_score(labels_mapped, preds_mapped)
        mcc = matthews_corrcoef(labels_mapped, preds_mapped)
        prec = precision_score(labels_mapped, preds_mapped, zero_division=0)
        rec = recall_score(labels_mapped, preds_mapped, zero_division=0)
        f1 = f1_score(labels_mapped, preds_mapped, zero_division=0)
        conf_matrix = confusion_matrix(labels_mapped, preds_mapped).tolist()

    return {
        'accuracy': acc,
        'mcc': mcc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'total': len(labels),
        'valid': len(labels_mapped),
        'invalid': len(labels) - len(labels_mapped)
    }

def save_metrics(metrics_result, model_name, save_dir="results", dataset_name=None):
    """
    Save evaluation results into a JSON file.
    Args:
        metrics_result: dict, output of calculate_metrics()
        model_name: str, model name
        save_dir: str, directory to save
        dataset_name: str, name of the dataset
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f"{model_name}_eval_{timestamp}.json")

    result_to_save = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "timestamp": timestamp,
        "total_samples": metrics_result['total'],
        "valid_samples": metrics_result['valid'],
        "invalid_samples": metrics_result['invalid'],
        "accuracy": metrics_result['accuracy'],
        "mcc": metrics_result['mcc'],
        "precision": metrics_result['precision'],
        "recall": metrics_result['recall'],
        "f1_score": metrics_result['f1'],
        "confusion_matrix": {
            "labels": ["Negative", "Positive"],
            "matrix": metrics_result['confusion_matrix']
        }
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(result_to_save, f, indent=4)

    return save_path