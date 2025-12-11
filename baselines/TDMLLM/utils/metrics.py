import os
import json
from datetime import datetime

import numpy as np

def calculate_metrics(preds, labels):
    """
    Args:
        preds: List[str], like ["UP", "DOWN", "Unknown", ...]
        labels: List[str], like ["UP", "DOWN", ...]
    
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
    label_map = {'UP': 1, 'POSITIVE': 1, 'DOWN': 0, 'NEGATIVE': 0}
    unknown_predictions = 0

    preds_mapped = []
    labels_mapped = []

    for p, l in zip(preds, labels):
        true_label = label_map.get(str(l).upper(), -1)
        pred_val = label_map.get(str(p).upper(), -1)

        if true_label == -1:
            continue

        if pred_val == -1:
            unknown_predictions += 1
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
        labels_arr = np.array(labels_mapped)
        preds_arr = np.array(preds_mapped)

        tp = int(np.sum((preds_arr == 1) & (labels_arr == 1)))
        tn = int(np.sum((preds_arr == 0) & (labels_arr == 0)))
        fp = int(np.sum((preds_arr == 1) & (labels_arr == 0)))
        fn = int(np.sum((preds_arr == 0) & (labels_arr == 1)))

        acc = (tp + tn) / len(labels_arr)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
        conf_matrix = [[tn, fp], [fn, tp]]

    return {
        'accuracy': acc,
        'mcc': mcc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'total': len(labels),
        'valid': len(labels_mapped),
        'invalid': len(labels) - len(labels_mapped),
        'unknown_predictions': unknown_predictions,
    }

def save_metrics(metrics_result, model_name, results_dir, dataset_name=None, experiment_duration=None):
    """
    Save evaluation results into a JSON file.
    Args:
        metrics_result: dict, output of calculate_metrics()
        model_name: str, model name
        results_dir: str, directory to save results
        dataset_name: str, name of the dataset
        experiment_duration: timedelta, optional experiment duration
    """
    # 確保目錄存在
    os.makedirs(results_dir, exist_ok=True)
    
    save_path = os.path.join(results_dir, "eval.json")

    result_to_save = {
        "model_name": model_name,
        "method_name": "TDMLLM",
        "dataset_name": dataset_name,
        "total_samples": metrics_result['total'],
        "valid_samples": metrics_result['valid'],
        "invalid_samples": metrics_result['invalid'],
        "accuracy": metrics_result['accuracy'],
        "mcc": metrics_result['mcc'],
        "precision": metrics_result['precision'],
        "recall": metrics_result['recall'],
        "f1_score": metrics_result['f1'],
        "unknown_predictions": metrics_result.get('unknown_predictions', 0),
        "confusion_matrix": {
            "labels": ["DOWN", "UP"],
            "matrix": metrics_result['confusion_matrix']
        }
    }
    
    if experiment_duration is not None:
        result_to_save["experiment_duration"] = {
            "duration": str(experiment_duration),
            "duration_hours": experiment_duration.total_seconds() / 3600,
        }

    with open(save_path, 'w') as f:
        json.dump(result_to_save, f, indent=4)

    return save_path
