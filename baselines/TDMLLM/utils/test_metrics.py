from metrics import calculate_metrics, save_metrics

def test_metrics():
    # 測試案例 1: 完美預測
    perfect_preds = ["Positive", "Negative", "Positive", "Negative", "Positive"]
    perfect_labels = ["Positive", "Negative", "Positive", "Negative", "Positive"]
    
    # 測試案例 2: 部分正確預測
    mixed_preds = ["Positive", "Negative", "Positive", "Positive", "Negative"]
    mixed_labels = ["Positive", "Negative", "Negative", "Positive", "Negative"]
    
    # 測試案例 3: 包含 Unknown 預測
    unknown_preds = ["Positive", "Unknown", "Negative", "Positive", "Unknown"]
    unknown_labels = ["Positive", "Negative", "Negative", "Positive", "Negative"]
    
    # 測試案例 4: 空列表
    empty_preds = []
    empty_labels = []

    # 執行測試
    print("\n=== 測試案例 1: 完美預測 ===")
    metrics1 = calculate_metrics(perfect_preds, perfect_labels)
    print_metrics(metrics1)
    
    print("\n=== 測試案例 2: 部分正確預測 ===")
    metrics2 = calculate_metrics(mixed_preds, mixed_labels)
    print_metrics(metrics2)
    
    print("\n=== 測試案例 3: 包含 Unknown 預測 ===")
    metrics3 = calculate_metrics(unknown_preds, unknown_labels)
    print_metrics(metrics3)
    
    print("\n=== 測試案例 4: 空列表 ===")
    metrics4 = calculate_metrics(empty_preds, empty_labels)
    print_metrics(metrics4)

    # 測試保存功能
    print("\n=== 測試保存功能 ===")
    save_path = save_metrics(metrics2, "test_model")
    print(f"結果已保存至: {save_path}")

def print_metrics(metrics):
    print(f"總樣本數: {metrics['total']}")
    print(f"有效樣本數: {metrics['valid']}")
    print(f"無效樣本數: {metrics['invalid']}")
    print(f"準確率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"精確率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall): {metrics['recall']:.4f}")
    print(f"F1 分數: {metrics['f1']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print("混淆矩陣:")
    print("          Predicted")
    print("          Negative Positive")
    print(f"Actual Negative    {metrics['confusion_matrix'][0][0]:<8} {metrics['confusion_matrix'][0][1]:<8}")
    print(f"        Positive    {metrics['confusion_matrix'][1][0]:<8} {metrics['confusion_matrix'][1][1]:<8}")

if __name__ == "__main__":
    test_metrics() 