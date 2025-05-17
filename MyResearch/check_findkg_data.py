import os
import pandas as pd
from pathlib import Path

def check_findkg_data():
    # 設定資料集路徑
    findkg_path = Path("../datasets/FinDKG/FinDKG_dataset/FinDKG")
    
    # 讀取實體和關係的映射
    print("=== 讀取實體和關係映射 ===")
    entity2id = {}
    entity_types = {}
    with open(findkg_path / "entity2id.txt", 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                entity, id, type_name, type_id = parts[:4]
                entity2id[id] = entity
                entity_types[entity] = type_name
    
    relation2id = {}
    with open(findkg_path / "relation2id.txt", 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                relation, id = parts[:2]
                relation2id[id] = relation
    
    # 讀取訓練資料
    print("\n=== 讀取訓練資料 ===")
    train_data = []
    with open(findkg_path / "train.txt", 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                subj, rel, obj, time, _ = parts[:5]
                train_data.append({
                    'subject': entity2id.get(subj, subj),
                    'subject_type': entity_types.get(entity2id.get(subj, subj), 'UNKNOWN'),
                    'relation': relation2id.get(rel, rel),
                    'object': entity2id.get(obj, obj),
                    'object_type': entity_types.get(entity2id.get(obj, obj), 'UNKNOWN'),
                    'time': time
                })
    
    # 轉換為 DataFrame 以便分析
    df = pd.DataFrame(train_data)
    
    # 顯示基本統計資訊
    print("\n=== 資料集統計資訊 ===")
    print(f"總三元組數量: {len(df)}")
    print(f"唯一實體數量: {len(set(df['subject'].unique()) | set(df['object'].unique()))}")
    print(f"唯一關係數量: {len(df['relation'].unique())}")
    
    # 顯示實體類型統計
    print("\n=== 實體類型統計 ===")
    subject_types = df['subject_type'].value_counts()
    object_types = df['object_type'].value_counts()
    print("\n主體實體類型:")
    print(subject_types)
    print("\n客體實體類型:")
    print(object_types)
    
    # 顯示一些範例資料
    print("\n=== 範例資料 ===")
    print(df.head())
    
    # 檢查是否包含股票相關的實體
    print("\n=== 檢查股票相關實體 ===")
    stock_entities = [e for e in entity2id.values() if any(x in e.lower() for x in ['stock', 'share', 'equity'])]
    print(f"找到 {len(stock_entities)} 個可能與股票相關的實體")
    if stock_entities:
        print("前 10 個股票相關實體:", stock_entities[:10])
    
    # 檢查公司實體
    print("\n=== 公司實體 ===")
    company_entities = [e for e, t in entity_types.items() if t == 'COMP']
    print(f"找到 {len(company_entities)} 個公司實體")
    if company_entities:
        print("前 10 個公司實體:", company_entities[:10])

if __name__ == "__main__":
    check_findkg_data()

price_dir = "baseline/SOTA_SEP/sep/data/sample_price/preprocessed"
tickers = [f[:-4] for f in os.listdir(price_dir) if f.endswith('.txt')]
print(f"Total stocks: {len(tickers)}")
print(tickers) 