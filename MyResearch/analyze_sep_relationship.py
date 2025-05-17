import os
import pandas as pd
from pathlib import Path
from datetime import datetime

def get_sep_tickers():
    """取得 SEP 55 檔股票的 ticker"""
    price_dir = "../datasets/SEP/sn2/price/preprocessed"
    return [f[:-4] for f in os.listdir(price_dir) if f.endswith('.txt')]

def load_findkg_entities():
    """讀取 FinDKG 的實體映射"""
    findkg_path = Path("../datasets/FinDKG/FinDKG_dataset/FinDKG")
    entity2id = {}
    entity_types = {}
    with open(findkg_path / "entity2id.txt", 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                entity, eid, type_name, type_id = parts[:4]
                entity2id[entity] = eid
                entity_types[entity] = type_name
    return entity2id, entity_types

def load_time_mapping():
    """讀取時間映射"""
    findkg_path = Path("../datasets/FinDKG/FinDKG_dataset/FinDKG-full")
    time2id = {}
    id2time = {}
    with open(findkg_path / "time2id.txt", 'r', encoding='utf-8') as f:
        next(f)  # Skip header line
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                time_id, date_wk = parts[:2]
                time2id[date_wk] = time_id
                id2time[time_id] = date_wk
    return time2id, id2time

def convert_date_to_findkg_time(date_str):
    """將日期字串轉換為 FinDKG 時間格式"""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return None

def get_company_entities(entity2id, entity_types):
    """篩選出所有公司實體"""
    companies = {}
    for entity, eid in entity2id.items():
        if entity_types[entity] == 'COMP':
            companies[entity] = eid
    return companies

def create_ticker_to_entity_mapping():
    """建立 ticker 到 FinDKG entity 的映射"""
    return {
        # 科技公司
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOG": "Alphabet Inc.",
        "META": "Meta Platforms Inc.",
        "NVDA": "Nvidia Corp.",
        "INTC": "Intel Corporation",
        "AMD": "Advanced Micro Devices",
        "QCOM": "Qualcomm Inc.",
        "TCEHY": "Tencent Holdings Ltd",
        "005930.KS": "Samsung Electronics Co",
        "9984.T": "SoftBank Group Corp.",
        
        # 電商/零售
        "AMZN": "Amazon.com Inc.",
        "BABA": "Alibaba Group Holding Ltd",
        "WMT": "Walmart Inc.",
        "COST": "Costco Wholesale Corp.",
        "HD": "Home Depot Inc.",
        
        # 金融服務
        "JPM": "JPMorgan Chase & Co",
        "BAC": "Bank of America Corp.",
        "WFC": "Wells Fargo Co.",
        "C": "Citigroup Inc.",
        "GS": "Goldman Sachs Group",
        "MS": "Morgan Stanley",
        "BRK-A": "Berkshire Hathaway Inc.",
        "V": "Visa Inc.",
        "MA": "Mastercard",
        "PYPL": "PayPal Holdings Inc.",
        
        # 消費品/餐飲
        "SBUX": "Starbucks",
        "MCD": "McDonald's",
        "NKE": "Nike",
        "PEP": "PepsiCo Inc.",
        "KO": "Coca-Cola",
        
        # 醫療保健
        "PFE": "Pfizer Inc.",
        "JNJ": "Johnson & Johnson",
        "MRK": "Merck MRK",
        "ABBV": "AbbVie Inc.",
        "TMO": "Thermo Fisher Scientific Inc.",
        "UNH": "UnitedHealth Group",
        "LLY": "Eli Lilly & Co.",
        "ABT": "Abbott Laboratories",
        "BNTX": "BioNTech SE",
        "MRNA": "Moderna Inc.",
        "AZN": "AstraZeneca PLC",
        
        # 能源
        "CVX": "Chevron Corp.",
        "XOM": "Exxon",
        "SHEL": "Shell PLC",
        "TTE": "TotalEnergies SE",
        
        # 通訊/媒體
        "T": "AT&T Inc.",
        "VZ": "Verizon",
        "DIS": "Disney",
        "NFLX": "Netflix Inc.",
        "CMCSA": "Comcast Corp.",
        
        # 交通運輸
        "TSLA": "Tesla Inc.",
        "BA": "Boeing Co.",
        "UBER": "Uber Technologies Inc.",
        "LYFT": "Lyft Inc.",
        "UPS": "UPS",
        
        # 工業/製造
        "CAT": "Caterpillar Inc.",
        "HON": "Honeywell International Inc.",
        "LMT": "Lockheed Martin Corp.",
        "NEE": "NextEra Energy Inc.",
        "DUK": "Duke Energy Corp.",
        "AEP": "American Electric Power Company Inc.",
        "SO": "Southern Company",
        "UNP": "Union Pacific Corp.",
        
        # 其他
        "D": "Dominion Energy Inc.",
        "TM": "Toyota Motor Corp.",
        "BHP": "BHP Group Ltd.",
        "VALE": "Vale",
        "SHW": "Sherwin-Williams Company",
        "APD": "Air Products and Chemicals Inc.",
        "PSA": "Public Storage",
        "CCI": "Crown Castle International Corp.",
        "AMT": "American Tower Corporation",
        "EQIX": "Equinix Inc.",
        "TSM": "Taiwan Semiconductor Manufacturing Co.",
        "COP": "ConocoPhillips",
        "AVGO": "Broadcom",
        "PG": "Procter & Gamble",
        "PLD": "Prologis Inc.",
        "RIO": "Rio Tinto"
    }

def analyze_sep_relationships():
    """分析 SEP 股票間在 FinDKG 中的關係"""
    # 1. 取得 SEP tickers
    sep_tickers = get_sep_tickers()
    print(f"找到 {len(sep_tickers)} 個 SEP 股票 ticker")
    
    # 2. 讀取 FinDKG 實體
    entity2id, entity_types = load_findkg_entities()
    print(f"找到 {len(entity2id)} 個 FinDKG 實體")
    
    # 2.1 讀取時間映射
    time2id, id2time = load_time_mapping()
    print(f"找到 {len(time2id)} 個時間點")
    
    # 2.2 篩選出所有公司實體
    company_entities = get_company_entities(entity2id, entity_types)
    # print(f"\nFinDKG 中的公司實體：")
    # for company, eid in company_entities.items():
    #     print(f"{company} (ID: {eid})")
    
    # 3. 建立 ticker 到 entity 的映射
    ticker_to_entity = create_ticker_to_entity_mapping()
    
    # 4. 建立 mapping
    mapping = {}
    for ticker in sep_tickers:
        entity_name = ticker_to_entity.get(ticker)
        if entity_name and entity_name in entity2id:
            mapping[ticker] = entity_name
        else:
            mapping[ticker] = None
    
    # 5. 輸出 mapping 結果
    print("\nTicker to FinDKG entity mapping:")
    for ticker, entity in mapping.items():
        print(f"{ticker} -> {entity}")
    
    # 6. 列出無法自動對應的 ticker
    unmapped = [ticker for ticker, entity in mapping.items() if entity is None]
    if unmapped:
        print("\n無法自動對應的 ticker，請人工補充：")
        print(unmapped)
    
    # 7. 讀取 FinDKG 的三元組，分析 SEP 股票間的關係
    findkg_path = Path("../datasets/FinDKG/FinDKG_dataset/FinDKG-full")
    triplets = []
    
    # 設定時間範圍
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    print("\n開始分析 SEP 股票間的關係...")
    # 讀取所有三個檔案
    for file_name in ['train.txt', 'valid.txt', 'test.txt']:
        print(f"\n處理 {file_name}...")
        with open(findkg_path / file_name, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    subj, rel, obj, time, _ = parts[:5]
                    subj_name = [k for k, v in entity2id.items() if v == subj]
                    obj_name = [k for k, v in entity2id.items() if v == obj]
                    
                    if subj_name and obj_name:
                        subj_name = subj_name[0]
                        obj_name = obj_name[0]
                        
                        # 轉換時間
                        time_str = id2time.get(time)
                        if time_str:
                            time_date = datetime.strptime(time_str, '%Y-%m-%d')
                            
                            # 檢查是否在指定時間範圍內
                            if start_date <= time_date <= end_date:
                                # 檢查是否為 SEP 股票
                                if (subj_name in mapping.values() and obj_name in mapping.values() and
                                    entity_types[subj_name] == 'COMP' and entity_types[obj_name] == 'COMP'):
                                    triplets.append({
                                        'subject': subj_name,
                                        'relation': rel,
                                        'object': obj_name,
                                        'time': time_str,
                                        'source_file': file_name
                                    })
    
    # 8. 轉換為 DataFrame 並分析
    df = pd.DataFrame(triplets)
    if not df.empty:
        print(f"\n找到 {len(df)} 個 SEP 股票間的關係")
        print("\n關係類型統計：")
        print(df['relation'].value_counts())
        print("\n所有關係：")
        print(df)
        
        print("\n詳細關係資料：")
        print("=" * 100)
        for idx, row in df.iterrows():
            print(f"\n關係 #{idx + 1}:")
            print(f"時間: {row['time']}")
            print(f"來源檔案: {row['source_file']}")
            print(f"主體公司: {row['subject']}")
            print(f"關係類型: {row['relation']}")
            print(f"客體公司: {row['object']}")
            print("-" * 50)
        
        # 9. 分析時間分布
        df['time'] = pd.to_datetime(df['time'])
        print("\n時間分布：")
        print(df['time'].dt.year.value_counts().sort_index())
        
        # # 10. 儲存結果到 CSV 檔案
        # output_dir = Path("../datasets/SEP/sn2/relationships")
        # output_dir.mkdir(parents=True, exist_ok=True)
        # output_file = output_dir / f"SEP_stock_relationships_2020_2022.csv"
        # df.to_csv(output_file, index=False, encoding='utf-8')
        # print(f"\n關係資料已儲存至：{output_file}")
    else:
        print("\n沒有找到 SEP 股票間的關係")

if __name__ == "__main__":
    analyze_sep_relationships()