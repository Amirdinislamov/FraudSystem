import pandas as pd
import requests
import time
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "http://127.0.0.1:8000/api/v1/analyze"

session = requests.Session()
adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
session.mount('http://', adapter)

def send_tx(row):
    payload = {
        "transaction_id": row['transaction_id'],
        "client_id": row['client_id'],
        "amount_usd": float(row['amount_usd']),
        "mcc_code": str(int(row['mcc_code'])) if pd.notna(row['mcc_code']) else None,
        "transfer_purpose": row['transfer_purpose'] if pd.notna(row['transfer_purpose']) else None,
        "receiver_id": str(row['receiver_id']) if pd.notna(row['receiver_id']) else None,
        "terminal_lat": float(row['terminal_lat']) if pd.notna(row['terminal_lat']) else None,
        "terminal_lon": float(row['terminal_lon']) if pd.notna(row['terminal_lon']) else None,
        "timestamp": str(row['timestamp'])
    }
    try:
        r = session.post(API_URL, json=payload, timeout=5)
        if r.status_code == 200:
            return r.json(), row['is_fraud'], row['scenario']
        return None
    except Exception:
        return None

def run_full_load_test():
    df = pd.read_csv("heavy_fraud_data.csv")
    print(f"🚀 ЗАПУСК ПОЛНОМАСШТАБНОГО ТЕСТА: {len(df)} транзакций...")
    
    start_time = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(send_tx, row) for _, row in df.iterrows()]
        
        processed = 0
        total = len(df)
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
            processed += 1
            if processed % 2000 == 0:
                print(f"⏳ Обработано: {processed} / {total}...")

    duration = time.time() - start_time
    
    res_df = pd.DataFrame([
        {"actual_fraud": r[1], "predicted_decision": r[0]['decision'], "scenario": r[2]} 
        for r in results
    ])
    
    print("\n" + "="*50)
    print("📊 ГЕНЕРАЛЬНЫЙ ОТЧЕТ ДЛЯ ДИПЛОМА (21 000+ TX)")
    print("="*50)

    scenarios = res_df.groupby('scenario')
    for name, group in scenarios:
        if name == "Normal":
            fp = len(group[group['predicted_decision'] == 'DECLINED'])
            total_normal = len(group)
            print(f"✅ {name:20}: Ложных блокировок: {fp} из {total_normal} ({(fp/total_normal)*100:.2f}%)")
        else:
            total_fraud = len(group)
            caught = len(group[group['predicted_decision'] != 'APPROVED'])
            print(f"🚨 {name:20}: Поймано {caught} из {total_fraud} ({(caught/total_fraud)*100:.1f}%)")

    total_fraud_actual = len(res_df[res_df['actual_fraud'] == 1])
    caught_total = len(res_df[(res_df['actual_fraud'] == 1) & (res_df['predicted_decision'] != 'APPROVED')])
    
    print("\n" + "="*50)
    print(f"⚡ СКОРОСТЬ ОБРАБОТКИ: {len(results)/duration:.1f} транзакций в секунду")
    print(f"🎯 ОБЩАЯ ЭФФЕКТИВНОСТЬ (RECALL): {caught_total/total_fraud_actual:.1%}")
    print("="*50)


    print("\n💾 Сохраняем датасет для обучения ML...")
    import csv
    with open('ml_dataset.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        if results:
            rule_names = list(results[0][0]['rule_scores'].keys())
            headers = rule_names + ['is_fraud']
            writer.writerow(headers)

            for res in results:
                response_json, actual_fraud, _ = res
                scores = response_json.get('rule_scores', {})
                row = [scores.get(rule, 0) for rule in rule_names] + [actual_fraud]
                writer.writerow(row)
                
    print("✅ Датасет 'ml_dataset.csv' успешно сохранен! В нем", len(results), "строк.")

if __name__ == "__main__":
    run_full_load_test()