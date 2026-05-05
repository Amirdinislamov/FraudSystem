import pandas as pd
import requests
import time
import concurrent.futures
from requests.adapters import HTTPAdapter

API_URL = "http://127.0.0.1:8000/api/v1/analyze"

session = requests.Session()
adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
session.mount("http://", adapter)


def send_tx(row):
    payload = {
        "transaction_id": str(row["transaction_id"]),
        "client_id": str(row["client_id"]),
        "amount_usd": float(row["amount_usd"]),
        "mcc_code": str(int(row["mcc_code"])) if pd.notna(row.get("mcc_code")) else None,
        "transfer_purpose": str(row["transfer_purpose"]) if pd.notna(row.get("transfer_purpose")) else None,
        "receiver_id": str(row["receiver_id"]) if pd.notna(row.get("receiver_id")) else None,
        "terminal_lat": float(row["terminal_lat"]) if pd.notna(row.get("terminal_lat")) else None,
        "terminal_lon": float(row["terminal_lon"]) if pd.notna(row.get("terminal_lon")) else None,
        "ip_address": str(row["ip_address"]) if pd.notna(row.get("ip_address")) else None,
        "device_id": str(row["device_id"]) if pd.notna(row.get("device_id")) else None,
        "timestamp": str(row["timestamp"]).replace(" ", "T"),
    }
    try:
        r = session.post(API_URL, json=payload, timeout=5)
        if r.status_code == 200:
            return r.json(), int(row["is_fraud"]), str(row.get("scenario", "Unknown"))
        return None
    except Exception as e:
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
        {
            "actual_fraud": r[1],
            "predicted_decision": r[0]["decision"],
            "scenario": r[2],
        }
        for r in results
    ])

    print("\n" + "=" * 60)
    print("📊 ГЕНЕРАЛЬНЫЙ ОТЧЕТ (ДИПЛОМ)")
    print("=" * 60)

    for name, group in res_df.groupby("scenario"):
        total_g = len(group)
        if name == "Normal":
            fp = len(group[group["predicted_decision"] == "DECLINED"])
            print(f"✅ {name:25}: Ложных блокировок: {fp}/{total_g} ({fp/total_g*100:.2f}%)")
        else:
            caught = len(group[group["predicted_decision"] != "APPROVED"])
            print(f"🚨 {name:25}: Поймано {caught}/{total_g} ({caught/total_g*100:.1f}%)")

    total_fraud_actual = len(res_df[res_df["actual_fraud"] == 1])
    caught_total = len(res_df[(res_df["actual_fraud"] == 1) & (res_df["predicted_decision"] != "APPROVED")])

    print("\n" + "=" * 60)
    print(f"⚡ СКОРОСТЬ: {len(results)/duration:.1f} транзакций/сек")
    print(f"🎯 ОБЩИЙ RECALL: {caught_total/total_fraud_actual:.1%}")
    print("=" * 60)
    print("\n💾 Сохранение датасета для обучения ML (rule_raw_values)...")

    rows = []
    for res_json, actual_fraud, _ in results:
        raw_values = res_json.get("rule_raw_values", {})
        if not raw_values:
            raw_values = res_json.get("rule_scores", {})
        raw_values["is_fraud"] = actual_fraud
        rows.append(raw_values)

    if rows:
        ml_df = pd.DataFrame(rows).fillna(0)
        cols = [c for c in ml_df.columns if c != "is_fraud"] + ["is_fraud"]
        ml_df = ml_df[cols]
        ml_df.to_csv("ml_dataset.csv", index=False)
        print(f"✅ ml_dataset.csv сохранён: {len(ml_df)} строк, {len(ml_df.columns)-1} признаков")
        print(f"   Признаки: {[c for c in ml_df.columns if c != 'is_fraud']}")


if __name__ == "__main__":
    run_full_load_test()