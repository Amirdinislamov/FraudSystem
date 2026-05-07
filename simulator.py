"""
Симулятор нагрузки. Отправляет транзакции в API строго в хронологическом
порядке (sequential). Это критично: ядро использует онлайн-статистику
(Welford, Geo-velocity, Poisson, Circular Time, Daily Volume, Graph Cycle),
которая становится невалидной при перестановке событий во времени.

Параллелизм через ThreadPoolExecutor + as_completed нарушает порядок
поступления и приводит к некорректным признакам в ml_dataset.csv.

Симулятор НЕ молчит про ошибки: первые DEBUG_FAILURE_DUMP неудач
выводятся целиком (status + body + payload), это позволяет быстро
найти и устранить причину.
"""

import json
import time
from collections import Counter

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "http://127.0.0.1:8000/api/v1/analyze"
DEBUG_FAILURE_DUMP = 5  


def make_session() -> requests.Session:
    """Сессия с автоматическим переподключением при сбросе соединения.

    Без retry на длинной симуляции часть запросов теряется не по вине
    бизнес-логики, а из-за keep-alive: сервер закрывает идле-соединение,
    клиент шлёт в дохлый сокет, получает ConnectionResetError. Retry со
    встроенным back-off снимает 99% таких сбоев.
    """
    s = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.2,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def build_payload(row):
    return {
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


def run_full_load_test():
    df = pd.read_csv("heavy_fraud_data.csv")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"🚀 Запуск симуляции (sequential): {len(df)} транзакций...")

    session = make_session()
    start_time = time.time()
    results = []
    failure_examples = []  
    failure_kinds = Counter()  

    total = len(df)
    for idx, row in df.iterrows():
        payload = build_payload(row)
        try:
            r = session.post(API_URL, json=payload, timeout=15)
            if r.status_code == 200:
                results.append((r.json(), int(row["is_fraud"]), str(row.get("scenario", "Unknown"))))
            else:
                failure_kinds[f"HTTP {r.status_code}"] += 1
                if len(failure_examples) < DEBUG_FAILURE_DUMP:
                    failure_examples.append({
                        "row_idx": int(idx),
                        "status": r.status_code,
                        "body": r.text[:500],
                        "payload": payload,
                    })
        except requests.exceptions.RequestException as e:
            failure_kinds[type(e).__name__] += 1
            if len(failure_examples) < DEBUG_FAILURE_DUMP:
                failure_examples.append({
                    "row_idx": int(idx),
                    "status": "EXC",
                    "body": f"{type(e).__name__}: {e}",
                    "payload": payload,
                })

        if (idx + 1) % 2000 == 0:
            elapsed = time.time() - start_time
            rps = (idx + 1) / elapsed if elapsed > 0 else 0.0
            failed_so_far = sum(failure_kinds.values())
            print(f"⏳ {idx + 1}/{total}  |  {rps:.1f} tx/s  |  ошибок: {failed_so_far}")

    duration = time.time() - start_time
    total_failures = sum(failure_kinds.values())

    if total_failures:
        print("\n" + "─" * 60)
        print(f"⚠️  Категории ошибок ({total_failures} всего):")
        for kind, n in failure_kinds.most_common():
            print(f"   {kind:25s} {n}")
        print("\nПервые ошибки целиком:")
        for ex in failure_examples:
            print(f"\n  row #{ex['row_idx']}  status={ex['status']}")
            print(f"  body: {ex['body']}")
            print(f"  payload: {json.dumps(ex['payload'], default=str)[:300]}")
        print("─" * 60)

    if not results:
        print("\n❌ Ни одного успешного ответа от API. Запустите uvicorn app.main:app")
        return

    res_df = pd.DataFrame([
        {
            "actual_fraud": r[1],
            "predicted_decision": r[0]["decision"],
            "scenario": r[2],
        }
        for r in results
    ])

    print("\n" + "=" * 60)
    print("📊 ГЕНЕРАЛЬНЫЙ ОТЧЁТ (heuristic-only baseline)")
    print("=" * 60)

    for name, group in res_df.groupby("scenario"):
        total_g = len(group)
        if name == "Normal":
            fp = len(group[group["predicted_decision"] == "DECLINED"])
            print(f"✅ {name:25}: ложных блокировок: {fp}/{total_g} ({fp/total_g*100:.2f}%)")
        else:
            caught = len(group[group["predicted_decision"] != "APPROVED"])
            print(f"🚨 {name:25}: поймано {caught}/{total_g} ({caught/total_g*100:.1f}%)")

    total_fraud_actual = len(res_df[res_df["actual_fraud"] == 1])
    caught_total = len(res_df[(res_df["actual_fraud"] == 1) & (res_df["predicted_decision"] != "APPROVED")])

    print("\n" + "=" * 60)
    print(f"⚡ Скорость: {len(results)/duration:.1f} tx/s   |   успех: {len(results)}/{total}   |   ошибок: {total_failures}")
    if total_fraud_actual:
        print(f"🎯 Recall (любой alert): {caught_total/total_fraud_actual:.1%}")
    print("=" * 60)
    print("\n💾 Сохранение датасета признаков для обучения ML...")

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
        print(f"✅ ml_dataset.csv: {len(ml_df)} строк, {len(ml_df.columns) - 1} признаков")
        print(f"   Признаки: {[c for c in ml_df.columns if c != 'is_fraud']}")


if __name__ == "__main__":
    run_full_load_test()
