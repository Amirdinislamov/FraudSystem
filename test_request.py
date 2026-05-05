import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/api/v1/analyze"
df = pd.read_csv("heavy_fraud_data.csv")
row = df.iloc[0]

payload = {
    "transaction_id": row['transaction_id'],
    "client_id": row['client_id'],
    "amount_usd": float(row['amount_usd']),
    "mcc_code": str(int(row['mcc_code'])) if pd.notna(row['mcc_code']) else None,
    "transfer_purpose": row['transfer_purpose'] if pd.notna(row['transfer_purpose']) else None,
    "receiver_id": str(row['receiver_id']) if pd.notna(row['receiver_id']) else None,
    "terminal_lat": float(row['terminal_lat']) if pd.notna(row['terminal_lat']) else None,
    "terminal_lon": float(row['terminal_lon']) if pd.notna(row['terminal_lon']) else None,
    "ip_address": str(row['ip_address']) if 'ip_address' in row and pd.notna(row['ip_address']) else None,
    "device_id": str(row['device_id']) if 'device_id' in row and pd.notna(row['device_id']) else None,
    "timestamp": str(row['timestamp'])
}

print(payload)
r = requests.post(API_URL, json=payload)
print(r.status_code)
print(r.text)
