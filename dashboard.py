import streamlit as st
import pandas as pd
import requests
import time
import plotly.express as px

API_URL = "http://127.0.0.1:8000/api/v1/analyze"

st.set_page_config(page_title="Stat-Guard Anti-Fraud", layout="wide", page_icon="🛡️")

st.title("🛡️ Stat-Guard: Real-Time Fraud Detection Engine")
st.markdown("Мониторинг транзакций и выявление аномалий на базе математических эвристик.")

if 'results' not in st.session_state:
    st.session_state.results = []

st.sidebar.header("Панель управления")
uploaded_file = st.sidebar.file_uploader("Загрузить CSV с транзакциями", type="csv")
speed = st.sidebar.slider("Скорость симуляции (задержка мс)", 0, 500, 50)
start_button = st.sidebar.button("🚀 Запустить симуляцию")

col1, col2, col3, col4 = st.columns(4)
metric_total = col1.empty()
metric_caught = col2.empty()
metric_fp = col3.empty()
metric_recall = col4.empty()

st.subheader("Последние подозрительные активности")
log_container = st.empty()

st.subheader("Аналитика решений")
chart_container = st.empty()

def update_dashboard():
    df = pd.DataFrame(st.session_state.results)
    if df.empty: return

    total = len(df)
    true_fraud = df[df['actual_fraud'] == 1]
    
    caught = df[(df['actual_fraud'] == 1) & (df['predicted_status'].isin(['DECLINED', 'MANUAL_REVIEW']))]
    fp = df[(df['actual_fraud'] == 0) & (df['predicted_status'] == 'DECLINED')]
    
    recall = (len(caught) / len(true_fraud) * 100) if len(true_fraud) > 0 else 0.0

    metric_total.metric("Обработано", total)
    metric_caught.metric("Фрода остановлено", f"{len(caught)} / {len(true_fraud)}")
    metric_fp.metric("Ложные блокировки", len(fp))
    metric_recall.metric("Точность (Recall)", f"{recall:.1f}%")

    status_counts = df['predicted_status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    fig = px.pie(status_counts, values='Count', names='Status', 
                 color='Status', color_discrete_map={'APPROVED':'#00CC96', 'MANUAL_REVIEW':'#FECB52', 'DECLINED':'#EF553B'},
                 hole=0.4)
    chart_container.plotly_chart(fig, use_container_width=True)

    suspicious = df[df['predicted_status'] != 'APPROVED'].tail(5)
    if not suspicious.empty:
        log_container.dataframe(
            suspicious[['id', 'amount_usd', 'predicted_status', 'risk_score', 'rules']], 
            use_container_width=True, hide_index=True
        )

if start_button and uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if len(df) > 300:
        df = df.sample(300, random_state=42).sort_values('timestamp').reset_index(drop=True)
    st.session_state.results = [] 
    
    progress_bar = st.progress(0)
    
    for index, row in df.iterrows():
        mcc = str(int(row['mcc_code'])) if pd.notna(row['mcc_code']) else None
        purpose = str(row['transfer_purpose']) if pd.notna(row['transfer_purpose']) else None
        lat = float(row['terminal_lat']) if pd.notna(row['terminal_lat']) else None
        lon = float(row['terminal_lon']) if pd.notna(row['terminal_lon']) else None
        
        timestamp_str = str(row['timestamp']).replace(' ', 'T')

        payload = {
            "transaction_id": str(row['transaction_id']),
            "client_id": str(row['client_id']),
            "amount_usd": float(row['amount_usd']),
            "mcc_code": mcc,
            "transfer_purpose": purpose,
            "receiver_id": str(row['receiver_id']) if pd.notna(row.get('receiver_id')) else None,
            "terminal_lat": lat,
            "terminal_lon": lon,
            "ip_address": str(row['ip_address']) if pd.notna(row.get('ip_address')) else None,
            "device_id": str(row['device_id']) if pd.notna(row.get('device_id')) else None,
            "timestamp": timestamp_str
        }

        try:
            res = requests.post(API_URL, json=payload)
            if res.status_code == 200:
                data = res.json()
                st.session_state.results.append({
                    "id": payload['transaction_id'],
                    "amount_usd": payload['amount_usd'],
                    "actual_fraud": int(row['is_fraud']),
                    "predicted_status": data['decision'],
                    "risk_score": data['risk_score'],
                    "rules": ", ".join(data['triggered_rules'])
                })
            else:
                st.error(f"🛑 Ошибка сервера (Строка {index}): {res.status_code} - {res.text}")
                st.stop()
        except Exception as e:
            st.error(f"🛑 Сервер недоступен! Проверьте, запущен ли uvicorn. Ошибка: {e}")
            st.stop()

        if index % 10 == 0:
            progress_bar.progress((index + 1) / len(df))
            update_dashboard()
            time.sleep(speed / 1000.0)
            
    progress_bar.progress(1.0)
    update_dashboard()
    st.success("✅ Симуляция завершена!")
elif start_button and uploaded_file is None:
    st.warning("Пожалуйста, загрузите файл CSV.")