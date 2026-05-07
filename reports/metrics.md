# Метрики итоговой ML-модели

| Метрика | Значение |
|---|---|
| Размер датасета | 65594 транзакций |
| Доля fraud | 5.94% |
| Размер train / test | 52475 / 13119 |
| ROC-AUC | **0.9984** |
| Average Precision | **0.9888** |
| Test F1 (fraud) | 0.9611 |
| Test Precision (fraud) | 0.9544 |
| Test Recall (fraud) | 0.9679 |
| Best F1 @ optimal threshold | 0.9643 @ t=0.81 |
| Cross-validation F1 (5-fold) | 0.9590 ± 0.0039 |

## Confusion matrix (test, threshold = 0.5)

| | Predicted Normal | Predicted Fraud |
|---|---|---|
| Actual Normal | 12304 | 36 |
| Actual Fraud | 25 | 754 |

## Feature importance

| Признак | Важность |
|---|---|
| `SHARED_ATTRIBUTE_ANOMALY` | 0.4663 |
| `CIRCULAR_TIME_ANOMALY` | 0.1321 |
| `GRAPH_PAGERANK_ANOMALY` | 0.1075 |
| `GRAPH_DROPPER_NETWORK` | 0.0891 |
| `GRAPH_CYCLE_DETECTED` | 0.0497 |
| `GRAPH_FAN_OUT_SMURFING` | 0.0457 |
| `DAILY_VOLUME_SPIKE` | 0.0384 |
| `Z_SCORE_AMOUNT_ANOMALY` | 0.0215 |
| `GRAPH_RECEIVER_VELOCITY` | 0.0202 |
| `TRANSFER_PURPOSE_ANOMALY` | 0.0171 |
| `GEO_VELOCITY_IMPOSSIBLE_TRAVEL` | 0.0090 |
| `BENFORD_LAW_ANOMALY` | 0.0034 |
| `POISSON_VELOCITY_ANOMALY` | 0.0000 |
