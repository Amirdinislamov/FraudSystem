# Stat-Guard — гибридная система выявления мошеннических транзакций

Дипломная работа по теме **«Разработка системы выявления мошеннических транзакций на основе статистических методов и графового анализа»**.

Stat-Guard — это real-time антифрод-система, объединяющая три уровня анализа:

1. **Статистические эвристики** — онлайн-алгоритмы, не требующие хранения всей истории клиента (Welford, круговая статистика, закон Бенфорда, распределение Пуассона).
2. **Графовый анализ** — гетерогенный граф `client–device–ip` в оперативной памяти; правила hub-and-spoke, циклов, fan-out, PageRank, receiver velocity.
3. **Мета-классификатор** — Random Forest, обучаемый на 13-мерных векторах сырых значений правил; даёт калиброванную вероятность фрода вместо ручной настройки порогов.

Архитектура построена по принципам SOLID: бизнес-логика (`FraudScoringEngine`) изолирована от инфраструктуры через паттерн Repository и инверсию зависимостей. In-Memory хранилище можно заменить на Redis/PostgreSQL без изменений ядра.

## Структура проекта

```
fraud_project/
├── app/
│   ├── main.py            FastAPI-приложение, /api/v1/analyze
│   ├── engine.py          FraudScoringEngine — оркестратор правил + ML
│   ├── rules.py           13 правил, реализующих BaseFraudRule
│   ├── graph.py           TransactionGraph (NetworkX MultiDiGraph)
│   ├── schemas.py         Pydantic-модели транзакций и профиля
│   ├── repositories.py    InMemoryProfileRepository
│   └── interfaces.py      IProfileRepository (для подмены backend-а)
├── data_generator.py      Стохастический генератор синтетических транзакций
├── simulator.py           Sequential нагрузочный прогон через API
├── train_model.py         Обучение Random Forest на rule_raw_values
├── evaluate.py            Графики и метрики для отчёта (reports/*.png)
├── dashboard.py           Streamlit-панель для визуальной демонстрации
├── test_smoke.py          End-to-end smoke ядра
├── test_math.py           Юнит-тесты математических формул
└── requirements.txt
```

## Правила (13 шт.)

### Статистические

| # | Правило | Метрика | Срабатывание |
|---|---|---|---|
| 1 | `Z_SCORE_AMOUNT_ANOMALY` | z-score суммы по MCC, формула Бесселя `σ² = m₂/(n-1)`, онлайн через Welford | z > 3σ |
| 2 | `GEO_VELOCITY_IMPOSSIBLE_TRAVEL` | формула Гаверсинуса, скорость перемещения | > 1000 км/ч или > 1000 км |
| 3 | `POISSON_VELOCITY_ANOMALY` | PMF Пуассона `P(k\|λ) = λᵏe⁻λ/k!` для частоты транзакций | P < 0.001 |
| 4 | `CIRCULAR_TIME_ANOMALY` | круговая статистика `μ = atan2(Σsinθ, Σcosθ)` | отклонение > 6 ч от привычного времени |
| 5 | `DAILY_VOLUME_SPIKE` | сумма за календарный день | > $3000 |
| 6 | `TRANSFER_PURPOSE_ANOMALY` | признак риска P2P-скама | INVESTMENT/CHARITY, новая категория, > $1000 |
| 7 | `BENFORD_LAW_ANOMALY` | χ² против `P(d) = log₁₀(1 + 1/d)` | χ² > 20.09 (df=8, α=0.01), n ≥ 25 |

### Графовые

| # | Правило | Метрика | Срабатывание |
|---|---|---|---|
| 8 | `GRAPH_DROPPER_NETWORK` | unique in-degree получателя | прогрессивно: 2/3/4+ → 35/70/90 |
| 9 | `GRAPH_RECEIVER_VELOCITY` | unique senders за 1 час | ≥ 3 → 60, ≥ 5 → 80 |
| 10 | `GRAPH_CYCLE_DETECTED` | shortest path target → source | путь ≤ 5 хопов |
| 11 | `SHARED_ATTRIBUTE_ANOMALY` | unique clients per device/ip | ≥ 4 → 95, ≥ 2 → 40 |
| 12 | `GRAPH_PAGERANK_ANOMALY` | нормализованный PageRank получателя | > 10× от среднего → 80 |
| 13 | `GRAPH_FAN_OUT_SMURFING` | unique receivers отправителя | ≥ 5 → 70, ≥ 8 → 85 |

## ML-метрики

Random Forest обучается на 13-мерных векторах `rule_raw_values`. Гиперпараметры: `n_estimators=150`, `max_depth=8`, `min_samples_leaf=5`, `class_weight="balanced"`.

Метрики обновляются автоматически после `python3 evaluate.py`. Актуальная таблица — в `reports/metrics.md`.

| Метрика | Значение (синтетика, 65 594 тр.) |
|---|---|
| ROC-AUC | **0.9984** |
| Average Precision | **0.9888** |
| Test F1 (fraud) | 0.9643 |
| Test recall (fraud) | 0.97 |
| Test FP rate | 0.29% (36 / 12 340) |
| 5-fold CV F1 | **0.959 ± 0.004** |
| Best F1 @ optimal threshold | 0.9643 @ t = 0.81 |

### Heuristic-only vs Heuristic + ML

Сравнение демонстрирует ценность мета-классификатора над линейной суммой баллов правил:

| Режим работы | Recall (fraud) | FP rate (на normal) | Пригодность |
|---|---|---|---|
| Heuristic-only baseline | 98.3% | 17.15% | непригоден к production |
| Heuristic + Random Forest | 97.0% | 0.29% | production-ready |

ML-надстройка снижает долю ложных блокировок в **59 раз** при практически том же recall за счёт обучения нелинейных комбинаций сигналов: например, `GRAPH_DROPPER_NETWORK = 2 senders` сам по себе теперь даёт 35 баллов в правилах, но модель оценивает этот признак в контексте `SHARED_ATTRIBUTE_ANOMALY` и `CIRCULAR_TIME_ANOMALY` и отбрасывает ложные срабатывания на популярных получателях.

## Развёртывание

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Полный пайплайн «с нуля до отчёта»:

```bash
# 1. Сгенерировать датасет (1500 клиентов × 60 дней)
python3 data_generator.py            # heavy_fraud_data.csv

# 2. В отдельном терминале — поднять API
uvicorn app.main:app --reload

# 3. Прогнать симуляцию (sequential по timestamp!)
python3 simulator.py                 # ml_dataset.csv

# 4. Обучить ML-модель
python3 train_model.py               # fraud_ml_model.pkl

# 5. Сгенерировать графики и метрики
python3 evaluate.py                  # reports/*.png + metrics.md

# (опционально) Streamlit dashboard
streamlit run dashboard.py
```

Проверка корректности математики и ядра:

```bash
python3 test_math.py        # формулы (Welford, Haversine, Benford, Poisson)
python3 test_smoke.py       # end-to-end ядра без HTTP
```

## API

`POST /api/v1/analyze`

```json
{
  "transaction_id": "uuid",
  "client_id": "USER_001",
  "amount_usd": 1250.0,
  "mcc_code": "5732",
  "transfer_purpose": "PURCHASE",
  "receiver_id": null,
  "terminal_lat": 41.31,
  "terminal_lon": 69.24,
  "ip_address": "10.0.0.1",
  "device_id": "dev-1",
  "timestamp": "2025-01-01T12:00:00"
}
```

Ответ:

```json
{
  "transaction_id": "...",
  "decision": "DECLINED",          // APPROVED | MANUAL_REVIEW | DECLINED
  "risk_score": 92,                // 0..100
  "triggered_rules": ["GRAPH_DROPPER_NETWORK", "Z_SCORE_AMOUNT_ANOMALY"],
  "rule_scores": { ... },          // балл каждого правила
  "rule_raw_values": { ... }       // сырые значения (для feature engineering)
}
```

`GET /api/v1/profile/{client_id}` — текущее состояние профиля.
`GET /api/v1/stats` — сводка по графу и репозиторию.

## Ограничения проекта (раздел «Limitations» для диплома)

1. **Синтетический датасет.** В `data_generator.py` все жертвы Bot Farm ATO ходят с одного `bot_farm_device` и `bot_farm_ip`, что делает `SHARED_ATTRIBUTE_ANOMALY` почти идеальным предиктором. На реальных данных доля этой фичи в feature importance будет ниже.
2. **Короткий горизонт.** Profile-зависимые правила (`POISSON_VELOCITY_ANOMALY`, `BENFORD_LAW_ANOMALY`) требуют истории > 25 транзакций; на 60-дневном датасете их вклад в модель невелик. На production-горизонте (6+ месяцев) их доля вырастет.
3. **In-memory state.** Для horizontal scaling нужна замена `InMemoryProfileRepository` на Redis (профили) и persistent graph-store (Neo4j/Memgraph). Интерфейс `IProfileRepository` для этого готов.
4. **PageRank пересчитывается раз в 50 транзакций.** На «холодном» старте кэш пуст и правило не срабатывает. Это компромисс между точностью и латентностью.
5. **Random Forest без калибровки вероятностей.** Для production-ready использования стоит обернуть в `CalibratedClassifierCV` (Platt scaling или isotonic).
