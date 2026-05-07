"""
Smoke-test ядра без HTTP-слоя. Прогоняет несколько синтетических
транзакций через FraudScoringEngine + InMemoryProfileRepository
и убеждается, что:

    1. Все правила импортируются и регистрируются.
    2. Welford-статистика корректно накапливается (z-score срабатывает на
       аномальной сумме при наличии разнообразной истории).
    3. GeoVelocity срабатывает при «телепортации».
    4. Граф фиксирует уникальные транзакции.
    5. Benford срабатывает при искусственно перекошенном распределении.

ВАЖНО: каждому тестовому клиенту даются УНИКАЛЬНЫЕ device_id и ip,
иначе SharedAttributeRule повышает риск-скор и движок решает
MANUAL_REVIEW → _update_state не вызывается → последующие правила
читают пустой профиль и не срабатывают.

Запуск:    python test_smoke.py
"""

import asyncio
import random
from datetime import datetime, timedelta
from uuid import uuid4

from app.engine import FraudScoringEngine
from app.graph import TransactionGraph
from app.repositories import InMemoryProfileRepository
from app.schemas import TransactionPayload, TransferPurpose
from app.rules import (
    ZScoreAmountRule, GeoVelocityRule, PoissonVelocityRule,
    CircularTimeRule, TransferPurposeAnomalyRule, DailyVolumeSpikeRule,
    BenfordsLawRule,
    GraphDropperNetworkRule, GraphCycleRule, SharedAttributeRule,
    PageRankAnomalyRule, FanOutSmurfingRule, ReceiverVelocityRule,
)


def make_tx(client, ts, amount=20.0, mcc="5411", lat=41.31, lon=69.24,
            receiver=None, purpose=None, dev=None, ip=None):
    return TransactionPayload(
        transaction_id=uuid4(),
        client_id=client,
        amount_usd=amount,
        mcc_code=mcc,
        transfer_purpose=purpose,
        receiver_id=receiver,
        terminal_lat=lat,
        terminal_lon=lon,
        # Уникальные device/ip по умолчанию — иначе SharedAttributeRule
        # начнёт триггерить как только два клиента используют одно поле.
        ip_address=ip or f"10.0.{hash(client) & 0xff}.1",
        device_id=dev or f"dev-{client}",
        timestamp=ts,
    )


def build_engine(rules):
    repo = InMemoryProfileRepository()
    graph = TransactionGraph()
    eng = FraudScoringEngine(repo=repo, rules=rules, graph=graph)
    eng.model = None  # принудительно эвристический режим
    return eng, repo, graph


async def run():
    rules = [
        ZScoreAmountRule(), GeoVelocityRule(), PoissonVelocityRule(),
        CircularTimeRule(), DailyVolumeSpikeRule(), TransferPurposeAnomalyRule(),
        BenfordsLawRule(),
        GraphDropperNetworkRule(), ReceiverVelocityRule(), GraphCycleRule(),
        SharedAttributeRule(), PageRankAnomalyRule(), FanOutSmurfingRule(),
    ]
    eng, repo, graph = build_engine(rules)

    base = datetime(2025, 1, 1, 12, 0, 0)
    failures = []
    rng = random.Random(42)

    # --- 1. История с разбросом сумм (Welford нуждается в σ > 0) ---
    # 15 транзакций примерно по 20$ ± 5$, чтобы σ ≈ 5.
    for i in range(15):
        amt = max(1.0, round(rng.gauss(20.0, 5.0), 2))
        await eng.process(make_tx("USER_A", base + timedelta(hours=i), amount=amt))

    # 2. Аномальная сумма ($5000 при mean≈20, σ≈5 → z≈1000) → Z-score должен сработать
    res = await eng.process(make_tx("USER_A", base + timedelta(hours=16), amount=5000.0))
    if "Z_SCORE_AMOUNT_ANOMALY" not in res.triggered_rules:
        failures.append(
            f"Z-score не сработал: scores={res.rule_scores}, "
            f"raw={res.rule_raw_values}"
        )

    # --- 3. Geo-velocity: уникальный клиент, две GPS-точки за 10 минут ---
    await eng.process(make_tx("USER_B", base, amount=20.0, lat=41.31, lon=69.24))
    res = await eng.process(make_tx("USER_B", base + timedelta(minutes=10),
                                    amount=20.0, lat=51.5, lon=-0.12))
    if "GEO_VELOCITY_IMPOSSIBLE_TRAVEL" not in res.triggered_rules:
        failures.append(f"Geo-velocity не сработал: scores={res.rule_scores}")

    # --- 4. Dropper network: 3 уникальных отправителя на одного получателя ---
    last_drop_res = None
    for sender in ("S1", "S2", "S3"):
        last_drop_res = await eng.process(
            make_tx(sender, base + timedelta(seconds=10 * len(sender)),
                    amount=50.0, mcc=None, receiver="DROP",
                    purpose=TransferPurpose.PURCHASE)
        )
    if "GRAPH_DROPPER_NETWORK" not in last_drop_res.triggered_rules:
        failures.append(f"Dropper network не сработал: scores={last_drop_res.rule_scores}")

    # --- 5. Benford: искусственно перекошенное распределение ---
    eng2, _, _ = build_engine([BenfordsLawRule()])
    # 30 транзакций, каждая начинается с цифры 9 → χ² ≫ порога
    for i in range(30):
        await eng2.process(make_tx("USER_C", base + timedelta(hours=i), amount=900.0 + i))
    res = await eng2.process(make_tx("USER_C", base + timedelta(hours=31), amount=950.0))
    if "BENFORD_LAW_ANOMALY" not in res.triggered_rules:
        failures.append(
            f"Benford не сработал: scores={res.rule_scores}, raw={res.rule_raw_values}"
        )

    # --- 6. Welford: после разнообразной истории дисперсия > 0 ---
    state_a = await repo.get_profile("USER_A")
    mcc = state_a.mcc_stats.get("5411")
    if not mcc or mcc.count < 2:
        failures.append(f"USER_A: mcc-история не накопилась ({mcc})")
    elif mcc.m2 <= 0:
        failures.append(f"Welford m2 = 0 при разнообразной истории — не должно быть")
    else:
        unbiased = mcc.m2 / (mcc.count - 1)
        biased = mcc.m2 / mcc.count
        # Бессель-поправка должна быть видна на n=15.
        if abs(unbiased - biased) < 1e-6:
            failures.append(
                f"Bias correction не работает: unbiased={unbiased}, biased={biased}"
            )

    # --- 7. Граф дедуплицирует device-edges ---
    summary = graph.summary()
    print(f"Graph summary: {summary}")
    if summary["edge_types"].get("uses_device", 0) == 0:
        failures.append("Граф не зафиксировал ни одного device-ребра")

    # --- 8. Дубль uses_device для одного клиента не должен расти ---
    eng3, _, graph3 = build_engine([])
    for i in range(20):
        await eng3.process(make_tx("USER_D", base + timedelta(hours=i),
                                   amount=10.0 + i, dev="same-dev", ip="1.1.1.1"))
    s3 = graph3.summary()
    if s3["edge_types"].get("uses_device", 0) > 1:
        failures.append(
            f"Дедупликация устройств сломана: {s3['edge_types']['uses_device']} рёбер"
        )

    # --- Финальный вердикт ---
    print("\n" + "=" * 60)
    if failures:
        print(f"❌ FAIL ({len(failures)}):")
        for f in failures:
            print(f"   - {f}")
        return 1
    print("✅ Все smoke-проверки прошли")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
