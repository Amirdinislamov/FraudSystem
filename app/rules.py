import math
from abc import ABC, abstractmethod
from app.schemas import TransactionPayload, ClientProfileState, RuleResult, TransferPurpose
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.graph import TransactionGraph


class BaseFraudRule(ABC):
    @property
    @abstractmethod
    def rule_name(self) -> str:
        pass

    @abstractmethod
    async def evaluate(
        self,
        tx: TransactionPayload,
        state: ClientProfileState,
        receiver_state: Optional[ClientProfileState] = None,
        graph: Optional["TransactionGraph"] = None,
    ) -> RuleResult:
        pass


class ZScoreAmountRule(BaseFraudRule):
    """
    Z-score аномалия суммы транзакции относительно истории клиента по MCC-коду.
    Онлайн-алгоритм Велфорда для μ и σ. Срабатывает при z > 3σ.
    """

    @property
    def rule_name(self) -> str:
        return "Z_SCORE_AMOUNT_ANOMALY"

    async def evaluate(self, tx, state, receiver_state=None, graph=None) -> RuleResult:
        mcc_stats = state.mcc_stats.get(tx.mcc_code)
        if not mcc_stats or mcc_stats.count < 3:
            return RuleResult(rule_name=self.rule_name, score=0, raw_value=0.0)

        variance = mcc_stats.m2 / (mcc_stats.count - 1)
        sigma = math.sqrt(variance) if variance > 0 else 0.0
        if sigma == 0:
            return RuleResult(rule_name=self.rule_name, score=0, raw_value=0.0)

        z_score = abs(tx.amount_usd - mcc_stats.mean) / sigma
        penalty = min(int((z_score - 3) * 12), 50) if z_score > 3.0 else 0
        reason = f"z={z_score:.2f}σ (mean=${mcc_stats.mean:.0f}, σ=${sigma:.0f})" if penalty else None
        return RuleResult(rule_name=self.rule_name, score=penalty, raw_value=float(z_score), reason=reason)


class GeoVelocityRule(BaseFraudRule):
    """
    Невозможная геоскорость. Формула Гаверсинуса для расстояния.
    Срабатывает при speed > 1000 км/ч или dist > 1000 км.
    """

    @property
    def rule_name(self) -> str:
        return "GEO_VELOCITY_IMPOSSIBLE_TRAVEL"

    async def evaluate(self, tx, state, receiver_state=None, graph=None) -> RuleResult:
        if (
            state.last_geo_lat is None
            or tx.terminal_lat is None
            or state.last_geo_timestamp is None
        ):
            return RuleResult(rule_name=self.rule_name, score=0, raw_value=0.0)

        R = 6371.0
        phi1 = math.radians(state.last_geo_lat)
        phi2 = math.radians(tx.terminal_lat)
        dphi = math.radians(tx.terminal_lat - state.last_geo_lat)
        dlam = math.radians(tx.terminal_lon - state.last_geo_lon)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        dist_km = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        elapsed_sec = (tx.timestamp - state.last_geo_timestamp).total_seconds()
        hours = elapsed_sec / 3600.0
        speed_kmh = dist_km / hours if hours > 0 else dist_km * 9999

        if speed_kmh > 1000:
            return RuleResult(
                rule_name=self.rule_name, score=100, raw_value=speed_kmh,
                reason=f"Impossible: {speed_kmh:.0f} km/h over {dist_km:.0f} km"
            )
        if dist_km > 1000:
            return RuleResult(
                rule_name=self.rule_name, score=85, raw_value=speed_kmh,
                reason=f"Suspicious distance: {dist_km:.0f} km"
            )
        return RuleResult(rule_name=self.rule_name, score=0, raw_value=speed_kmh)


class PoissonVelocityRule(BaseFraudRule):
    """
    Пуассоновская аномалия частоты транзакций.
    λ оценивается по всей истории клиента. Если P(k|λ) < 0.001 — аномалия.
    """

    @property
    def rule_name(self) -> str:
        return "POISSON_VELOCITY_ANOMALY"

    async def evaluate(self, tx, state, receiver_state=None, graph=None) -> RuleResult:
        if state.total_tx_count < 5 or state.recent_tx_hour_count < 4 or state.first_tx_timestamp is None:
            return RuleResult(rule_name=self.rule_name, score=0, raw_value=1.0)

        total_hours = (tx.timestamp - state.first_tx_timestamp).total_seconds() / 3600.0
        lam = state.total_tx_count / max(total_hours, 1)
        k = min(state.recent_tx_hour_count, 20)

        try:
            prob = (math.pow(lam, k) * math.exp(-lam)) / math.factorial(k)
        except (OverflowError, ValueError):
            prob = 0.0

        score = 85 if prob < 0.001 else 0
        reason = f"P(k={k} | λ={lam:.2f}) = {prob:.6f}" if score else None
        return RuleResult(rule_name=self.rule_name, score=score, raw_value=prob, reason=reason)


class CircularTimeRule(BaseFraudRule):
    """
    Аномалия времени суток на основе кольцевой (circular) статистики.
    μ = atan2(Σsin(θ), Σcos(θ)). Отклонение > 6 часов — аномалия.
    """

    @property
    def rule_name(self) -> str:
        return "CIRCULAR_TIME_ANOMALY"

    async def evaluate(self, tx, state, receiver_state=None, graph=None) -> RuleResult:
        if state.total_tx_count < 5 or state.mean_time_angle is None:
            return RuleResult(rule_name=self.rule_name, score=0, raw_value=0.0)

        angle = ((tx.timestamp.hour + tx.timestamp.minute / 60.0) / 24.0) * 2 * math.pi
        diff = math.atan2(math.sin(angle - state.mean_time_angle), math.cos(angle - state.mean_time_angle))
        diff_hours = abs(diff * 24.0 / (2 * math.pi))

        score = 15 if diff_hours > 6 else 0
        reason = f"Time deviation: {diff_hours:.1f}h from usual pattern" if score else None
        return RuleResult(rule_name=self.rule_name, score=score, raw_value=diff_hours, reason=reason)


class TransferPurposeAnomalyRule(BaseFraudRule):
    """
    Аномалия цели перевода: высокорисковые категории, первое использование,
    крупные суммы при INVESTMENT.
    """

    @property
    def rule_name(self) -> str:
        return "TRANSFER_PURPOSE_ANOMALY"

    async def evaluate(self, tx, state, receiver_state=None, graph=None) -> RuleResult:
        if not tx.transfer_purpose:
            return RuleResult(rule_name=self.rule_name, score=0, raw_value=0.0)

        purpose = tx.transfer_purpose
        score = 0
        reasons = []

        if purpose in [TransferPurpose.INVESTMENT, TransferPurpose.CHARITY]:
            score += 25
            reasons.append(f"High-risk category: {purpose.value}")

        if state.total_tx_count > 5 and purpose.value not in state.used_transfer_purposes:
            score += 30
            reasons.append("First-time purpose usage")

        if purpose == TransferPurpose.INVESTMENT and float(tx.amount_usd) > 1000:
            score += 40
            reasons.append(f"Large investment: ${tx.amount_usd:.0f}")

        return RuleResult(
            rule_name=self.rule_name,
            score=min(score, 100),
            raw_value=float(score),
            reason=" | ".join(reasons) if reasons else None,
        )


class DailyVolumeSpikeRule(BaseFraudRule):
    """
    Превышение дневного лимита расходов > $3000. Прогрессивный штраф.
    """

    @property
    def rule_name(self) -> str:
        return "DAILY_VOLUME_SPIKE"

    async def evaluate(self, tx, state, receiver_state=None, graph=None) -> RuleResult:
        current_date = tx.timestamp.strftime("%Y-%m-%d")
        spent_today = state.spent_last_24h if state.last_day_date == current_date else 0.0
        new_total = spent_today + float(tx.amount_usd)

        score, reason = 0, None
        if new_total > 3000.0:
            if spent_today <= 3000.0:
                score = 60
                reason = f"Daily limit breached: ${new_total:.0f} (limit $3000)"
            else:
                score = 85
                reason = f"Continued over-limit spending: ${new_total:.0f}"

        return RuleResult(rule_name=self.rule_name, score=score, raw_value=new_total, reason=reason)


class BenfordsLawRule(BaseFraudRule):
    """
    Закон Бенфорда. Распределение первой значащей цифры в естественных
    финансовых данных подчиняется P(d) = log10(1 + 1/d).

    Гипотеза: фабрикованные / синтетические транзакции (например, мошеннические
    выводы со взломанного аккаунта или искусственные «суммы» при отмывании)
    сильно отклоняются от закона Бенфорда.

    Метрика — статистика хи-квадрат: χ² = Σ (O_i − E_i)² / E_i.
    Для df=8 и α=0.01 критическое значение ≈ 20.09.
    Срабатывает только при достаточной истории (≥ 25 транзакций),
    иначе оценка ненадёжна.
    """


    EXPECTED = [math.log10(1 + 1.0 / d) for d in range(1, 10)]
    MIN_HISTORY = 25
    THRESHOLD_CHI2 = 20.09  

    @property
    def rule_name(self) -> str:
        return "BENFORD_LAW_ANOMALY"

    async def evaluate(self, tx, state, receiver_state=None, graph=None) -> RuleResult:
        counts = state.first_digit_counts
        n = sum(counts)
        if n < self.MIN_HISTORY:
            return RuleResult(rule_name=self.rule_name, score=0, raw_value=0.0)

        chi2 = 0.0
        for i in range(9):
            expected = self.EXPECTED[i] * n
            if expected > 0:
                chi2 += (counts[i] - expected) ** 2 / expected

        score = 0
        if chi2 > self.THRESHOLD_CHI2:
            score = min(int((chi2 - self.THRESHOLD_CHI2) * 1.5) + 30, 60)
        reason = f"χ²={chi2:.1f} (n={n}, threshold={self.THRESHOLD_CHI2})" if score else None
        return RuleResult(rule_name=self.rule_name, score=score, raw_value=float(chi2), reason=reason)


class GraphDropperNetworkRule(BaseFraudRule):
    """
    [ГРАФ] Детекция дропперов — hub-and-spoke паттерн.

    Прогрессивный скоринг по числу уникальных отправителей к получателю:
        2 senders  → 35 баллов (MANUAL_REVIEW)  — ранний сигнал
        3 senders  → 70 баллов (DECLINED)        — типичный hub
        4+ senders → 90 баллов                   — устоявшийся дроппер

    Бинарный порог ≥3 на синтетическом датасете давал recall ≈ 39%
    на «organic» переводах к дропперам, потому что первые 1–2 жертвы
    проскакивают как обычный P2P. Прогрессивная шкала ловит раньше,
    почти не повышая false positives, т.к. для обычных получателей
    «двух разных отправителей за всю историю» — редкий случай.
    """

    @property
    def rule_name(self) -> str:
        return "GRAPH_DROPPER_NETWORK"

    async def evaluate(self, tx, state, receiver_state=None, graph=None) -> RuleResult:
        if not tx.receiver_id or not graph:
            return RuleResult(rule_name=self.rule_name, score=0, raw_value=0.0)

        unique_senders = graph.get_in_degree(tx.receiver_id)
        if not graph.G.has_edge(tx.client_id, tx.receiver_id):
            unique_senders += 1

        if unique_senders >= 4:
            score = 90
        elif unique_senders == 3:
            score = 70
        elif unique_senders == 2:
            score = 35
        else:
            score = 0

        reason = f"Dropper hub: {unique_senders} unique senders" if score else None
        return RuleResult(rule_name=self.rule_name, score=score, raw_value=float(unique_senders), reason=reason)


class ReceiverVelocityRule(BaseFraudRule):
    """
    [ГРАФ] Бурст входящих переводов на получателя за короткое окно.

    Если на одного получателя за < 1 часа поступают переводы от ≥ 3
    разных отправителей — это типичный признак активного дроппера в
    момент атаки (синхронный вывод средств с скомпрометированных
    аккаунтов). В отличие от GraphDropperNetworkRule, который смотрит
    на всю историю, это правило фиксирует моментальный всплеск.
    """

    WINDOW_HOURS = 1

    @property
    def rule_name(self) -> str:
        return "GRAPH_RECEIVER_VELOCITY"

    async def evaluate(self, tx, state, receiver_state=None, graph=None) -> RuleResult:
        if not tx.receiver_id or not graph:
            return RuleResult(rule_name=self.rule_name, score=0, raw_value=0.0)

        recent = graph.get_recent_unique_senders(
            tx.receiver_id, current_time=tx.timestamp, window_hours=self.WINDOW_HOURS
        )
        if not graph.G.has_edge(tx.client_id, tx.receiver_id):
            recent += 1

        if recent >= 5:
            score = 80
        elif recent >= 3:
            score = 60
        else:
            score = 0
        reason = (
            f"Receiver velocity: {recent} unique senders in last {self.WINDOW_HOURS}h"
            if score else None
        )
        return RuleResult(rule_name=self.rule_name, score=score, raw_value=float(recent), reason=reason)


class GraphCycleRule(BaseFraudRule):
    """
    [ГРАФ] Детекция циклических переводов (Smurfing / Money Laundering).
    Путь receiver→sender длиной ≤ 5 хопов = замкнутый цикл.
    """

    @property
    def rule_name(self) -> str:
        return "GRAPH_CYCLE_DETECTED"

    async def evaluate(self, tx, state, receiver_state=None, graph=None) -> RuleResult:
        if not tx.receiver_id or not graph:
            return RuleResult(rule_name=self.rule_name, score=0, raw_value=0.0)

        has_cycle = graph.detect_cycle(source_id=tx.client_id, target_id=tx.receiver_id)
        score = 100 if has_cycle else 0
        reason = "Cyclic transfer detected (Smurfing/AML pattern)" if has_cycle else None
        return RuleResult(rule_name=self.rule_name, score=score, raw_value=float(has_cycle), reason=reason)


class SharedAttributeRule(BaseFraudRule):
    """
    [ГРАФ] Детекция бот-ферм через общие атрибуты (device_id / ip_address).
    ≥4 пользователей на один атрибут → бот-ферма.
    """

    @property
    def rule_name(self) -> str:
        return "SHARED_ATTRIBUTE_ANOMALY"

    async def evaluate(self, tx, state, receiver_state=None, graph=None) -> RuleResult:
        if not graph:
            return RuleResult(rule_name=self.rule_name, score=0, raw_value=0.0)

        dev_count = graph.get_shared_device_count(tx.device_id) if tx.device_id else 0
        ip_count = graph.get_shared_ip_count(tx.ip_address) if tx.ip_address else 0

        if tx.device_id and not graph.G.has_edge(tx.client_id, f"DEV_{tx.device_id}"):
            dev_count += 1
        if tx.ip_address and not graph.G.has_edge(tx.client_id, f"IP_{tx.ip_address}"):
            ip_count += 1

        max_shared = max(dev_count, ip_count)
        score, reason = 0, None
        if max_shared >= 4:
            score = 95
            reason = f"Bot farm: device/IP shared by {max_shared} users"
        elif max_shared >= 2:
            score = 40
            reason = f"Shared attribute: {max_shared} users on same device/IP"

        return RuleResult(rule_name=self.rule_name, score=score, raw_value=float(max_shared), reason=reason)


class PageRankAnomalyRule(BaseFraudRule):
    """
    [ГРАФ] PageRank-аномалия получателя в сети денежных переводов.

    PageRank (Brin & Page, 1998) оценивает «влиятельность» узла в направленном графе:
    узел важен, если на него ссылаются другие важные узлы.

    В финансовом контексте высокий PageRank получателя означает,
    что он аккумулирует средства из «влиятельных» частей сети —
    типичный признак дроппера в ATO/AML схемах.

    Нормализация: normalized_pr = pr(node) * N, где N — число узлов.
    Среднее нормализованного = 1.0. Порог: > 10x выше среднего → 80 баллов.
    """

    @property
    def rule_name(self) -> str:
        return "GRAPH_PAGERANK_ANOMALY"

    async def evaluate(self, tx, state, receiver_state=None, graph=None) -> RuleResult:
        if not tx.receiver_id or not graph:
            return RuleResult(rule_name=self.rule_name, score=0, raw_value=0.0)

        graph._refresh_pagerank_if_needed(tx.timestamp)

        if not graph._pagerank_cache:
            return RuleResult(rule_name=self.rule_name, score=0, raw_value=0.0)

        pr_value = graph.get_pagerank_score(tx.receiver_id)
        n_nodes = len(graph._pagerank_cache)
        mean_pr = 1.0 / n_nodes if n_nodes > 0 else 1.0
        normalized_pr = pr_value / mean_pr if mean_pr > 0 else 0.0

        score, reason = 0, None
        if normalized_pr > 10:
            score = 80
            reason = f"PageRank hub: {normalized_pr:.1f}x above network average"
        elif normalized_pr > 5:
            score = 40
            reason = f"Elevated PageRank: {normalized_pr:.1f}x above average"

        return RuleResult(rule_name=self.rule_name, score=score, raw_value=float(normalized_pr), reason=reason)


class FanOutSmurfingRule(BaseFraudRule):
    """
    [ГРАФ] Детекция смёрфинга через fan-out паттерн (out-degree аномалия).

    Смёрфинг (structuring) — дробление крупной суммы на множество мелких переводов
    разным получателям для обхода AML-лимитов.

    Признак: клиент переводит деньги большому числу уникальных получателей.
    Метрика: out-degree клиента в transfer-подграфе.
    Порог: ≥5 уникальных получателей → 70 баллов, ≥8 → 85 баллов.
    """

    @property
    def rule_name(self) -> str:
        return "GRAPH_FAN_OUT_SMURFING"

    async def evaluate(self, tx, state, receiver_state=None, graph=None) -> RuleResult:
        if not tx.receiver_id or not graph:
            return RuleResult(rule_name=self.rule_name, score=0, raw_value=0.0)

        current_fan_out = graph.get_fan_out(tx.client_id)
        tg = graph._get_transfer_graph()
        is_new_receiver = not (
            tg.has_node(tx.client_id)
            and tg.has_node(tx.receiver_id)
            and tg.has_successor(tx.client_id, tx.receiver_id)
        )
        projected_fan_out = current_fan_out + (1 if is_new_receiver else 0)

        score, reason = 0, None
        if projected_fan_out >= 8:
            score = 85
            reason = f"Extreme fan-out: {projected_fan_out} unique recipients"
        elif projected_fan_out >= 5:
            score = 70
            reason = f"Smurfing pattern: {projected_fan_out} unique recipients"

        return RuleResult(
            rule_name=self.rule_name,
            score=score,
            raw_value=float(projected_fan_out),
            reason=reason,
        )