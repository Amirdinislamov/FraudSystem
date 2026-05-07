import math
import pickle
import numpy as np
from typing import List
from app.interfaces import IProfileRepository
from app.schemas import (
    TransactionPayload, ClientProfileState, FraudDecision,
    DecisionStatus, MccStats
)
from app.rules import BaseFraudRule
from app.graph import TransactionGraph


class FraudScoringEngine:
    def __init__(self, repo: IProfileRepository, rules: List[BaseFraudRule], graph: TransactionGraph = None):
        self.repo = repo
        self.rules = rules
        self.graph = graph or TransactionGraph()
        self.model = None
        self.feature_names: List[str] = []

        try:
            with open("fraud_ml_model.pkl", "rb") as f:
                saved = pickle.load(f)
                if isinstance(saved, dict):
                    self.model = saved["model"]
                    self.feature_names = saved["feature_names"]
                else:
                    self.model = saved
                    self.feature_names = []
            print("🧠 ML-модель успешно загружена!")
        except FileNotFoundError:
            print("⚠️  fraud_ml_model.pkl не найден — используется эвристический скоринг.")

    async def process(self, tx: TransactionPayload) -> FraudDecision:
        state = await self.repo.get_profile(tx.client_id)
        receiver_state = await self.repo.get_profile(tx.receiver_id) if tx.receiver_id else None

        total_heuristic_score = 0
        triggered = []
        rule_scores: dict[str, int] = {}
        rule_raw_values: dict[str, float] = {}

        for rule in self.rules:
            res = await rule.evaluate(tx, state, receiver_state, self.graph)
            rule_scores[rule.rule_name] = res.score
            rule_raw_values[rule.rule_name] = res.raw_value

            if res.score > 0:
                total_heuristic_score += res.score
                triggered.append(rule.rule_name)

        if self.model is not None:
            if self.feature_names:
                feature_vector = [rule_raw_values.get(name, 0.0) for name in self.feature_names]
            else:
                feature_vector = [rule_raw_values[r.rule_name] for r in self.rules]

            X_input = np.array(feature_vector).reshape(1, -1)
            proba = self.model.predict_proba(X_input)[0]

            if 1 in self.model.classes_:
                fraud_index = list(self.model.classes_).index(1)
                fraud_prob = proba[fraud_index]
            else:
                fraud_prob = 0.0

            final_score = int(fraud_prob * 100)
        else:
            final_score = min(total_heuristic_score, 100)

        decision = DecisionStatus.APPROVED
        if final_score >= 80:
            decision = DecisionStatus.DECLINED
        elif final_score >= 40:
            decision = DecisionStatus.MANUAL_REVIEW

        if decision == DecisionStatus.APPROVED:
            self._update_state(state, tx)
            await self.repo.save_profile(state)

        self.graph.add_transaction(tx)

        return FraudDecision(
            transaction_id=tx.transaction_id,
            decision=decision,
            risk_score=final_score,
            triggered_rules=triggered,
            rule_scores=rule_scores,
            rule_raw_values=rule_raw_values,
        )

    def _update_state(self, state: ClientProfileState, tx: TransactionPayload):
        if state.total_tx_count == 0:
            state.first_tx_timestamp = tx.timestamp

        state.total_tx_count += 1

        if state.last_tx_timestamp and (tx.timestamp - state.last_tx_timestamp).total_seconds() > 3600:
            state.recent_tx_hour_count = 1
        else:
            state.recent_tx_hour_count += 1

        state.last_tx_timestamp = tx.timestamp

        if tx.terminal_lat is not None and tx.terminal_lon is not None:
            state.last_geo_lat = tx.terminal_lat
            state.last_geo_lon = tx.terminal_lon
            state.last_geo_timestamp = tx.timestamp

        current_date = tx.timestamp.strftime("%Y-%m-%d")
        if state.last_day_date == current_date:
            state.spent_last_24h += float(tx.amount_usd)
        else:
            state.spent_last_24h = float(tx.amount_usd)
            state.last_day_date = current_date

        angle = ((tx.timestamp.hour + tx.timestamp.minute / 60.0) / 24.0) * 2 * math.pi
        state.time_sin_sum += math.sin(angle)
        state.time_cos_sum += math.cos(angle)

        if tx.mcc_code:
            mcc_stat = state.mcc_stats.setdefault(tx.mcc_code, MccStats())
            mcc_stat.count += 1
            delta = tx.amount_usd - mcc_stat.mean
            mcc_stat.mean += delta / mcc_stat.count
            delta2 = tx.amount_usd - mcc_stat.mean
            mcc_stat.m2 += delta * delta2

        if tx.transfer_purpose:
            state.used_transfer_purposes.add(tx.transfer_purpose.value)

        amt = float(tx.amount_usd)
        if amt > 0:
            first = int(str(amt).lstrip("0.").lstrip(".")[0]) if amt < 1 else int(str(int(amt))[0])
            if 1 <= first <= 9:
                state.first_digit_counts[first - 1] += 1