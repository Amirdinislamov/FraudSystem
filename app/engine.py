import math
import pickle 
import numpy as np 
from typing import List
from app.interfaces import IProfileRepository  
from app.schemas import TransactionPayload, ClientProfileState, FraudDecision, DecisionStatus, MccStats
from app.rules import BaseFraudRule

class FraudScoringEngine:
    def __init__(self, repo: IProfileRepository, rules: List[BaseFraudRule]):
        self.repo = repo
        self.rules = rules

        try:
            with open("fraud_ml_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            print("🧠 ИИ-модель успешно загружена в движок!")
        except FileNotFoundError:
            print("⚠️ ВНИМАНИЕ: Файл fraud_ml_model.pkl не найден.")
            self.model = None

    async def process(self, tx: TransactionPayload) -> FraudDecision:
        state = await self.repo.get_profile(tx.client_id)
        receiver_state = await self.repo.get_profile(tx.receiver_id) if tx.receiver_id else None
        
        total_score = 0
        triggered = []
        rule_scores = {}
        features_for_ml = [] 
        
        for rule in self.rules:
            res = await rule.evaluate(tx, state, receiver_state)
            
            rule_scores[rule.rule_name] = res.score
            features_for_ml.append(res.score) 
            
            if res.score > 0:
                total_score += res.score
                triggered.append(rule.rule_name)

        if self.model:
            X_input = np.array(features_for_ml).reshape(1, -1)
            
            proba = self.model.predict_proba(X_input)[0]
            if 1 in self.model.classes_:
                fraud_index = list(self.model.classes_).index(1)
                fraud_prob = proba[fraud_index]
            else:
                fraud_prob = 0.0 
            
            final_score = int(fraud_prob * 100)
        else:
            final_score = min(total_score, 100)

        decision = DecisionStatus.APPROVED
        if final_score >= 80: decision = DecisionStatus.DECLINED
        elif final_score >= 40: decision = DecisionStatus.MANUAL_REVIEW

        if decision == DecisionStatus.APPROVED:
            self._update_state(state, tx)
            await self.repo.save_profile(state)
            
            if receiver_state:
                if not isinstance(receiver_state.incoming_senders, set):
                    receiver_state.incoming_senders = set(receiver_state.incoming_senders or [])
                receiver_state.incoming_senders.add(tx.client_id)
                await self.repo.save_profile(receiver_state)

        return FraudDecision(
            transaction_id=tx.transaction_id, 
            decision=decision, 
            risk_score=final_score, 
            triggered_rules=triggered,
            rule_scores=rule_scores
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
        
        if tx.terminal_lat and tx.terminal_lon:
            state.last_geo_lat = tx.terminal_lat
            state.last_geo_lon = tx.terminal_lon
            
        current_date = tx.timestamp.strftime("%Y-%m-%d")
        if state.last_day_date == current_date:
            state.spent_last_24h += float(tx.amount_usd)
        else:
            state.spent_last_24h = float(tx.amount_usd)
            state.last_day_date = current_date

        angle = ((tx.timestamp.hour + tx.timestamp.minute / 60.0) / 24.0) * 2 * math.pi
        state.time_sin_sum += math.sin(angle)
        state.time_cos_sum += math.cos(angle)
        
        mcc_stat = state.mcc_stats.setdefault(tx.mcc_code, MccStats())
        mcc_stat.count += 1
        delta = tx.amount_usd - mcc_stat.mean
        mcc_stat.mean += delta / mcc_stat.count
        delta2 = tx.amount_usd - mcc_stat.mean
        mcc_stat.m2 += delta * delta2

        if tx.transfer_purpose:
            state.used_transfer_purposes.add(tx.transfer_purpose.value)