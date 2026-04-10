import math
from abc import ABC, abstractmethod
from app.schemas import TransactionPayload, ClientProfileState, RuleResult, TransferPurpose
from typing import Optional

class BaseFraudRule(ABC):
    @property
    @abstractmethod
    def rule_name(self) -> str: pass

    @abstractmethod
    async def evaluate(self, tx: TransactionPayload, state: ClientProfileState, receiver_state: Optional[ClientProfileState] = None) -> RuleResult: pass

class ZScoreAmountRule(BaseFraudRule):
    @property
    def rule_name(self) -> str: return "Z_SCORE_AMOUNT_ANOMALY"
    async def evaluate(self, tx: TransactionPayload, state: ClientProfileState, receiver_state=None) -> RuleResult:
        mcc_stats = state.mcc_stats.get(tx.mcc_code)
        if not mcc_stats or mcc_stats.count < 3: return RuleResult(rule_name=self.rule_name, score=0)
        sigma = math.sqrt(mcc_stats.m2 / mcc_stats.count)
        if sigma == 0: return RuleResult(rule_name=self.rule_name, score=0)
        z_score = abs(tx.amount_usd - mcc_stats.mean) / sigma
        penalty = min(int((z_score - 3) * 12), 50) if z_score > 3.0 else 0
        return RuleResult(rule_name=self.rule_name, score=penalty)

class GeoVelocityRule(BaseFraudRule):
    @property
    def rule_name(self) -> str: return "GEO_VELOCITY_IMPOSSIBLE_TRAVEL"
    async def evaluate(self, tx: TransactionPayload, state: ClientProfileState, receiver_state=None) -> RuleResult:
        if not state.last_geo_lat or not tx.terminal_lat: return RuleResult(rule_name=self.rule_name, score=0)
        R = 6371.0
        phi1, phi2 = math.radians(state.last_geo_lat), math.radians(tx.terminal_lat)
        dphi = math.radians(tx.terminal_lat - state.last_geo_lat)
        dlam = math.radians(tx.terminal_lon - state.last_geo_lon)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
        dist = R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))
        hours = (tx.timestamp - state.last_tx_timestamp).total_seconds() / 3600.0
        if hours > 0 and (dist/hours) > 1000: return RuleResult(rule_name=self.rule_name, score=100)
        if dist > 1000: return RuleResult(rule_name=self.rule_name, score=85)
        return RuleResult(rule_name=self.rule_name, score=0)

class PoissonVelocityRule(BaseFraudRule):
    @property
    def rule_name(self) -> str: return "POISSON_VELOCITY_ANOMALY"
    async def evaluate(self, tx: TransactionPayload, state: ClientProfileState, receiver_state=None) -> RuleResult:
        if state.total_tx_count < 5 or state.recent_tx_hour_count < 4: return RuleResult(rule_name=self.rule_name, score=0)
        total_hours = (tx.timestamp - state.first_tx_timestamp).total_seconds() / 3600.0
        lam = state.total_tx_count / max(total_hours, 1)
        prob = (math.pow(lam, state.recent_tx_hour_count) * math.exp(-lam)) / math.factorial(min(state.recent_tx_hour_count, 20))
        return RuleResult(rule_name=self.rule_name, score=85) if prob < 0.001 else RuleResult(rule_name=self.rule_name, score=0)



class CircularTimeRule(BaseFraudRule):
    @property
    def rule_name(self) -> str: return "CIRCULAR_TIME_ANOMALY"

    async def evaluate(self, tx: TransactionPayload, state: ClientProfileState, receiver_state: Optional[ClientProfileState] = None) -> RuleResult:
        if state.total_tx_count < 5 or state.mean_time_angle is None:
            return RuleResult(rule_name=self.rule_name, score=0)
        
        angle = ((tx.timestamp.hour + tx.timestamp.minute / 60.0) / 24.0) * 2 * math.pi
        mean_angle = state.mean_time_angle 
        
        diff = math.atan2(math.sin(angle - mean_angle), math.cos(angle - mean_angle))
        diff_hours = abs(diff * 24.0 / (2 * math.pi))
        
        if diff_hours > 6:
            return RuleResult(rule_name=self.rule_name, score=15, reason=f"Time diff: {diff_hours:.1f}h")
        return RuleResult(rule_name=self.rule_name, score=0)


class BenfordLawRule(BaseFraudRule):
    @property
    def rule_name(self) -> str: return "BENFORD_LAW_ANOMALY"

    async def evaluate(self, tx: TransactionPayload, state: ClientProfileState, receiver_state: Optional[ClientProfileState] = None) -> RuleResult:
        digit_str = str(float(tx.amount_usd)).replace('0', '').replace('.', '')
        if not digit_str: return RuleResult(rule_name=self.rule_name, score=0)
            
        first_digit = int(digit_str[0])
        
        if first_digit >= 8 and tx.amount_usd % 10 == 0:
            return RuleResult(rule_name=self.rule_name, score=15, reason=f"Benford violation, digit {first_digit}")
        return RuleResult(rule_name=self.rule_name, score=0)

class TransferPurposeAnomalyRule(BaseFraudRule):
    @property
    def rule_name(self) -> str: 
        return "TRANSFER_PURPOSE_ANOMALY"

    async def evaluate(self, tx: TransactionPayload, state: ClientProfileState, receiver_state: Optional[ClientProfileState] = None) -> RuleResult:
        if not tx.transfer_purpose: return RuleResult(rule_name=self.rule_name, score=0)
        
        purpose = tx.transfer_purpose
        score = 0
        reasons = []

        if purpose in [TransferPurpose.INVESTMENT, TransferPurpose.CHARITY]:
            score += 25
            reasons.append(f"High-risk category")
            
        if state.total_tx_count > 5 and purpose.value not in state.used_transfer_purposes:
            score += 30
            reasons.append(f"First time use")

        if purpose == TransferPurpose.INVESTMENT and float(tx.amount_usd) > 1000:
            score += 40
            reasons.append("Large Investment sum")

        if score > 0:
            return RuleResult(rule_name=self.rule_name, score=min(score, 100), reason=" | ".join(reasons))
        return RuleResult(rule_name=self.rule_name, score=0)
    


class GraphDropperNetworkRule(BaseFraudRule):
    @property
    def rule_name(self) -> str: return "GRAPH_DROPPER_NETWORK"
    async def evaluate(self, tx: TransactionPayload, state: ClientProfileState, receiver_state: Optional[ClientProfileState] = None) -> RuleResult:
        if not receiver_state or tx.client_id in receiver_state.incoming_senders: return RuleResult(rule_name=self.rule_name, score=0)
        unique_senders = len(receiver_state.incoming_senders)
        if unique_senders >= 3: return RuleResult(rule_name=self.rule_name, score=90, reason=f"Dropper hub: {unique_senders} senders")
        return RuleResult(rule_name=self.rule_name, score=0)
    

class DailyVolumeSpikeRule(BaseFraudRule):
    @property
    def rule_name(self) -> str: return "DAILY_VOLUME_SPIKE"

    async def evaluate(self, tx: TransactionPayload, state: ClientProfileState, receiver_state: Optional[ClientProfileState] = None) -> RuleResult:
        current_date = tx.timestamp.strftime("%Y-%m-%d")
        
        spent_today = state.spent_last_24h if state.last_day_date == current_date else 0.0
        
        new_spent_today = spent_today + float(tx.amount_usd)

        if new_spent_today > 3000.0:
            if spent_today <= 3000.0:
                return RuleResult(rule_name=self.rule_name, score=60, reason=f"Daily volume limit breached: ${new_spent_today:.0f}")
            else:
                return RuleResult(rule_name=self.rule_name, score=85, reason="Continued spending after daily limit breach")
                
        return RuleResult(rule_name=self.rule_name, score=0)