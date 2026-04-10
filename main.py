from fastapi import FastAPI
from schemas import TransactionPayload, FraudDecision, ClientProfileState
from engine import InMemoryProfileRepository, FraudScoringEngine
from rules import (
    ZScoreAmountRule, GeoVelocityRule, PoissonVelocityRule, 
    CircularTimeRule, BenfordLawRule, TransferPurposeAnomalyRule, DailyVolumeSpikeRule
)

app = FastAPI(title="Stat-Guard Engine")

repository = InMemoryProfileRepository()
active_rules = [
    ZScoreAmountRule(),
    GeoVelocityRule(),
    PoissonVelocityRule(),
    CircularTimeRule(),
    BenfordLawRule(),
    DailyVolumeSpikeRule(),
    TransferPurposeAnomalyRule()
]
engine = FraudScoringEngine(repo=repository, rules=active_rules)

@app.post("/api/v1/analyze", response_model=FraudDecision)
async def analyze_transaction(tx: TransactionPayload):
    return await engine.process(tx)

@app.get("/api/v1/profile/{client_id}", response_model=ClientProfileState)
async def get_profile(client_id: str):
    return await repository.get(client_id)