from fastapi import FastAPI
from app.schemas import TransactionPayload, FraudDecision, ClientProfileState
from app.engine import FraudScoringEngine
from app.repositories import InMemoryProfileRepository
from app.rules import (
    ZScoreAmountRule, GeoVelocityRule, PoissonVelocityRule, 
    CircularTimeRule, BenfordLawRule, TransferPurposeAnomalyRule, DailyVolumeSpikeRule
)

app = FastAPI(title="Stat-Guard Engine")

db = InMemoryProfileRepository()
active_rules = [
    ZScoreAmountRule(),
    GeoVelocityRule(),
    PoissonVelocityRule(),
    CircularTimeRule(),
    BenfordLawRule(),
    DailyVolumeSpikeRule(),
    TransferPurposeAnomalyRule()
]

engine = FraudScoringEngine(repo=db, rules=active_rules)

@app.post("/api/v1/analyze", response_model=FraudDecision)
async def analyze_transaction(tx: TransactionPayload):
    return await engine.process(tx)

@app.get("/api/v1/profile/{client_id}", response_model=ClientProfileState)
async def get_profile(client_id: str):
    return await db.get(client_id)