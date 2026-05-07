from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import TransactionPayload, FraudDecision, ClientProfileState
from app.engine import FraudScoringEngine
from app.repositories import InMemoryProfileRepository
from app.rules import (
    ZScoreAmountRule, GeoVelocityRule, PoissonVelocityRule,
    CircularTimeRule, TransferPurposeAnomalyRule, DailyVolumeSpikeRule,
    BenfordsLawRule,
    GraphDropperNetworkRule, GraphCycleRule, SharedAttributeRule,
    PageRankAnomalyRule, FanOutSmurfingRule, ReceiverVelocityRule,
)
from app.graph import TransactionGraph

app = FastAPI(
    title="Stat-Guard Anti-Fraud Engine",
    description="Система выявления мошеннических транзакций на основе статистических методов и графового анализа",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = InMemoryProfileRepository()
graph_db = TransactionGraph()

active_rules = [
    # --- Статистические правила ---
    ZScoreAmountRule(),
    GeoVelocityRule(),
    PoissonVelocityRule(),
    CircularTimeRule(),
    DailyVolumeSpikeRule(),
    TransferPurposeAnomalyRule(),
    BenfordsLawRule(),
    # --- Графовые правила ---
    GraphDropperNetworkRule(),
    ReceiverVelocityRule(),
    GraphCycleRule(),
    SharedAttributeRule(),
    PageRankAnomalyRule(),
    FanOutSmurfingRule(),
]

engine = FraudScoringEngine(repo=db, rules=active_rules, graph=graph_db)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "rules_loaded": len(active_rules),
        "ml_model_loaded": engine.model is not None,
    }


@app.post("/api/v1/analyze", response_model=FraudDecision)
async def analyze_transaction(tx: TransactionPayload):
    return await engine.process(tx)


@app.get("/api/v1/profile/{client_id}", response_model=ClientProfileState)
async def get_profile(client_id: str):
    return await db.get_profile(client_id)


@app.get("/api/v1/stats")
async def get_engine_stats():
    return {
        "profiles_in_memory": len(db._storage),
        "graph": graph_db.summary(),
    }