from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Simulation API",
    version="1.0.0",
    description="API unificada para simulaciones Monte Carlo, SDE, dinámicas híbridas, telco y logística.",
)

# ============================================================
# ROOT / HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Simulation API running",
        "endpoints": {
            "audit": "/audit/simulate",
            "risk": "/risk/simulate",
            "hybrid": "/hybrid/simulate",
            "telco": "/telco/simulate",
            "logistics": "/logistics/simulate",
        },
    }

@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# 1) AUDIT / MONTE CARLO
# ============================================================

class AuditSimulationRequest(BaseModel):
    revenue: float = 100_000_000
    materiality: float = 2_000_000
    mean_error_rate: float = 0.02
    std_error_rate: float = 0.005
    simulations: int = 10_000
    seed: Optional[int] = None


@app.post("/audit/simulate", tags=["Audit"])
def audit_simulate(payload: AuditSimulationRequest):
    if payload.simulations <= 0:
        raise HTTPException(status_code=400, detail="simulations must be > 0")

    if payload.seed is not None:
        np.random.seed(int(payload.seed))

    error_rate = np.random.normal(
        loc=payload.mean_error_rate,
        scale=payload.std_error_rate,
        size=payload.simulations,
    )
    error_rate = np.clip(error_rate, 0.0, 1.0)
    misstatement = payload.revenue * error_rate

    mean_ms = float(np.mean(misstatement))
    sd_ms = float(np.std(misstatement, ddof=1)) if payload.simulations > 1 else 0.0
    percentiles = {p: float(np.percentile(misstatement, p)) for p in (50, 90, 95, 99)}
    prob_exceed = float((misstatement > payload.materiality).mean())

    return {
        "inputs": payload.model_dump(),
        "mean_misstatement": mean_ms,
        "std_misstatement": sd_ms,
        "percentiles": percentiles,
        "probability_exceed_materiality": prob_exceed,
    }


# ============================================================
# 2) RISK / SDE
# ============================================================

class RiskSimulationRequest(BaseModel):
    S0: float = 100.0
    mu: float = 0.05
    sigma: float = 0.2
    T: float = 1.0
    paths: int = 10_000
    seed: Optional[int] = None


@app.post("/risk/simulate", tags=["Risk"])
def risk_simulate(payload: RiskSimulationRequest):
    if payload.paths <= 0:
        raise HTTPException(status_code=400, detail="paths must be > 0")

    if payload.seed is not None:
        np.random.seed(int(payload.seed))

    Z = np.random.normal(size=payload.paths)
    ST = payload.S0 * np.exp(
        (payload.mu - 0.5 * payload.sigma**2) * payload.T
        + payload.sigma * np.sqrt(payload.T) * Z
    )
    PL = ST - payload.S0

    return {
        "inputs": payload.model_dump(),
        "mean_PL": float(np.mean(PL)),
        "VaR_95": float(-np.percentile(PL, 5)),
        "VaR_99": float(-np.percentile(PL, 1)),
    }


# ============================================================
# 3) HYBRID
# ============================================================

class HybridSimulationRequest(BaseModel):
    T: int = 120
    D0: float = 1_000_000
    B0: float = 800_000
    ID0: float = 900_000
    IB0: float = 700_000
    R0: float = 0.0
    gD: float = 0.01
    gB: float = 0.005
    lD: float = 0.008
    lB: float = 0.006
    seed: Optional[int] = 42


@app.post("/hybrid/simulate", tags=["Hybrid"])
def hybrid_simulate(payload: HybridSimulationRequest):
    if payload.seed is not None:
        np.random.seed(int(payload.seed))

    D, B, ID, IB, R = payload.D0, payload.B0, payload.ID0, payload.IB0, payload.R0
    history = []

    for t in range(payload.T):
        D = max(D + payload.gD * D - payload.lD * D, 0)
        B = max(B + payload.gB * B - payload.lB * B, 0)
        revenue = D * 0.015 + B * 0.02
        R += revenue

        history.append({"t": t, "revenue": float(R)})

    return {
        "inputs": payload.model_dump(),
        "final_revenue": float(R),
        "steps": history,
    }


# ============================================================
# 4) TELCO (simplificado)
# ============================================================

class TelcoSimulationRequest(BaseModel):
    T: int = 36
    M: int = 2_000_000
    A0: float = 0.08


@app.post("/telco/simulate", tags=["Telco"])
def telco_simulate(payload: TelcoSimulationRequest):
    A = payload.A0
    results = []

    for t in range(payload.T):
        A = A + 0.03 * (1 - A)
        users = payload.M * A
        revenue = users * 15

        results.append({
            "t": t,
            "adoption": float(A),
            "users": float(users),
            "revenue": float(revenue)
        })

    return {
        "inputs": payload.model_dump(),
        "results": results,
    }


# ============================================================
# 5) LOGISTICS (simplificado)
# ============================================================

class LogisticsSimulationRequest(BaseModel):
    seed: int = 42
    demand_variability: float = 0.2


@app.post("/logistics/simulate", tags=["Logistics"])
def logistics_simulate(payload: LogisticsSimulationRequest):
    random.seed(payload.seed)

    demand = [max(1, int(random.gauss(100, 20))) for _ in range(10)]
    total = sum(demand)

    return {
        "inputs": payload.model_dump(),
        "total_demand": total,
        "demand_distribution": demand,
    }
