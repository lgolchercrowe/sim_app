from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional, Tuple, Literal, Any

import numpy as np
from fastapi import FastAPI, Body, HTTPException
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

    # Se mantienen para compatibilidad con el GPT actual
    mean_error_rate: float = 0.02
    std_error_rate: float = 0.005

    simulations: int = 10_000
    seed: Optional[int] = 42

    # Nueva opción de distribución
    distribution: Literal["normal", "triangular", "lognormal"] = "normal"

    # Parámetros opcionales para triangular
    min_error_rate: Optional[float] = None
    mode_error_rate: Optional[float] = None
    max_error_rate: Optional[float] = None


def _sample_error_rate(payload: AuditSimulationRequest) -> np.ndarray:
    n = payload.simulations

    if payload.distribution == "normal":
        samples = np.random.normal(
            loc=payload.mean_error_rate,
            scale=payload.std_error_rate,
            size=n,
        )

    elif payload.distribution == "triangular":
        # Si no vienen los tres parámetros, se construye una aproximación razonable
        low = payload.min_error_rate
        mode = payload.mode_error_rate
        high = payload.max_error_rate

        if low is None or mode is None or high is None:
            spread = max(payload.std_error_rate, 1e-9)
            low = max(0.0, payload.mean_error_rate - 2.0 * spread)
            mode = payload.mean_error_rate if mode is None else mode
            high = min(1.0, payload.mean_error_rate + 2.0 * spread)

        if low < 0 or high > 1:
            raise HTTPException(
                status_code=400,
                detail="triangular distribution requires 0 <= min_error_rate < max_error_rate <= 1",
            )

        if not (low <= mode <= high):
            raise HTTPException(
                status_code=400,
                detail="triangular distribution requires min_error_rate <= mode_error_rate <= max_error_rate",
            )

        samples = np.random.triangular(low, mode, high, size=n)

    elif payload.distribution == "lognormal":
        if payload.mean_error_rate <= 0:
            raise HTTPException(
                status_code=400,
                detail="lognormal distribution requires mean_error_rate > 0",
            )
        if payload.std_error_rate < 0:
            raise HTTPException(
                status_code=400,
                detail="std_error_rate must be >= 0",
            )

        # Convertimos media y desviación objetivo a parámetros de la lognormal
        cv = payload.std_error_rate / payload.mean_error_rate if payload.mean_error_rate > 0 else 0.0
        sigma2 = math.log(1.0 + cv**2) if cv > 0 else 0.0
        mu = math.log(payload.mean_error_rate) - 0.5 * sigma2 if payload.mean_error_rate > 0 else 0.0

        samples = np.random.lognormal(mean=mu, sigma=math.sqrt(sigma2), size=n)

    else:
        raise HTTPException(status_code=400, detail="Unsupported distribution")

    # El error rate siempre debe quedar entre 0 y 1
    return np.clip(samples, 0.0, 1.0)


@app.post("/audit/simulate", tags=["Audit"])
def audit_simulate(payload: AuditSimulationRequest):
    if payload.simulations <= 0:
        raise HTTPException(status_code=400, detail="simulations must be > 0")
    if payload.revenue < 0:
        raise HTTPException(status_code=400, detail="revenue must be >= 0")
    if payload.materiality <= 0:
        raise HTTPException(status_code=400, detail="materiality must be > 0")
    if payload.mean_error_rate < 0:
        raise HTTPException(status_code=400, detail="mean_error_rate must be >= 0")
    if payload.std_error_rate < 0:
        raise HTTPException(status_code=400, detail="std_error_rate must be >= 0")

    if payload.seed is not None:
        np.random.seed(int(payload.seed))

    error_rate = _sample_error_rate(payload)
    misstatement = payload.revenue * error_rate

    exceedances = int(np.sum(misstatement > payload.materiality))
    prob_exceed = float(exceedances / payload.simulations)

    mean_ms = float(np.mean(misstatement))
    sd_ms = float(np.std(misstatement, ddof=1)) if payload.simulations > 1 else 0.0

    percentiles = {
        "p2_5": float(np.percentile(misstatement, 2.5)),
        "p5": float(np.percentile(misstatement, 5)),
        "p25": float(np.percentile(misstatement, 25)),
        "p50": float(np.percentile(misstatement, 50)),
        "p75": float(np.percentile(misstatement, 75)),
        "p90": float(np.percentile(misstatement, 90)),
        "p95": float(np.percentile(misstatement, 95)),
        "p97_5": float(np.percentile(misstatement, 97.5)),
        "p99": float(np.percentile(misstatement, 99)),
    }

    error_rate_summary = {
        "mean": float(np.mean(error_rate)),
        "std": float(np.std(error_rate, ddof=1)) if payload.simulations > 1 else 0.0,
        "percentiles": {
            "p50": float(np.percentile(error_rate, 50)),
            "p90": float(np.percentile(error_rate, 90)),
            "p95": float(np.percentile(error_rate, 95)),
            "p99": float(np.percentile(error_rate, 99)),
        },
    }

    scenarios = {
        "base": percentiles["p50"],
        "conservative": percentiles["p90"],
        "stress": percentiles["p95"],
        "extreme": percentiles["p99"],
    }

    return {
        "inputs": payload.model_dump(),
        "distribution": payload.distribution,
        "mean_misstatement": mean_ms,
        "std_misstatement": sd_ms,
        "materiality": payload.materiality,
        "exceedances": exceedances,
        "probability_exceed_materiality": prob_exceed,
        "percentiles": percentiles,
        "scenarios": scenarios,
        "error_rate_summary": error_rate_summary,
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

    if payload.S0 <= 0:
        raise HTTPException(status_code=400, detail="S0 must be > 0")

    if payload.T <= 0:
        raise HTTPException(status_code=400, detail="T must be > 0")

    if payload.sigma < 0:
        raise HTTPException(status_code=400, detail="sigma must be >= 0")

    if payload.seed is not None:
        np.random.seed(int(payload.seed))

    Z = np.random.normal(size=payload.paths)

    ST = payload.S0 * np.exp(
        (payload.mu - 0.5 * payload.sigma**2) * payload.T
        + payload.sigma * np.sqrt(payload.T) * Z
    )

    PL = ST - payload.S0

    mean_pl = float(np.mean(PL))
    p5_pl = float(np.percentile(PL, 5))
    p1_pl = float(np.percentile(PL, 1))

    # VaR como pérdida: nunca reportar valor negativo como "pérdida".
    # Si el percentil adverso sigue siendo positivo, el VaR de pérdida es 0.
    var_95 = max(0.0, -p5_pl)
    var_99 = max(0.0, -p1_pl)

    return {
        "inputs": payload.model_dump(),
        "mean_PL": mean_pl,
        "P5_PL": p5_pl,
        "P1_PL": p1_pl,
        "VaR_95": var_95,
        "VaR_99": var_99,
        "interpretation": (
            "VaR_95 y VaR_99 están expresados como pérdida potencial. "
            "Si el percentil de cola es positivo, el resultado adverso sigue siendo ganancia "
            "y por eso el VaR de pérdida se reporta como 0."
        ),
    }

# ============================================================
# 2B) RISK / PORTFOLIO SDE
# ============================================================

class PortfolioAsset(BaseModel):
    name: str
    S0: float = Field(..., gt=0)
    mu: float
    sigma: float = Field(..., ge=0)
    weight: float


class PortfolioSimulationRequest(BaseModel):
    assets: List[PortfolioAsset]
    correlation_matrix: List[List[float]]
    T: float = Field(default=1.0, gt=0)
    paths: int = Field(default=10_000, gt=0)
    seed: Optional[int] = 42


@app.post("/risk/portfolio/simulate", tags=["Risk"])
def risk_portfolio_simulate(payload: PortfolioSimulationRequest):
    n_assets = len(payload.assets)

    if n_assets == 0:
        raise HTTPException(status_code=400, detail="assets must not be empty")

    if len(payload.correlation_matrix) != n_assets:
        raise HTTPException(
            status_code=400,
            detail="correlation_matrix must have the same number of rows as assets",
        )

    for row in payload.correlation_matrix:
        if len(row) != n_assets:
            raise HTTPException(
                status_code=400,
                detail="correlation_matrix must be square and match the number of assets",
            )

    if payload.seed is not None:
        np.random.seed(int(payload.seed))

    corr = np.array(payload.correlation_matrix, dtype=float)

    # Validaciones básicas de la matriz de correlación
    if not np.allclose(corr, corr.T, atol=1e-8):
        raise HTTPException(
            status_code=400,
            detail="correlation_matrix must be symmetric",
        )

    if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
        raise HTTPException(
            status_code=400,
            detail="correlation_matrix diagonal must be 1.0",
        )

    # Cholesky para generar shocks correlacionados
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        raise HTTPException(
            status_code=400,
            detail="correlation_matrix must be positive definite",
        )

    # Pesos del portafolio
    weights = np.array([a.weight for a in payload.assets], dtype=float)
    S0 = np.array([a.S0 for a in payload.assets], dtype=float)
    mu = np.array([a.mu for a in payload.assets], dtype=float)
    sigma = np.array([a.sigma for a in payload.assets], dtype=float)

    # Normalmente el peso total puede o no sumar 1; no lo forzamos,
    # porque puede representar una exposición económica total.
    initial_portfolio_value = float(np.sum(weights * S0))

    # Generar normales independientes y luego correlacionarlas
    Z = np.random.normal(size=(payload.paths, n_assets))
    Z_corr = Z @ L.T

    # Simulación conjunta por activo
    ST = np.zeros((payload.paths, n_assets))
    for i in range(n_assets):
        ST[:, i] = S0[i] * np.exp(
            (mu[i] - 0.5 * sigma[i] ** 2) * payload.T
            + sigma[i] * np.sqrt(payload.T) * Z_corr[:, i]
        )

    # P&L del portafolio
    portfolio_final_value = np.sum(ST * weights, axis=1)
    portfolio_PL = portfolio_final_value - initial_portfolio_value

    mean_pl = float(np.mean(portfolio_PL))
    p5_pl = float(np.percentile(portfolio_PL, 5))
    p1_pl = float(np.percentile(portfolio_PL, 1))

    # VaR como pérdida potencial, nunca negativo
    var_95 = max(0.0, -p5_pl)
    var_99 = max(0.0, -p1_pl)

    asset_details = []
    for i, asset in enumerate(payload.assets):
        asset_details.append({
            "name": asset.name,
            "S0": asset.S0,
            "mu": asset.mu,
            "sigma": asset.sigma,
            "weight": asset.weight,
        })

    return {
        "inputs": {
            "assets": asset_details,
            "correlation_matrix": payload.correlation_matrix,
            "T": payload.T,
            "paths": payload.paths,
            "seed": payload.seed,
        },
        "portfolio": {
            "initial_value": initial_portfolio_value,
            "mean_PL": mean_pl,
            "P5_PL": p5_pl,
            "P1_PL": p1_pl,
            "VaR_95": var_95,
            "VaR_99": var_99,
        },
        "interpretation": (
            "El portafolio fue simulado con shocks correlacionados entre activos. "
            "VaR_95 y VaR_99 se reportan como pérdida potencial. "
            "Si el percentil adverso del P&L sigue siendo positivo, el VaR de pérdida se reporta como 0."
        ),
    }

# ============================================================
# 3) HYBRID
# ============================================================

class HybridSimulationRequest(BaseModel):
    T: int = Field(default=120, gt=0)
    D0: float = Field(default=1_000_000, ge=0)
    B0: float = Field(default=800_000, ge=0)
    ID0: float = Field(default=900_000, ge=0)
    IB0: float = Field(default=700_000, ge=0)
    R0: float = 0.0
    gD: float = 0.01
    gB: float = 0.005
    lD: float = 0.008
    lB: float = 0.006
    seed: Optional[int] = 42


@app.post("/hybrid/simulate", tags=["Hybrid"])
def hybrid_simulate(payload: HybridSimulationRequest = Body(default_factory=HybridSimulationRequest)):
    try:
        if payload.seed is not None:
            np.random.seed(int(payload.seed))

        D, B, ID, IB, R = payload.D0, payload.B0, payload.ID0, payload.IB0, payload.R0
        history = []

        for t in range(payload.T):
            D = max(D + payload.gD * D - payload.lD * D, 0)
            B = max(B + payload.gB * B - payload.lB * B, 0)
            revenue = D * 0.015 + B * 0.02
            R += revenue

            history.append({
                "t": t,
                "revenue": float(R),
                "D": float(D),
                "B": float(B),
            })

        return {
            "inputs": payload.model_dump(),
            "final_revenue": float(R),
            "steps": history,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid simulation failed: {str(e)}")

# ============================================================
# 4) TELCO
# ============================================================

class TelcoSimulationRequest(BaseModel):
    horizon_months: int = Field(default=36, ge=1, le=120)
    market_size: int = Field(default=2_000_000, ge=1)

    # Adoption / demand
    initial_adoption: float = Field(default=0.08, ge=0.0, le=1.0)
    base_monthly_growth: float = Field(default=0.03, ge=0.0, le=1.0)
    adoption_ceiling: float = Field(default=0.75, ge=0.0, le=1.0)

    # Commercials
    monthly_arpu: float = Field(default=15.0, ge=0.0)
    price_change_pct: float = Field(default=0.0, ge=-1.0, le=1.0)
    price_elasticity: float = Field(default=0.6, ge=0.0, le=5.0)

    # Churn / competition
    base_churn_rate: float = Field(default=0.015, ge=0.0, le=1.0)
    competitive_pressure: float = Field(default=1.0, ge=0.0, le=3.0)
    regulatory_factor: float = Field(default=1.0, ge=0.0, le=3.0)

    # Network / rollout
    network_capacity_users: int = Field(default=1_600_000, ge=1)
    rollout_capex: float = Field(default=250_000_000, ge=0.0)
    monthly_opex: float = Field(default=8_000_000, ge=0.0)
    variable_cost_per_user: float = Field(default=2.5, ge=0.0)

    # Financials
    discount_rate_monthly: float = Field(default=0.01, ge=0.0, le=1.0)

    # Scenario drivers
    demand_shock: float = Field(default=1.0, ge=0.0, le=3.0)
    adoption_shock: float = Field(default=1.0, ge=0.0, le=3.0)


@app.post("/telco/simulate", tags=["Telco"])
def telco_simulate(payload: TelcoSimulationRequest):
    price = payload.monthly_arpu * (1 + payload.price_change_pct)

    adoption = payload.initial_adoption
    cumulative_discounted_cashflow = -payload.rollout_capex
    results: List[Dict[str, Any]] = []

    for t in range(payload.horizon_months):
        # Demand and adoption dynamics
        price_effect = max(0.0, 1.0 - payload.price_elasticity * abs(payload.price_change_pct))
        growth = (
            payload.base_monthly_growth
            * payload.demand_shock
            * payload.adoption_shock
            * payload.regulatory_factor
            * price_effect
            / payload.competitive_pressure
        )

        churn_penalty = payload.base_churn_rate * payload.competitive_pressure
        adoption = adoption + growth * (payload.adoption_ceiling - adoption) - churn_penalty * adoption
        adoption = max(0.0, min(payload.adoption_ceiling, adoption))

        users = payload.market_size * adoption
        revenue = users * price
        opex = payload.monthly_opex + users * payload.variable_cost_per_user
        ebitda = revenue - opex
        capacity_pressure = users / payload.network_capacity_users if payload.network_capacity_users else None

        discounted_cashflow = ebitda / ((1 + payload.discount_rate_monthly) ** (t + 1))
        cumulative_discounted_cashflow += discounted_cashflow

        results.append({
            "t": t + 1,
            "adoption": round(adoption, 6),
            "users": round(users, 2),
            "price": round(price, 2),
            "revenue": round(revenue, 2),
            "opex": round(opex, 2),
            "ebitda": round(ebitda, 2),
            "capacity_pressure": round(capacity_pressure, 4) if capacity_pressure is not None else None,
            "discounted_cashflow": round(discounted_cashflow, 2),
            "cumulative_discounted_cashflow": round(cumulative_discounted_cashflow, 2),
        })

    final = results[-1]
    peak_pressure = max(r["capacity_pressure"] for r in results if r["capacity_pressure"] is not None)
    avg_ebitda = sum(r["ebitda"] for r in results) / len(results)

    return {
        "inputs": payload.model_dump(),
        "summary": {
            "final_adoption": final["adoption"],
            "final_users": final["users"],
            "final_revenue": final["revenue"],
            "final_ebitda": final["ebitda"],
            "average_monthly_ebitda": round(avg_ebitda, 2),
            "peak_capacity_pressure": round(peak_pressure, 4),
            "npv_proxy": round(cumulative_discounted_cashflow, 2),
            "payback_signaled": cumulative_discounted_cashflow > 0,
        },
        "results": results,
    }

# ============================================================
# 5) LOGISTICS (simplificado)
# ============================================================

from typing import List, Optional, Dict, Tuple, Literal
from pydantic import BaseModel, Field
from fastapi import HTTPException

class DepotIn(BaseModel):
    name: str
    x: float
    y: float

class CustomerIn(BaseModel):
    name: str
    x: float
    y: float
    base_qty: int
    unit_weight_t: float
    unit_volume_m3: float
    service_min_fixed: float
    service_min_per_unit: float
    preferred_depot: Optional[str] = None

class TruckIn(BaseModel):
    name: str
    cap_qty: int
    cap_weight_t: float
    cap_volume_m3: float
    fixed_cost: float
    cost_per_km: float
    cost_per_hour: float
    avg_speed_kmh: float
    fleet_size: int = 1

class LogisticsSimulationRequest(BaseModel):
    depots: List[DepotIn]
    customers: List[CustomerIn]
    trucks: List[TruckIn]
    seed: int = 42
    demand_variability: float = 0.20
    n_scenarios: int = 1
    assignment_mode: Literal["nearest", "preferred"] = "nearest"

@app.post("/logistics/simulate", tags=["Logistics"])
def logistics_simulate(payload: LogisticsSimulationRequest):
    if payload.n_scenarios <= 0:
        raise HTTPException(status_code=400, detail="n_scenarios must be > 0")
    if not payload.depots:
        raise HTTPException(status_code=400, detail="At least one depot is required")
    if not payload.customers:
        raise HTTPException(status_code=400, detail="At least one customer is required")
    if not payload.trucks:
        raise HTTPException(status_code=400, detail="At least one truck is required")

    rng = random.Random(payload.seed)

    depots = [Depot(d.name, d.x, d.y) for d in payload.depots]
    customers = [Customer(
        c.name, c.x, c.y, c.base_qty, c.unit_weight_t, c.unit_volume_m3,
        c.service_min_fixed, c.service_min_per_unit
    ) for c in payload.customers]
    trucks = [Truck(
        t.name, t.cap_qty, t.cap_weight_t, t.cap_volume_m3,
        t.fixed_cost, t.cost_per_km, t.cost_per_hour, t.avg_speed_kmh
    ) for t in payload.trucks]

    def choose_truck(load_qty: int, load_weight: float, load_volume: float) -> Truck:
        feasible = [
            t for t in trucks
            if load_qty <= t.cap_qty and load_weight <= t.cap_weight_t and load_volume <= t.cap_volume_m3
        ]
        if not feasible:
            raise ValueError("No truck can fit the load")
        return sorted(feasible, key=lambda t: (t.fixed_cost, t.cap_qty, t.cap_weight_t, t.cap_volume_m3))[0]

    def assign_depot(c: Customer) -> Depot:
        if payload.assignment_mode == "preferred" and c.preferred_depot:
            found = next((d for d in depots if d.name == c.preferred_depot), None)
            if found:
                return found
        return min(depots, key=lambda d: dist(d, c))

    all_scenarios = []

    for s in range(payload.n_scenarios):
        scenario_qty: Dict[str, int] = {}
        for c in customers:
            factor = max(0.25, rng.gauss(1.0, payload.demand_variability))
            qty = max(1, int(round(c.base_qty * factor)))
            scenario_qty[c.name] = qty

        by_depot: Dict[str, List[Customer]] = {d.name: [] for d in depots}
        for c in customers:
            d = assign_depot(c)
            by_depot[d.name].append(c)

        routes: List[Dict] = []

        for d in depots:
            remaining = {c.name: int(scenario_qty[c.name]) for c in by_depot[d.name]}
            ordered_customers = sorted(by_depot[d.name], key=lambda c: dist(d, c))
            trip_id = 1

            while any(qty > 0 for qty in remaining.values()):
                stops: List[Tuple[Customer, int]] = []
                load_qty = 0
                load_weight = 0.0
                load_volume = 0.0
                current_loc = d

                while True:
                    candidates = [c for c in ordered_customers if remaining[c.name] > 0]
                    if not candidates:
                        break

                    feasible = []
                    for c in candidates:
                        qty_left = remaining[c.name]
                        if load_qty + qty_left <= max(t.cap_qty for t in trucks) and \
                           load_weight + qty_left * c.unit_weight_t <= max(t.cap_weight_t for t in trucks) and \
                           load_volume + qty_left * c.unit_volume_m3 <= max(t.cap_volume_m3 for t in trucks):
                            feasible.append(c)

                    if not stops:
                        if feasible:
                            next_customer = min(feasible, key=lambda c: dist(current_loc, c))
                            qty_to_ship = remaining[next_customer.name]
                        else:
                            next_customer = min(candidates, key=lambda c: dist(current_loc, c))
                            best_truck = choose_truck(
                                load_qty + 1,
                                load_weight + next_customer.unit_weight_t,
                                load_volume + next_customer.unit_volume_m3
                            )
                            qty_to_ship = max_qty_that_fits(
                                best_truck, load_qty, load_weight, load_volume, next_customer
                            )
                            if qty_to_ship <= 0:
                                raise ValueError(f"Not enough capacity for {next_customer.name}")
                    else:
                        if not feasible:
                            break
                        next_customer = min(feasible, key=lambda c: dist(current_loc, c))
                        qty_to_ship = remaining[next_customer.name]

                    stops.append((next_customer, qty_to_ship))
                    remaining[next_customer.name] -= qty_to_ship
                    load_qty += qty_to_ship
                    load_weight += qty_to_ship * next_customer.unit_weight_t
                    load_volume += qty_to_ship * next_customer.unit_volume_m3
                    current_loc = next_customer

                if stops:
                    truck = choose_truck(load_qty, load_weight, load_volume)
                    distance_km = route_travel_distance(d, stops)
                    travel_h = distance_km / truck.avg_speed_kmh
                    service_min = sum(c.service_min_fixed + qty * c.service_min_per_unit for c, qty in stops)
                    load_min = 15 + 0.15 * load_qty
                    total_h = travel_h + (service_min + load_min) / 60.0
                    cost = truck.fixed_cost + distance_km * truck.cost_per_km + total_h * truck.cost_per_hour

                    routes.append({
                        "trip_id": f"{d.name}-{trip_id}",
                        "depot": d.name,
                        "truck": truck.name,
                        "stops": [(c.name, qty) for c, qty in stops],
                        "load_qty": load_qty,
                        "load_weight_t": round(load_weight, 3),
                        "load_volume_m3": round(load_volume, 3),
                        "util_qty": round(load_qty / truck.cap_qty, 3),
                        "util_weight": round(load_weight / truck.cap_weight_t, 3),
                        "util_volume": round(load_volume / truck.cap_volume_m3, 3),
                        "distance_km": round(distance_km, 2),
                        "travel_h": round(travel_h, 2),
                        "service_h": round((service_min + load_min) / 60.0, 2),
                        "total_h": round(total_h, 2),
                        "cost": round(cost, 2),
                    })
                    trip_id += 1

        total_cost = sum(r["cost"] for r in routes)
        total_distance = sum(r["distance_km"] for r in routes)
        total_qty = sum(r["load_qty"] for r in routes)

        all_scenarios.append({
            "scenario_id": s + 1,
            "scenario_qty": scenario_qty,
            "routes": routes,
            "summary": {
                "trips": len(routes),
                "total_cost": round(total_cost, 2),
                "total_distance_km": round(total_distance, 2),
                "total_qty": total_qty,
                "avg_util_weight": round(mean(r["util_weight"] for r in routes), 3) if routes else 0,
                "avg_util_volume": round(mean(r["util_volume"] for r in routes), 3) if routes else 0,
                "avg_util_qty": round(mean(r["util_qty"] for r in routes), 3) if routes else 0,
            }
        })

    if payload.n_scenarios == 1:
        return {
            "mode": "single",
            "inputs": payload.model_dump(),
            "result": all_scenarios[0],
        }

    costs = [s["summary"]["total_cost"] for s in all_scenarios]
    trips = [s["summary"]["trips"] for s in all_scenarios]
    distances = [s["summary"]["total_distance_km"] for s in all_scenarios]

    return {
        "mode": "many",
        "inputs": payload.model_dump(),
        "aggregate": {
            "n_scenarios": payload.n_scenarios,
            "avg_cost": round(mean(costs), 2),
            "min_cost": round(min(costs), 2),
            "max_cost": round(max(costs), 2),
            "avg_trips": round(mean(trips), 2),
            "avg_distance_km": round(mean(distances), 2),
        },
        "scenarios": all_scenarios,
    }
