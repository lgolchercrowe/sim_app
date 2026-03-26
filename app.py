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
