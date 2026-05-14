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

# ============================================================
# 2) RISK / SDE
# ============================================================

class RiskSimulationRequest(BaseModel):
    S0: float = Field(
        default=100.0,
        gt=0,
        description="Initial asset or portfolio value.",
    )
    mu: float = Field(
        default=0.05,
        description="Expected return or drift. Example: 0.05 means 5%.",
    )
    sigma: float = Field(
        default=0.2,
        ge=0.0,
        description="Volatility. Example: 0.2 means 20%.",
    )
    T: float = Field(
        default=1.0,
        gt=0,
        description="Time horizon in years.",
    )
    paths: int = Field(
        default=10_000,
        gt=0,
        description="Number of Monte Carlo simulation paths.",
    )
    seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducible simulations.",
    )


class PortfolioAssetIn(BaseModel):
    name: str = Field(
        ...,
        description="Asset name, for example: Stocks, Bonds, FX, Crypto.",
    )
    value: float = Field(
        ...,
        gt=0,
        description="Initial monetary value invested in this asset.",
    )
    mu: float = Field(
        ...,
        description="Expected return or drift for this asset.",
    )
    sigma: float = Field(
        ...,
        ge=0.0,
        description="Volatility for this asset.",
    )


class PortfolioRiskSimulationRequest(BaseModel):
    assets: List[PortfolioAssetIn] = Field(
        ...,
        min_length=1,
        description="List of assets included in the portfolio.",
    )
    correlation: Optional[List[List[float]]] = Field(
        default=None,
        description=(
            "Correlation matrix between assets. "
            "If omitted, assets are assumed to be uncorrelated."
        ),
    )
    T: float = Field(
        default=1.0,
        gt=0,
        description="Time horizon in years.",
    )
    paths: int = Field(
        default=10_000,
        gt=0,
        description="Number of Monte Carlo simulation paths.",
    )
    seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducible simulations.",
    )


class StressScenarioIn(BaseModel):
    name: str = Field(
        ...,
        description="Scenario name, for example: Recession, Market Crash, FX Shock.",
    )
    mu_shift: float = Field(
        default=0.0,
        description=(
            "Shift applied to each asset's expected return. "
            "Example: -0.10 means expected return is reduced by 10 percentage points."
        ),
    )
    sigma_multiplier: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Multiplier applied to volatility. "
            "Example: 1.8 means volatility increases by 80%."
        ),
    )
    market_shock_pct: float = Field(
        default=0.0,
        ge=-1.0,
        description=(
            "Immediate shock applied to terminal simulated values. "
            "Example: -0.20 means a 20% adverse market shock."
        ),
    )
    correlation_blend_to_one: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Stress parameter that pushes correlations toward 1. "
            "0 means no change. 1 means all assets move together."
        ),
    )


class PortfolioStressSimulationRequest(PortfolioRiskSimulationRequest):
    scenarios: List[StressScenarioIn] = Field(
        ...,
        min_length=1,
        description="Stress scenarios to simulate.",
    )


def calculate_risk_metrics_from_pl(PL: np.ndarray) -> Dict[str, Any]:
    """
    Calculates risk metrics from simulated P&L.

    PL means Profit and Loss:
        PL = final value - initial value

    Losses are calculated as:
        losses = -PL

    VaR is reported as a positive potential loss amount.
    Therefore, a positive VaR does NOT mean a gain.
    """

    losses = -PL

    pnl_percentile_5 = float(np.percentile(PL, 5))
    pnl_percentile_1 = float(np.percentile(PL, 1))

    raw_loss_percentile_95 = float(np.percentile(losses, 95))
    raw_loss_percentile_99 = float(np.percentile(losses, 99))

    VaR_95_loss_amount = float(max(0.0, raw_loss_percentile_95))
    VaR_99_loss_amount = float(max(0.0, raw_loss_percentile_99))

    tail_95 = losses[losses >= raw_loss_percentile_95]
    tail_99 = losses[losses >= raw_loss_percentile_99]

    expected_shortfall_95_loss_amount = (
        float(max(0.0, np.mean(tail_95)))
        if len(tail_95) > 0
        else VaR_95_loss_amount
    )

    expected_shortfall_99_loss_amount = (
        float(max(0.0, np.mean(tail_99)))
        if len(tail_99) > 0
        else VaR_99_loss_amount
    )

    loss_probability = float(np.mean(PL < 0))

    return {
        "mean_PL": float(np.mean(PL)),

        "pnl_percentile_5": pnl_percentile_5,
        "pnl_percentile_1": pnl_percentile_1,

        "raw_loss_percentile_95": raw_loss_percentile_95,
        "raw_loss_percentile_99": raw_loss_percentile_99,

        "VaR_95_loss_amount": VaR_95_loss_amount,
        "VaR_99_loss_amount": VaR_99_loss_amount,

        "VaR_95": VaR_95_loss_amount,
        "VaR_99": VaR_99_loss_amount,

        "expected_shortfall_95_loss_amount": expected_shortfall_95_loss_amount,
        "expected_shortfall_99_loss_amount": expected_shortfall_99_loss_amount,

        "loss_probability": loss_probability,

        "sign_convention": (
            "VaR is reported as a positive potential loss amount, not as a gain."
        ),
    }


def validate_correlation_matrix(corr: np.ndarray, asset_count: int) -> None:
    """
    Validates that the correlation matrix is usable for portfolio simulation.
    """

    if corr.shape != (asset_count, asset_count):
        raise HTTPException(
            status_code=400,
            detail="correlation matrix size must match the number of assets",
        )

    if not np.allclose(corr, corr.T, atol=1e-8):
        raise HTTPException(
            status_code=400,
            detail="correlation matrix must be symmetric",
        )

    if not np.allclose(np.diag(corr), np.ones(asset_count), atol=1e-8):
        raise HTTPException(
            status_code=400,
            detail="correlation matrix diagonal must be 1",
        )

    if np.any(corr < -1.0) or np.any(corr > 1.0):
        raise HTTPException(
            status_code=400,
            detail="correlation values must be between -1 and 1",
        )

    eigenvalues = np.linalg.eigvalsh(corr)

    if np.min(eigenvalues) < -1e-8:
        raise HTTPException(
            status_code=400,
            detail="correlation matrix must be positive semidefinite",
        )


def build_correlation_matrix(
    correlation: Optional[List[List[float]]],
    asset_count: int,
) -> np.ndarray:
    """
    Builds the correlation matrix.

    If no correlation matrix is provided, assets are assumed to be uncorrelated.
    """

    if correlation is None:
        corr = np.eye(asset_count)
    else:
        corr = np.array(correlation, dtype=float)

    validate_correlation_matrix(corr, asset_count)

    return corr


def correlation_matrix_square_root(corr: np.ndarray) -> np.ndarray:
    """
    Creates a square root matrix for the correlation matrix.

    This uses eigenvalue decomposition instead of Cholesky because it can handle
    positive semidefinite matrices, not only positive definite matrices.
    """

    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    return eigenvectors @ np.diag(np.sqrt(eigenvalues))


def simulate_correlated_gbm_terminal_values(
    values: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    T: float,
    paths: int,
    seed: Optional[int],
) -> np.ndarray:
    """
    Simulates terminal values for multiple correlated assets using GBM.

    Each asset follows:

        ST = S0 * exp((mu - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)

    The Z shocks are correlated using the correlation matrix.
    """

    asset_count = len(values)

    rng = np.random.default_rng(seed)

    corr_sqrt = correlation_matrix_square_root(corr)

    Z_independent = rng.standard_normal(size=(paths, asset_count))
    Z_correlated = Z_independent @ corr_sqrt.T

    terminal_values = values * np.exp(
        (mu - 0.5 * sigma**2) * T
        + sigma * np.sqrt(T) * Z_correlated
    )

    return terminal_values


@app.post("/risk/simulate", tags=["Risk"])
def risk_simulate(payload: RiskSimulationRequest):
    """
    Runs a single-asset Geometric Brownian Motion Monte Carlo simulation.

    This endpoint is for one asset or one aggregated portfolio value.

    Model:
        ST = S0 * exp((mu - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)

    P&L:
        PL = ST - S0

    VaR:
        VaR is calculated from simulated losses and reported as a positive
        potential loss amount.
    """

    rng = np.random.default_rng(payload.seed)

    Z = rng.standard_normal(size=payload.paths)

    ST = payload.S0 * np.exp(
        (payload.mu - 0.5 * payload.sigma**2) * payload.T
        + payload.sigma * np.sqrt(payload.T) * Z
    )

    PL = ST - payload.S0

    metrics = calculate_risk_metrics_from_pl(PL)

    return {
        "inputs": payload.model_dump(),
        "model": "Geometric Brownian Motion single-asset simulation",
        "initial_value": payload.S0,
        "paths": payload.paths,
        "time_horizon_years": payload.T,
        **metrics,
    }


@app.post("/risk/portfolio/simulate", tags=["Risk"])
def portfolio_risk_simulate(payload: PortfolioRiskSimulationRequest):
    """
    Runs a correlated multi-asset portfolio GBM simulation.

    This endpoint is used when the user wants to analyze a complete portfolio,
    for example stocks, bonds, FX, commodities, or crypto together.

    It incorporates diversification and joint risk through the correlation matrix.
    """

    asset_count = len(payload.assets)

    values = np.array([asset.value for asset in payload.assets], dtype=float)
    mu = np.array([asset.mu for asset in payload.assets], dtype=float)
    sigma = np.array([asset.sigma for asset in payload.assets], dtype=float)

    corr = build_correlation_matrix(payload.correlation, asset_count)

    terminal_values = simulate_correlated_gbm_terminal_values(
        values=values,
        mu=mu,
        sigma=sigma,
        corr=corr,
        T=payload.T,
        paths=payload.paths,
        seed=payload.seed,
    )

    portfolio_initial_value = float(np.sum(values))
    portfolio_terminal_value = np.sum(terminal_values, axis=1)

    PL = portfolio_terminal_value - portfolio_initial_value

    metrics = calculate_risk_metrics_from_pl(PL)

    asset_inputs = [
        {
            "name": asset.name,
            "value": asset.value,
            "mu": asset.mu,
            "sigma": asset.sigma,
            "portfolio_weight": float(asset.value / portfolio_initial_value),
        }
        for asset in payload.assets
    ]

    return {
        "inputs": payload.model_dump(),
        "model": "Correlated multi-asset Geometric Brownian Motion portfolio simulation",
        "portfolio_initial_value": portfolio_initial_value,
        "asset_count": asset_count,
        "assets": asset_inputs,
        "correlation_matrix_used": corr.tolist(),
        "paths": payload.paths,
        "time_horizon_years": payload.T,
        **metrics,
    }


@app.post("/risk/portfolio/stress", tags=["Risk"])
def portfolio_stress_simulate(payload: PortfolioStressSimulationRequest):
    """
    Runs portfolio risk simulations under future or stress scenarios.

    Stress scenarios can modify:
    - expected returns through mu_shift,
    - volatility through sigma_multiplier,
    - terminal values through market_shock_pct,
    - diversification through correlation_blend_to_one.

    This helps capture possible extreme losses under adverse market conditions.
    """

    asset_count = len(payload.assets)

    values = np.array([asset.value for asset in payload.assets], dtype=float)
    base_mu = np.array([asset.mu for asset in payload.assets], dtype=float)
    base_sigma = np.array([asset.sigma for asset in payload.assets], dtype=float)

    base_corr = build_correlation_matrix(payload.correlation, asset_count)

    portfolio_initial_value = float(np.sum(values))

    asset_inputs = [
        {
            "name": asset.name,
            "value": asset.value,
            "mu": asset.mu,
            "sigma": asset.sigma,
            "portfolio_weight": float(asset.value / portfolio_initial_value),
        }
        for asset in payload.assets
    ]

    scenario_results = []

    for scenario in payload.scenarios:
        stressed_mu = base_mu + scenario.mu_shift
        stressed_sigma = base_sigma * scenario.sigma_multiplier

        ones_corr = np.ones_like(base_corr)

        stressed_corr = (
            (1.0 - scenario.correlation_blend_to_one) * base_corr
            + scenario.correlation_blend_to_one * ones_corr
        )

        validate_correlation_matrix(stressed_corr, asset_count)

        terminal_values = simulate_correlated_gbm_terminal_values(
            values=values,
            mu=stressed_mu,
            sigma=stressed_sigma,
            corr=stressed_corr,
            T=payload.T,
            paths=payload.paths,
            seed=payload.seed,
        )

        terminal_values = terminal_values * (1.0 + scenario.market_shock_pct)

        portfolio_terminal_value = np.sum(terminal_values, axis=1)

        PL = portfolio_terminal_value - portfolio_initial_value

        metrics = calculate_risk_metrics_from_pl(PL)

        scenario_results.append({
            "scenario_name": scenario.name,
            "scenario_inputs": scenario.model_dump(),
            "stressed_mu_used": stressed_mu.tolist(),
            "stressed_sigma_used": stressed_sigma.tolist(),
            "stressed_correlation_matrix_used": stressed_corr.tolist(),
            **metrics,
        })

    return {
        "inputs": payload.model_dump(),
        "model": "Correlated portfolio GBM stress scenario simulation",
        "portfolio_initial_value": portfolio_initial_value,
        "asset_count": asset_count,
        "assets": asset_inputs,
        "base_correlation_matrix_used": base_corr.tolist(),
        "paths": payload.paths,
        "time_horizon_years": payload.T,
        "scenarios": scenario_results,
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
