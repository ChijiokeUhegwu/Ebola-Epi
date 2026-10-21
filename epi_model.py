"""SIR simulation and calibration utilities for the Ebola Epi Watch dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Model inputs and outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SIRParams:
    """Inputs required to run one SIR scenario."""

    beta: float
    gamma: float
    population: float
    initial_cases: float
    initial_infectious: float


@dataclass
class SIRTrajectory:
    """Compartment sizes and incidence produced by a simulation."""

    days: np.ndarray
    susceptible: np.ndarray
    infectious: np.ndarray
    recovered: np.ndarray
    cumulative: np.ndarray
    incidence: np.ndarray


@dataclass
class SIRFit:
    """Result of calibrating beta and gamma to an observed cumulative curve."""

    beta: float
    gamma: float
    rmse: float
    success: bool
    message: str


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_sir(params: SIRParams, horizon_days: int, t_eval: np.ndarray | None = None) -> SIRTrajectory:
    """Integrate the SIR equations over the requested horizon.

    The observed cumulative case count is split into active infections and
    recovered people. That keeps the model's starting cumulative total aligned
    with the real-world series while allowing the active infections to be tuned
    independently.
    """
    population = float(params.population)
    if population <= 0:
        raise ValueError("Population must be positive")
    if params.beta <= 0 or params.gamma <= 0:
        raise ValueError("beta and gamma must be positive")

    # Keep every starting compartment internally consistent and inside the population.
    initial_cases = float(np.clip(params.initial_cases, 1.0, population - 1.0))
    initial_infectious = float(
        np.clip(params.initial_infectious, 1.0, max(initial_cases, 1.0))
    )
    initial_recovered = max(initial_cases - initial_infectious, 0.0)
    initial_susceptible = max(population - initial_cases, 0.0)
    initial_state = [initial_susceptible, initial_infectious, initial_recovered]

    # Normalize evaluation times so the solver always begins at day zero.
    horizon = max(float(horizon_days), 1.0)
    if t_eval is None:
        t_eval = np.arange(0.0, horizon + 1.0, 1.0)
    t_eval = np.asarray(t_eval, dtype=float)
    t_eval = np.clip(t_eval, 0.0, horizon)
    t_eval = np.unique(t_eval)
    if t_eval[0] != 0.0:
        t_eval = np.insert(t_eval, 0, 0.0)

    beta = float(params.beta)
    gamma = float(params.gamma)

    def derivatives(_time: float, state: np.ndarray) -> list[float]:
        susceptible, infectious, _recovered = state
        force = beta * susceptible * infectious / population
        return [-force, force - gamma * infectious, gamma * infectious]

    solution = solve_ivp(
        derivatives,
        (0.0, horizon),
        initial_state,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-8,
        atol=1e-4,
        max_step=1.0,
    )
    if not solution.success:
        raise RuntimeError(f"SIR integration failed: {solution.message}")

    susceptible = np.clip(solution.y[0], 0.0, population)
    infectious = np.clip(solution.y[1], 0.0, population)
    recovered = np.clip(solution.y[2], 0.0, population)

    # Cumulative infections equal everyone who has left the susceptible pool.
    cumulative = np.clip(population - susceptible, 0.0, population)

    # Daily incidence is the day-over-day change in cumulative infections.
    incidence = np.diff(cumulative, prepend=cumulative[0])
    incidence = np.clip(incidence, 0.0, population)

    return SIRTrajectory(
        days=solution.t,
        susceptible=susceptible,
        infectious=infectious,
        recovered=recovered,
        cumulative=cumulative,
        incidence=incidence,
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def fit_sir(
    observed_cumulative: np.ndarray,
    days: np.ndarray,
    population: float,
    initial_cases: float,
    initial_infectious: float,
    beta_start: float = 0.40,
    gamma_start: float = 0.15,
) -> SIRFit:
    """Fit beta and gamma by matching the log of cumulative cases.

    Log scaling stops very large later counts from dominating the objective.
    The robust soft-L1 loss limits the influence of reporting outliers and data
    revisions.
    """
    observed = np.asarray(observed_cumulative, dtype=float)
    timepoints = np.asarray(days, dtype=float)

    # Days with zero reported cases cannot be compared on a log scale.
    mask = observed > 0
    if mask.sum() < 5:
        return SIRFit(beta_start, gamma_start, float("nan"), False, "Not enough observations")
    observed_log = np.log1p(observed[mask])
    timepoints = timepoints[mask]
    horizon = int(np.ceil(timepoints.max())) + 1

    def residuals(parameters: np.ndarray) -> np.ndarray:
        beta, gamma = parameters
        try:
            trajectory = simulate_sir(
                SIRParams(
                    beta=float(beta),
                    gamma=float(gamma),
                    population=population,
                    initial_cases=initial_cases,
                    initial_infectious=initial_infectious,
                ),
                horizon,
                t_eval=timepoints,
            )
        except Exception:
            # A failed integration should be strongly penalized, not crash the fit.
            return np.full(mask.sum(), 1e6)
        model_log = np.log1p(np.maximum(trajectory.cumulative, 0.0))
        return model_log - observed_log

    result = least_squares(
        residuals,
        x0=np.array([beta_start, gamma_start], dtype=float),
        bounds=([0.01, 0.01], [3.0, 1.0]),
        method="trf",
        loss="soft_l1",
        f_scale=0.5,
        x_scale=np.array([0.1, 0.1]),
        max_nfev=400,
    )
    rmse = float(np.sqrt(np.mean(np.square(residuals(result.x)))))
    return SIRFit(
        beta=float(result.x[0]),
        gamma=float(result.x[1]),
        rmse=rmse,
        success=bool(result.success),
        message=str(result.message),
    )


# ---------------------------------------------------------------------------
# Derived indicators
# ---------------------------------------------------------------------------


def model_metrics(trajectory: SIRTrajectory, beta: float, gamma: float, population: float) -> dict:
    """Summarize reproduction numbers, peak timing, and final epidemic size."""
    if trajectory.infectious.size == 0:
        return {}
    peak_index = int(np.argmax(trajectory.infectious))
    r0 = beta / gamma if gamma > 0 else float("inf")

    # Effective R reflects how much of the population is still susceptible.
    initial_susceptible = float(trajectory.susceptible[0])
    effective_r = (
        beta * initial_susceptible / (gamma * population) if gamma > 0 and population > 0 else float("inf")
    )
    final_cumulative = float(trajectory.cumulative[-1])
    return {
        "r0": r0,
        "effective_r": effective_r,
        "peak_day": float(trajectory.days[peak_index]),
        "peak_active": float(trajectory.infectious[peak_index]),
        "peak_new_infections": float(np.max(trajectory.incidence)),
        "final_cumulative": final_cumulative,
        "attack_rate": final_cumulative / population if population > 0 else 0.0,
        "herd_immunity": max(0.0, 1.0 - 1.0 / r0) if r0 > 1 else 0.0,
    }
