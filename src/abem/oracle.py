from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .checkpoints import SearchCheckpoint, Trajectory, continue_from_checkpoint
from .config import MetricConfig, SearchConfig


@dataclass(frozen=True)
class HorizonValue:
    horizon: int
    effective_horizon: int
    mean_improvement: float
    cost: float
    value: float


@dataclass(frozen=True)
class OracleEstimate:
    checkpoint_step: int
    values: tuple[HorizonValue, ...]
    oracle_value: float
    action: str


def rollout_seed(problem_seed: int, checkpoint_step: int, horizon: int, rollout: int) -> int:
    """A deterministic stream disjoint from problem and generating-agent streams."""

    return int(
        90_000_001
        + int(problem_seed) * 1_000_003
        + int(checkpoint_step) * 10_007
        + int(horizon) * 1_009
        + int(rollout) * 101
    ) % (2**63 - 1)


def estimate_oracle_value(
    landscape,
    checkpoint: SearchCheckpoint,
    *,
    problem_seed: int,
    search: SearchConfig,
    metric: MetricConfig,
    horizons: Iterable[int],
    rollouts: int,
) -> OracleEstimate:
    if rollouts < 1:
        raise ValueError("rollouts must be positive")

    values: list[HorizonValue] = []
    for horizon in horizons:
        effective = min(int(horizon), search.t_max - checkpoint.step)
        if effective <= 0:
            continue
        improvements = []
        for rollout in range(rollouts):
            future = continue_from_checkpoint(
                landscape,
                checkpoint,
                config=search,
                horizon=effective,
                future_seed=rollout_seed(problem_seed, checkpoint.step, int(horizon), rollout),
            )
            improvements.append(max(0.0, future.best_score - checkpoint.best_score))
        mean_improvement = float(np.mean(improvements))
        cost = float(metric.cost_weight * effective / search.t_max)
        values.append(
            HorizonValue(
                horizon=int(horizon),
                effective_horizon=effective,
                mean_improvement=mean_improvement,
                cost=cost,
                value=mean_improvement - cost,
            )
        )

    oracle_value = max((item.value for item in values), default=0.0)
    return OracleEstimate(
        checkpoint_step=checkpoint.step,
        values=tuple(values),
        oracle_value=float(oracle_value),
        action="CONTINUE" if oracle_value > 0.0 else "STOP",
    )


def oracle_stop_step(trajectory: Trajectory, estimates: Iterable[OracleEstimate]) -> int:
    by_step = {estimate.checkpoint_step: estimate for estimate in estimates}
    for checkpoint in trajectory.checkpoints:
        estimate = by_step[checkpoint.step]
        if estimate.action == "STOP":
            return checkpoint.step
    return trajectory.step_records[-1].step
