from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .checkpoints import Trajectory


@dataclass(frozen=True)
class MinimalBoundaryParameters:
    """Frozen parameters for the minimal gain-conditioned patience boundary."""

    p0: int
    delta_p: int
    g0: float
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if self.p0 <= 0 or self.delta_p < 0:
            raise ValueError("p0 must be positive and delta_p non-negative")
        if self.g0 <= 0 or not 0 < self.alpha <= 1:
            raise ValueError("g0 and alpha must be positive")


def ema_gain_series(gains, *, alpha: float = 0.5) -> np.ndarray:
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    output = np.empty(len(gains), dtype=float)
    current = 0.0
    for index, gain in enumerate(gains):
        current = alpha * max(0.0, float(gain)) + (1.0 - alpha) * current
        output[index] = current
    return output


def boundary_value(gbar: float, params: MinimalBoundaryParameters, *, reversed_gain: bool = False) -> float:
    response = max(0.0, float(gbar)) / (max(0.0, float(gbar)) + params.g0)
    if reversed_gain:
        response = 1.0 - response
    return float(params.p0 + params.delta_p * response)


def boundary_series(trajectory: Trajectory, params: MinimalBoundaryParameters) -> np.ndarray:
    gains = [record.gain for record in trajectory.step_records]
    return np.asarray([boundary_value(x, params) for x in ema_gain_series(gains, alpha=params.alpha)])


def minimal_stop_step(
    trajectory: Trajectory,
    *,
    params: MinimalBoundaryParameters,
    t_min: int,
    gbar_override: np.ndarray | None = None,
    reversed_gain: bool = False,
) -> int:
    """Apply the boundary using only present/past gain and present stagnation."""

    if gbar_override is None:
        gbar = ema_gain_series([record.gain for record in trajectory.step_records], alpha=params.alpha)
    else:
        gbar = np.asarray(gbar_override, dtype=float)
        if gbar.shape != (len(trajectory.step_records),):
            raise ValueError("gbar override length mismatch")
    for record, recent_gain in zip(trajectory.step_records, gbar, strict=True):
        boundary = boundary_value(float(recent_gain), params, reversed_gain=reversed_gain)
        if record.step >= t_min and record.stagnation >= boundary:
            return record.step
    return trajectory.step_records[-1].step


def shuffled_gbar(trajectory: Trajectory, *, alpha: float, seed: int) -> np.ndarray:
    """Destroy temporal alignment while preserving the exact EMA-gain marginal."""

    values = ema_gain_series([record.gain for record in trajectory.step_records], alpha=alpha)
    rng = np.random.default_rng(seed)
    return values[rng.permutation(len(values))]
