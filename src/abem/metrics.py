from __future__ import annotations

import numpy as np

from .config import MetricConfig, SearchConfig


def normalized_regret(best_score: float) -> float:
    return float(np.clip(1.0 - best_score, 0.0, 1.0))


def normalized_cost(steps: int, search: SearchConfig) -> float:
    return float(np.clip(steps / search.t_max, 0.0, 1.0))


def efficiency_error(best_score: float, steps: int, search: SearchConfig, metric: MetricConfig) -> float:
    return normalized_regret(best_score) + metric.cost_weight * normalized_cost(steps, search)
