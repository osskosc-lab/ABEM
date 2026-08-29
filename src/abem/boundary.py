from __future__ import annotations

import math
from dataclasses import dataclass

from .config import SearchConfig


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class AdaptiveBoundary:
    config: SearchConfig
    cumulative_hazard: float = 0.0

    def reset(self) -> None:
        self.cumulative_hazard = 0.0

    def step(
        self,
        *,
        gain: float,
        uncertainty: float,
        diversity: float,
        stagnation: int,
    ) -> float:
        x = (
            self.config.b0
            + self.config.b_stagnation * stagnation
            - self.config.b_gain * max(gain, 0.0)
            - self.config.b_uncertainty * max(uncertainty, 0.0)
            - self.config.b_diversity * max(diversity, 0.0)
        )
        h = sigmoid(x)
        self.cumulative_hazard += h
        return h

    def should_stop(self, t: int) -> bool:
        steps = t + 1
        if steps < self.config.t_min:
            return False
        if steps >= self.config.t_max:
            return True
        return self.cumulative_hazard >= self.config.hazard_threshold
