from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MemoryConfig


@dataclass
class SearchMemory:
    weights: np.ndarray
    config: MemoryConfig

    @classmethod
    def zeros(cls, dimension: int, config: MemoryConfig) -> "SearchMemory":
        return cls(weights=np.zeros(dimension, dtype=float), config=config)

    def copy(self) -> "SearchMemory":
        return SearchMemory(self.weights.copy(), self.config)

    def update(self, candidates: np.ndarray, scores: np.ndarray) -> None:
        candidates = np.asarray(candidates, dtype=float)
        scores = np.asarray(scores, dtype=float)
        if candidates.ndim != 2 or scores.ndim != 1:
            raise ValueError("invalid trace shapes")
        if candidates.shape[0] != scores.shape[0]:
            raise ValueError("candidate/score trace length mismatch")
        if candidates.shape[0] == 0:
            return

        centered = scores - float(np.mean(scores))
        directions = 2.0 * candidates - 1.0
        denom = float(np.sum(np.abs(centered))) + 1e-12
        delta = np.sum(centered[:, None] * directions, axis=0) / denom

        self.weights = (
            (1.0 - self.config.decay) * self.weights
            + self.config.learning_rate * delta
        )
        self.weights = np.clip(self.weights, -self.config.clip, self.config.clip)

    def shuffled_copy(self, rng: np.random.Generator) -> "SearchMemory":
        shuffled = self.weights.copy()
        rng.shuffle(shuffled)
        return SearchMemory(shuffled, self.config)
