from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .boundary import sigmoid
from .checkpoints import SearchCheckpoint, Trajectory
from .config import MetricConfig, SearchConfig
from .metrics import efficiency_error


@dataclass(frozen=True)
class BoundaryParameters:
    hazard_threshold: float
    b0: float
    b_stagnation: float
    b_gain: float
    b_uncertainty: float
    b_diversity: float

    @classmethod
    def from_search(cls, search: SearchConfig) -> "BoundaryParameters":
        return cls(
            hazard_threshold=search.hazard_threshold,
            b0=search.b0,
            b_stagnation=search.b_stagnation,
            b_gain=search.b_gain,
            b_uncertainty=search.b_uncertainty,
            b_diversity=search.b_diversity,
        )

    def without(self, feature: str) -> "BoundaryParameters":
        mapping = {
            "gain": "b_gain",
            "score_std": "b_uncertainty",
            "diversity": "b_diversity",
            "stagnation": "b_stagnation",
        }
        return replace(self, **{mapping[feature]: 0.0})


def instantaneous_hazard(record: SearchCheckpoint, params: BoundaryParameters) -> float:
    x = (
        params.b0
        + params.b_stagnation * record.stagnation
        - params.b_gain * max(record.gain, 0.0)
        - params.b_uncertainty * max(record.score_std, 0.0)
        - params.b_diversity * max(record.diversity, 0.0)
    )
    return sigmoid(x)


def adaptive_stop_step(
    trajectory: Trajectory,
    *,
    search: SearchConfig,
    params: BoundaryParameters,
    feature_records: tuple[SearchCheckpoint, ...] | None = None,
) -> int:
    records = feature_records or trajectory.step_records
    if len(records) != len(trajectory.step_records):
        raise ValueError("feature record count mismatch")
    cumulative = 0.0
    for actual, feature in zip(trajectory.step_records, records, strict=True):
        cumulative += instantaneous_hazard(feature, params)
        if actual.step >= search.t_min and cumulative >= params.hazard_threshold:
            return actual.step
    return trajectory.step_records[-1].step


def patience_stop_step(trajectory: Trajectory, *, patience: int, t_min: int) -> int:
    for record in trajectory.step_records:
        if record.step >= t_min and record.stagnation >= patience:
            return record.step
    return trajectory.step_records[-1].step


def efficiency_at_step(
    trajectory: Trajectory,
    step: int,
    *,
    search: SearchConfig,
    metric: MetricConfig,
) -> float:
    record = trajectory.at_step(int(step))
    return efficiency_error(record.best_score, int(step), search, metric)


def shuffled_feature_records(trajectory: Trajectory, seed: int) -> tuple[SearchCheckpoint, ...]:
    """Independently shuffle feature times while preserving every marginal."""

    rng = np.random.default_rng(seed)
    n = len(trajectory.step_records)
    features = {
        name: np.asarray([getattr(x, name) for x in trajectory.step_records])
        for name in ("gain", "score_std", "diversity", "stagnation")
    }
    for values in features.values():
        rng.shuffle(values)
    return tuple(
        replace(
            record,
            gain=float(features["gain"][i]),
            score_std=float(features["score_std"][i]),
            diversity=float(features["diversity"][i]),
            stagnation=int(features["stagnation"][i]),
        )
        for i, record in enumerate(trajectory.step_records)
    )


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(scores, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("labels and scores must be same-shape vectors")
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        auroc = float("nan")
    else:
        order = np.argsort(p, kind="mergesort")
        ranks = np.empty(len(p), dtype=float)
        i = 0
        while i < len(p):
            j = i + 1
            while j < len(p) and p[order[j]] == p[order[i]]:
                j += 1
            ranks[order[i:j]] = (i + 1 + j) / 2.0
            i = j
        rank_sum = float(np.sum(ranks[y == 1]))
        auroc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

    order = np.argsort(-p, kind="mergesort")
    sorted_y = y[order]
    tp = np.cumsum(sorted_y == 1)
    precision = tp / np.arange(1, len(y) + 1)
    auprc = float(np.sum(precision[sorted_y == 1]) / positives) if positives else float("nan")
    brier = float(np.mean((np.clip(p, 0.0, 1.0) - y) ** 2))
    return {"auroc": float(auroc), "auprc": auprc, "brier": brier}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        out = np.empty(len(values), dtype=float)
        i = 0
        while i < len(values):
            j = i + 1
            while j < len(values) and values[order[j]] == values[order[i]]:
                j += 1
            out[order[i:j]] = (i + 1 + j) / 2.0
            i = j
        return out

    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def paired_bootstrap_difference(
    a: np.ndarray,
    b: np.ndarray,
    *,
    samples: int,
    seed: int = 20260829,
) -> dict[str, float | int]:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired arrays must be same-shape vectors")
    diff = left - right
    rng = np.random.default_rng(seed)
    boots = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = rng.integers(0, len(diff), size=len(diff))
        boots[i] = float(np.mean(diff[idx]))
    lower, upper = np.quantile(boots, [0.025, 0.975])
    return {
        "mean": float(np.mean(diff)),
        "median": float(np.median(diff)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "negative_seeds": int(np.sum(diff < 0)),
        "total_seeds": int(len(diff)),
    }
