from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MetricConfig


@dataclass(frozen=True)
class BootstrapRatio:
    ratio: float
    lower: float
    upper: float


def paired_bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    samples: int,
    seed: int = 20260829,
) -> BootstrapRatio:
    a = np.asarray(numerator, dtype=float)
    b = np.asarray(denominator, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired bootstrap arrays must be same-shape 1D arrays")
    if len(a) < 2:
        mean_b = float(np.mean(b))
        ratio = float(np.mean(a) / mean_b) if mean_b > 0 else float("inf")
        return BootstrapRatio(ratio=ratio, lower=ratio, upper=ratio)

    mean_b = float(np.mean(b))
    observed = float(np.mean(a) / mean_b) if mean_b > 0 else float("inf")
    rng = np.random.default_rng(seed)
    boots = np.empty(samples, dtype=float)
    n = len(a)
    for k in range(samples):
        idx = rng.integers(0, n, size=n)
        denom = float(np.mean(b[idx]))
        boots[k] = float(np.mean(a[idx]) / denom) if denom > 0 else np.inf

    finite = boots[np.isfinite(boots)]
    if finite.size == 0:
        return BootstrapRatio(observed, float("inf"), float("inf"))
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return BootstrapRatio(observed, float(lower), float(upper))


def classify_verdict(
    *,
    base: np.ndarray,
    ab: np.ndarray,
    mem: np.ndarray,
    abem: np.ndarray,
    memory_shuffled: np.ndarray,
    boundary_clamp: np.ndarray,
    metric: MetricConfig,
) -> tuple[str, dict[str, float | bool]]:
    ratio = paired_bootstrap_ratio(
        abem,
        base,
        samples=metric.bootstrap_samples,
    )
    support = ratio.ratio <= metric.support_ratio and ratio.upper < 1.0

    ab_mean = float(np.mean(ab))
    mem_mean = float(np.mean(mem))
    abem_mean = float(np.mean(abem))
    shuffled_mean = float(np.mean(memory_shuffled))
    clamp_mean = float(np.mean(boundary_clamp))

    memory_mechanism = shuffled_mean >= abem_mean + metric.mechanism_margin
    boundary_mechanism = clamp_mean >= abem_mean + metric.mechanism_margin

    boundary_only_support = ab_mean <= metric.support_ratio * float(np.mean(base))
    memory_only_support = mem_mean <= metric.support_ratio * float(np.mean(base))

    if support and memory_mechanism and boundary_mechanism:
        verdict = "PASS_COMBINED"
    elif support and boundary_mechanism and not memory_mechanism:
        verdict = "PASS_BOUNDARY_ONLY"
    elif support and memory_mechanism and not boundary_mechanism:
        verdict = "PASS_MEMORY_ONLY"
    elif not support and boundary_only_support and not memory_only_support:
        verdict = "PASS_BOUNDARY_ONLY"
    elif not support and memory_only_support and not boundary_only_support:
        verdict = "PASS_MEMORY_ONLY"
    else:
        verdict = "NO_GO"

    diagnostics: dict[str, float | bool] = {
        "abem_over_base": ratio.ratio,
        "bootstrap_ci_lower": ratio.lower,
        "bootstrap_ci_upper": ratio.upper,
        "support_threshold_met": support,
        "memory_mechanism": memory_mechanism,
        "boundary_mechanism": boundary_mechanism,
        "mean_base": float(np.mean(base)),
        "mean_ab": ab_mean,
        "mean_mem": mem_mean,
        "mean_abem": abem_mean,
        "mean_memory_shuffled": shuffled_mean,
        "mean_boundary_clamp": clamp_mean,
    }
    return verdict, diagnostics
