from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary import AdaptiveBoundary
from .config import SearchConfig
from .landscapes import DeceptiveModularLandscape
from .memory import SearchMemory


@dataclass(frozen=True)
class EpisodeResult:
    best_score: float
    best_candidate: np.ndarray
    steps: int
    evaluations: int
    trace_candidates: np.ndarray
    trace_scores: np.ndarray


def _memory_probabilities(memory: SearchMemory | None, config: SearchConfig, dimension: int) -> np.ndarray:
    if memory is None:
        return np.full(dimension, 0.5, dtype=float)
    logits = np.clip(config.memory_bias * memory.weights, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _initialize_population(
    rng: np.random.Generator,
    dimension: int,
    config: SearchConfig,
    memory: SearchMemory | None,
) -> np.ndarray:
    probs = _memory_probabilities(memory, config, dimension)
    return (rng.random((config.population_size, dimension)) < probs).astype(np.int8)


def _mean_pairwise_hamming(population: np.ndarray) -> float:
    n = population.shape[0]
    if n < 2:
        return 0.0
    # For each bit, k * (n-k) unordered pairs disagree. This is exactly the
    # loop definition above but avoids a Python loop inside every Oracle rollout.
    ones = np.sum(population, axis=0, dtype=np.int64)
    disagreeing_bits = np.sum(ones * (n - ones), dtype=np.int64)
    pairs = n * (n - 1) / 2
    return float(disagreeing_bits / (pairs * population.shape[1]))


def _propose(
    rng: np.random.Generator,
    elites: np.ndarray,
    dimension: int,
    config: SearchConfig,
    memory: SearchMemory | None,
) -> np.ndarray:
    parent_idx = rng.integers(0, len(elites), size=config.population_size)
    children = elites[parent_idx].copy()
    mutation_mask = rng.random(children.shape) < config.mutation_rate
    memory_probs = _memory_probabilities(memory, config, dimension)
    replacement = (rng.random(children.shape) < memory_probs[None, :]).astype(np.int8)
    children[mutation_mask] = replacement[mutation_mask]

    # Small unbiased restart stream prevents memory from becoming a hard constraint.
    restart_n = max(1, config.population_size // 8)
    children[:restart_n] = rng.integers(0, 2, size=(restart_n, dimension), dtype=np.int8)
    return children


def run_episode(
    landscape: DeceptiveModularLandscape,
    *,
    rng: np.random.Generator,
    config: SearchConfig,
    memory: SearchMemory | None,
    adaptive_boundary: bool,
) -> EpisodeResult:
    population = _initialize_population(rng, landscape.dimension, config, memory)
    boundary = AdaptiveBoundary(config)
    boundary.reset()

    best_score = -np.inf
    best_candidate = population[0].copy()
    previous_best = -np.inf
    last_improvement_step = 0
    trace_candidates: list[np.ndarray] = []
    trace_scores: list[np.ndarray] = []

    for t in range(config.t_max):
        scores = landscape.evaluate(population)
        trace_candidates.append(population.copy())
        trace_scores.append(scores.copy())

        idx = int(np.argmax(scores))
        current_best = float(scores[idx])
        if current_best > best_score + 1e-12:
            best_score = current_best
            best_candidate = population[idx].copy()
            last_improvement_step = t

        gain = 0.0 if not np.isfinite(previous_best) else max(0.0, best_score - previous_best)
        uncertainty = float(np.std(scores))
        diversity = _mean_pairwise_hamming(population)
        stagnation = t - last_improvement_step

        boundary.step(
            gain=gain,
            uncertainty=uncertainty,
            diversity=diversity,
            stagnation=stagnation,
        )

        steps = t + 1
        stop = boundary.should_stop(t) if adaptive_boundary else steps >= min(config.fixed_depth, config.t_max)
        if stop:
            break

        elite_n = min(config.elite_size, config.population_size)
        elite_idx = np.argsort(scores)[-elite_n:]
        elites = population[elite_idx]
        population = _propose(rng, elites, landscape.dimension, config, memory)
        previous_best = best_score

    all_candidates = np.concatenate(trace_candidates, axis=0)
    all_scores = np.concatenate(trace_scores, axis=0)
    return EpisodeResult(
        best_score=float(best_score),
        best_candidate=best_candidate,
        steps=steps,
        evaluations=int(all_scores.size),
        trace_candidates=all_candidates,
        trace_scores=all_scores,
    )
