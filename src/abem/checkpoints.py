from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .agents import _initialize_population, _mean_pairwise_hamming, _propose
from .boundary import AdaptiveBoundary
from .config import SearchConfig


@dataclass(frozen=True)
class SearchCheckpoint:
    """Sufficient state to replay or branch a memory-free search trajectory."""

    step: int
    population: np.ndarray
    population_scores: np.ndarray
    best_candidate: np.ndarray
    best_score: float
    previous_best: float
    last_improvement_step: int
    gain: float
    score_std: float
    diversity: float
    stagnation: int
    cumulative_hazard: float
    hazard: float
    rng_state: dict


@dataclass(frozen=True)
class Trajectory:
    checkpoints: tuple[SearchCheckpoint, ...]
    step_records: tuple[SearchCheckpoint, ...]

    def at_step(self, step: int) -> SearchCheckpoint:
        return self.step_records[step - 1]


def _snapshot(
    *,
    step: int,
    population: np.ndarray,
    scores: np.ndarray,
    best_candidate: np.ndarray,
    best_score: float,
    previous_best: float,
    last_improvement_step: int,
    gain: float,
    score_std: float,
    diversity: float,
    stagnation: int,
    boundary: AdaptiveBoundary,
    hazard: float,
    rng: np.random.Generator,
) -> SearchCheckpoint:
    return SearchCheckpoint(
        step=step,
        population=population.copy(),
        population_scores=scores.copy(),
        best_candidate=best_candidate.copy(),
        best_score=float(best_score),
        previous_best=float(previous_best),
        last_improvement_step=int(last_improvement_step),
        gain=float(gain),
        score_std=float(score_std),
        diversity=float(diversity),
        stagnation=int(stagnation),
        cumulative_hazard=float(boundary.cumulative_hazard),
        hazard=float(hazard),
        rng_state=copy.deepcopy(rng.bit_generator.state),
    )


def generate_trajectory(
    landscape,
    *,
    agent_seed: int,
    config: SearchConfig,
    checkpoint_steps: Iterable[int],
) -> Trajectory:
    """Run the kernel to ``t_max`` without allowing a stopping rule to affect it."""

    requested = {int(x) for x in checkpoint_steps}
    if any(x < 1 or x > config.t_max for x in requested):
        raise ValueError("checkpoint steps must be within [1, t_max]")

    rng = np.random.default_rng(agent_seed)
    population = _initialize_population(rng, landscape.dimension, config, memory=None)
    boundary = AdaptiveBoundary(config)
    boundary.reset()
    best_score = -np.inf
    best_candidate = population[0].copy()
    previous_best = -np.inf
    last_improvement_step = 0
    records: list[SearchCheckpoint] = []

    for t in range(config.t_max):
        scores = landscape.evaluate(population)
        idx = int(np.argmax(scores))
        current_best = float(scores[idx])
        if current_best > best_score + 1e-12:
            best_score = current_best
            best_candidate = population[idx].copy()
            last_improvement_step = t

        gain = 0.0 if not np.isfinite(previous_best) else max(0.0, best_score - previous_best)
        score_std = float(np.std(scores))
        diversity = _mean_pairwise_hamming(population)
        stagnation = t - last_improvement_step
        hazard = boundary.step(
            gain=gain,
            uncertainty=score_std,
            diversity=diversity,
            stagnation=stagnation,
        )
        records.append(
            _snapshot(
                step=t + 1,
                population=population,
                scores=scores,
                best_candidate=best_candidate,
                best_score=best_score,
                previous_best=previous_best,
                last_improvement_step=last_improvement_step,
                gain=gain,
                score_std=score_std,
                diversity=diversity,
                stagnation=stagnation,
                boundary=boundary,
                hazard=hazard,
                rng=rng,
            )
        )

        if t + 1 >= config.t_max:
            break
        elite_n = min(config.elite_size, config.population_size)
        elites = population[np.argsort(scores)[-elite_n:]]
        population = _propose(rng, elites, landscape.dimension, config, memory=None)
        previous_best = best_score

    checkpoints = tuple(record for record in records if record.step in requested)
    return Trajectory(checkpoints=checkpoints, step_records=tuple(records))


def continue_from_checkpoint(
    landscape,
    checkpoint: SearchCheckpoint,
    *,
    config: SearchConfig,
    horizon: int,
    future_seed: int | None = None,
) -> SearchCheckpoint:
    """Continue a checkpoint for at most ``horizon`` new evaluation steps.

    ``future_seed=None`` restores the saved RNG exactly. Supplying a seed creates
    an independent counterfactual future and never reads the generating future.
    """

    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if horizon == 0 or checkpoint.step >= config.t_max:
        return checkpoint

    rng = np.random.default_rng(future_seed)
    if future_seed is None:
        rng.bit_generator.state = copy.deepcopy(checkpoint.rng_state)

    population = checkpoint.population.copy()
    scores = checkpoint.population_scores.copy()
    best_score = checkpoint.best_score
    best_candidate = checkpoint.best_candidate.copy()
    previous_best = checkpoint.previous_best
    last_improvement_step = checkpoint.last_improvement_step
    boundary = AdaptiveBoundary(config, cumulative_hazard=checkpoint.cumulative_hazard)
    current = checkpoint
    final_step = min(config.t_max, checkpoint.step + int(horizon))

    for step in range(checkpoint.step + 1, final_step + 1):
        elite_n = min(config.elite_size, config.population_size)
        elites = population[np.argsort(scores)[-elite_n:]]
        population = _propose(rng, elites, landscape.dimension, config, memory=None)
        previous_best = best_score
        scores = landscape.evaluate(population)
        idx = int(np.argmax(scores))
        current_best = float(scores[idx])
        if current_best > best_score + 1e-12:
            best_score = current_best
            best_candidate = population[idx].copy()
            last_improvement_step = step - 1

        gain = max(0.0, best_score - previous_best)
        score_std = float(np.std(scores))
        diversity = _mean_pairwise_hamming(population)
        stagnation = (step - 1) - last_improvement_step
        hazard = boundary.step(
            gain=gain,
            uncertainty=score_std,
            diversity=diversity,
            stagnation=stagnation,
        )
        current = _snapshot(
            step=step,
            population=population,
            scores=scores,
            best_candidate=best_candidate,
            best_score=best_score,
            previous_best=previous_best,
            last_improvement_step=last_improvement_step,
            gain=gain,
            score_std=score_std,
            diversity=diversity,
            stagnation=stagnation,
            boundary=boundary,
            hazard=hazard,
            rng=rng,
        )

    return current
