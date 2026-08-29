from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LandscapeConfig:
    dimension: int = 32
    block_size: int = 4
    interaction_strength: float = 0.25
    deceptive_bonus: float = 0.72
    optimum_bonus: float = 1.0
    interaction_edges: int = 8
    decoy_compatible_edge_fraction: float = 0.0


@dataclass(frozen=True)
class SearchConfig:
    population_size: int = 24
    elite_size: int = 6
    mutation_rate: float = 0.08
    memory_bias: float = 0.35
    t_min: int = 4
    t_max: int = 64
    fixed_depth: int = 16
    hazard_threshold: float = 4.0
    b0: float = -1.0
    b_stagnation: float = 0.25
    b_gain: float = 5.0
    b_uncertainty: float = 1.2
    b_diversity: float = 1.0


@dataclass(frozen=True)
class MemoryConfig:
    learning_rate: float = 0.12
    decay: float = 0.08
    clip: float = 2.5


@dataclass(frozen=True)
class MetricConfig:
    cost_weight: float = 0.25
    support_ratio: float = 0.90
    bootstrap_samples: int = 2000
    mechanism_margin: float = 0.02


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "phase0"
    seeds: tuple[int, ...] = field(default_factory=lambda: tuple(range(5)))
    episodes_per_seed: int = 12
    confirmatory: bool = False
    landscape: LandscapeConfig = field(default_factory=LandscapeConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)


def _construct(cls: type, data: dict[str, Any] | None):
    return cls(**(data or {}))


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return ExperimentConfig(
        name=raw.get("name", "phase0"),
        seeds=tuple(int(x) for x in raw.get("seeds", range(5))),
        episodes_per_seed=int(raw.get("episodes_per_seed", 12)),
        confirmatory=bool(raw.get("confirmatory", False)),
        landscape=_construct(LandscapeConfig, raw.get("landscape")),
        search=_construct(SearchConfig, raw.get("search")),
        memory=_construct(MemoryConfig, raw.get("memory")),
        metric=_construct(MetricConfig, raw.get("metric")),
    )
