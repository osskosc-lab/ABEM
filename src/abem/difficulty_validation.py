from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .boundary_analysis import efficiency_at_step
from .checkpoints import generate_trajectory
from .config import LandscapeConfig, MetricConfig, SearchConfig
from .experiment import _agent_seed, _problem_seed
from .landscapes import make_landscape


def generator_config(raw: dict, name: str) -> LandscapeConfig:
    base = dict(raw["landscape"])
    base.update(raw["generators"][name])
    return LandscapeConfig(**base)


def validate_difficulty(raw: dict) -> tuple[list[dict], dict]:
    search = SearchConfig(**raw["search"])
    metric = MetricConfig(**raw["metric"])
    depth = int(raw["d0"]["fixed_search_depth"])
    rows: list[dict] = []
    for seed in raw["d0"]["seeds"]:
        for generator in raw["generators"]:
            errors = []
            for episode in range(int(raw["d0"]["episodes_per_seed"])):
                problem_seed = _problem_seed(int(seed), episode)
                trajectory = generate_trajectory(
                    make_landscape(problem_seed, generator_config(raw, generator)),
                    agent_seed=_agent_seed(int(seed), episode),
                    config=search,
                    checkpoint_steps=(),
                )
                errors.append(efficiency_at_step(trajectory, depth, search=search, metric=metric))
            rows.append(
                {
                    "seed": int(seed),
                    "generator": generator,
                    "mean_efficiency_error": float(np.mean(errors)),
                    "fixed_search_depth": depth,
                    "episodes": len(errors),
                }
            )
    means = {
        name: float(np.mean([row["mean_efficiency_error"] for row in rows if row["generator"] == name]))
        for name in raw["generators"]
    }
    order = list(raw["generators"])
    passed = all(means[left] < means[right] for left, right in zip(order, order[1:]))
    return rows, {
        "pass": bool(passed),
        "required_order": " < ".join(order),
        "observed_mean_efficiency_error": means,
        "generator_configs": {name: asdict(generator_config(raw, name)) for name in order},
    }
