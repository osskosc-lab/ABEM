from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np

from .agents import run_episode
from .config import load_config
from .experiment import _agent_seed, _problem_seed
from .landscapes import make_landscape
from .metrics import efficiency_error


def calibrate_fixed_depth(config_path: str, depths: tuple[int, ...]) -> dict:
    cfg = load_config(config_path)
    if cfg.confirmatory:
        raise ValueError("baseline calibration is forbidden on confirmatory data")

    means: dict[int, float] = {}
    for depth in depths:
        search = replace(cfg.search, fixed_depth=int(depth))
        seed_scores = []
        for seed in cfg.seeds:
            episode_scores = []
            for episode in range(cfg.episodes_per_seed):
                landscape = make_landscape(_problem_seed(int(seed), episode), cfg.landscape)
                result = run_episode(
                    landscape,
                    rng=np.random.default_rng(_agent_seed(int(seed), episode)),
                    config=search,
                    memory=None,
                    adaptive_boundary=False,
                )
                episode_scores.append(efficiency_error(result.best_score, result.steps, search, cfg.metric))
            seed_scores.append(float(np.mean(episode_scores)))
        means[int(depth)] = float(np.mean(seed_scores))

    best = min(means, key=means.get)
    return {
        "config": config_path,
        "candidate_depths": list(depths),
        "mean_efficiency_error": {str(k): v for k, v in means.items()},
        "recommended_fixed_depth": int(best),
        "instruction": "Freeze the selected depth before opening confirmatory results.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate ABEM fixed-depth BASE on non-confirmatory data")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--depths", nargs="+", type=int, default=[8, 16, 32, 64])
    args = parser.parse_args()
    result = calibrate_fixed_depth(args.config, tuple(args.depths))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
