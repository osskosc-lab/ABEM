from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .agents import run_episode
from .config import ExperimentConfig, load_config
from .landscapes import make_landscape
from .memory import SearchMemory
from .metrics import efficiency_error
from .statistics import classify_verdict


CONDITIONS = (
    "BASE",
    "AB",
    "MEM",
    "ABEM",
    "MEMORY_SHUFFLED",
    "BOUNDARY_CLAMP",
)


def _problem_seed(seed: int, episode: int) -> int:
    return 1_000_003 + seed * 10_007 + episode * 101


def _agent_seed(seed: int, episode: int) -> int:
    return 7_000_001 + seed * 20_011 + episode * 211


def _derangement(n: int, seed: int) -> np.ndarray:
    if n <= 1:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    base = np.arange(n)
    for _ in range(1000):
        p = rng.permutation(n)
        if np.all(p != base):
            return p
    return np.roll(base, 1)


def _run_one(
    cfg: ExperimentConfig,
    *,
    landscape,
    seed: int,
    episode: int,
    memory: SearchMemory | None,
    adaptive: bool,
):
    rng = np.random.default_rng(_agent_seed(seed, episode))
    result = run_episode(
        landscape,
        rng=rng,
        config=cfg.search,
        memory=memory,
        adaptive_boundary=adaptive,
    )
    error = efficiency_error(result.best_score, result.steps, cfg.search, cfg.metric)
    return result, error


def run_seed(cfg: ExperimentConfig, seed: int) -> tuple[dict[str, float], list[dict]]:
    landscapes = [
        make_landscape(_problem_seed(seed, e), cfg.landscape)
        for e in range(cfg.episodes_per_seed)
    ]
    rows: list[dict] = []
    values: dict[str, list[float]] = {name: [] for name in CONDITIONS}

    # ABEM runs first so its pre-episode memory states can be re-used as matched controls.
    abem_memory = SearchMemory.zeros(cfg.landscape.dimension, cfg.memory)
    abem_snapshots: list[SearchMemory] = []
    for e, landscape in enumerate(landscapes):
        abem_snapshots.append(abem_memory.copy())
        result, error = _run_one(
            cfg,
            landscape=landscape,
            seed=seed,
            episode=e,
            memory=abem_memory,
            adaptive=True,
        )
        values["ABEM"].append(error)
        rows.append(_row(seed, e, "ABEM", result.best_score, result.steps, error))
        abem_memory.update(result.trace_candidates, result.trace_scores)

    # Independent online conditions.
    mem_memory = SearchMemory.zeros(cfg.landscape.dimension, cfg.memory)
    for e, landscape in enumerate(landscapes):
        for condition, memory, adaptive in (
            ("BASE", None, False),
            ("AB", None, True),
            ("MEM", mem_memory, False),
        ):
            result, error = _run_one(
                cfg,
                landscape=landscape,
                seed=seed,
                episode=e,
                memory=memory,
                adaptive=adaptive,
            )
            values[condition].append(error)
            rows.append(_row(seed, e, condition, result.best_score, result.steps, error))
            if condition == "MEM":
                mem_memory.update(result.trace_candidates, result.trace_scores)

    # Mechanism controls preserve ABEM's actual memory-state distribution.
    permutation = _derangement(cfg.episodes_per_seed, seed + 99_991)
    for e, landscape in enumerate(landscapes):
        shuffled_memory = abem_snapshots[int(permutation[e])]
        result, error = _run_one(
            cfg,
            landscape=landscape,
            seed=seed,
            episode=e,
            memory=shuffled_memory,
            adaptive=True,
        )
        values["MEMORY_SHUFFLED"].append(error)
        rows.append(_row(seed, e, "MEMORY_SHUFFLED", result.best_score, result.steps, error))

        result, error = _run_one(
            cfg,
            landscape=landscape,
            seed=seed,
            episode=e,
            memory=abem_snapshots[e],
            adaptive=False,
        )
        values["BOUNDARY_CLAMP"].append(error)
        rows.append(_row(seed, e, "BOUNDARY_CLAMP", result.best_score, result.steps, error))

    seed_means = {name: float(np.mean(vals)) for name, vals in values.items()}
    return seed_means, rows


def _row(seed: int, episode: int, condition: str, score: float, steps: int, error: float) -> dict:
    return {
        "seed": seed,
        "episode": episode,
        "condition": condition,
        "best_score": float(score),
        "steps": int(steps),
        "efficiency_error": float(error),
    }


def run_experiment(cfg: ExperimentConfig) -> dict:
    per_seed: dict[str, list[float]] = {name: [] for name in CONDITIONS}
    rows: list[dict] = []

    for seed in cfg.seeds:
        means, seed_rows = run_seed(cfg, int(seed))
        rows.extend(seed_rows)
        for condition in CONDITIONS:
            per_seed[condition].append(means[condition])

    arrays = {k: np.asarray(v, dtype=float) for k, v in per_seed.items()}
    verdict, diagnostics = classify_verdict(
        base=arrays["BASE"],
        ab=arrays["AB"],
        mem=arrays["MEM"],
        abem=arrays["ABEM"],
        memory_shuffled=arrays["MEMORY_SHUFFLED"],
        boundary_clamp=arrays["BOUNDARY_CLAMP"],
        metric=cfg.metric,
    )

    if not cfg.confirmatory:
        verdict = "NON_CONFIRMATORY_DO_NOT_INTERPRET"

    return {
        "phase": cfg.name,
        "confirmatory": cfg.confirmatory,
        "verdict": verdict,
        "diagnostics": diagnostics,
        "per_seed_efficiency_error": per_seed,
        "records": rows,
        "config": asdict(cfg),
        "claim_firewall": (
            "Phase 0 evaluates synthetic search efficiency only; it does not establish "
            "quantum decoherence control, consciousness, creativity, or autonomous breakthroughs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ABEM Phase 0")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_experiment(cfg)
    output = Path(args.output) if args.output else Path("results") / cfg.name / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"phase": result["phase"], "verdict": result["verdict"], "diagnostics": result["diagnostics"]}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
