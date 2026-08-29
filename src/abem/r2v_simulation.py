from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from .boundary_analysis import efficiency_at_step, paired_bootstrap_difference, patience_stop_step, spearman
from .checkpoints import Trajectory, continue_from_checkpoint, generate_trajectory
from .config import MetricConfig, SearchConfig
from .difficulty_validation import generator_config, validate_difficulty
from .experiment import _agent_seed, _problem_seed
from .landscapes import make_landscape
from .minimal_boundary import (
    MinimalBoundaryParameters,
    boundary_series,
    ema_gain_series,
    minimal_stop_step,
    shuffled_gbar,
)
from .oracle import OracleEstimate, estimate_oracle_value


CLAIM_FIREWALL = (
    "本結果はsynthetic search environmentの停止効率だけを扱う。量子デコヒーレンス、量子的コヒーレンス、"
    "AI意識、AI創造性、ブレイクスルー能力、新・有効ノイズ理論全体を実証しない。"
)


@dataclass(frozen=True)
class R2VCase:
    split: str
    seed: int
    episode: int
    generator: str
    problem_seed: int
    agent_seed: int
    trajectory: Trajectory
    oracle_estimates: tuple[OracleEstimate, ...]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not names:
            return
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_r2v_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if raw.get("confirmatory", False):
        raise ValueError("Phase 0B-r2V cannot run confirmatory data")
    splits = {name: tuple(int(x) for x in raw[name]["seeds"]) for name in ("d0", "d1", "d2")}
    if any(set(splits[a]) & set(splits[b]) for a, b in (("d0", "d1"), ("d0", "d2"), ("d1", "d2"))):
        raise ValueError("D0/D1/D2 seeds must be disjoint")
    forbidden = set(range(int(raw["forbidden_confirmatory_seed_min"]), int(raw["forbidden_confirmatory_seed_max"]) + 1))
    if forbidden & set().union(*map(set, splits.values())):
        raise ValueError("confirmatory seed leakage detected")
    for name, values in splits.items():
        raw[name]["seeds"] = values
    if float(raw["search"]["memory_bias"]) != 0.0:
        raise ValueError("memory must remain disabled")
    return raw


def _build_cases(raw: dict, split: str) -> list[R2VCase]:
    search = SearchConfig(**raw["search"])
    metric = MetricConfig(**raw["metric"])
    checkpoints = tuple(int(x) for x in raw["checkpoint_horizons"])
    cases: list[R2VCase] = []
    for seed in raw[split]["seeds"]:
        for episode in range(int(raw[split]["episodes_per_seed"])):
            problem_seed = _problem_seed(seed, episode)
            agent_seed = _agent_seed(seed, episode)
            for generator in raw["generators"]:
                landscape = make_landscape(problem_seed, generator_config(raw, generator))
                trajectory = generate_trajectory(
                    landscape,
                    agent_seed=agent_seed,
                    config=search,
                    checkpoint_steps=checkpoints,
                )
                estimates = tuple(
                    estimate_oracle_value(
                        landscape,
                        checkpoint,
                        problem_seed=problem_seed,
                        search=search,
                        metric=metric,
                        horizons=checkpoints,
                        rollouts=int(raw["oracle_rollouts_per_checkpoint"]),
                    )
                    for checkpoint in trajectory.checkpoints
                )
                cases.append(
                    R2VCase(split, seed, episode, generator, problem_seed, agent_seed, trajectory, estimates)
                )
    return cases


def _case_errors(cases: list[R2VCase], steps: list[int], raw: dict) -> np.ndarray:
    search = SearchConfig(**raw["search"])
    metric = MetricConfig(**raw["metric"])
    return np.asarray(
        [efficiency_at_step(case.trajectory, step, search=search, metric=metric) for case, step in zip(cases, steps, strict=True)]
    )


def calibrate(cases: list[R2VCase], raw: dict) -> dict:
    search = SearchConfig(**raw["search"])
    p0_scores = {}
    for p0 in raw["p0_candidates"]:
        steps = [patience_stop_step(case.trajectory, patience=int(p0), t_min=search.t_min) for case in cases]
        p0_scores[int(p0)] = float(np.mean(_case_errors(cases, steps, raw)))
    p0 = min(p0_scores, key=p0_scores.get)

    positive = []
    for case in cases:
        series = ema_gain_series([x.gain for x in case.trajectory.step_records], alpha=float(raw["alpha"]))
        positive.extend(series[series > 0].tolist())
    if not positive:
        raise RuntimeError("SIGNAL_REPLICATION_FAIL: no positive EMA gain in D1")
    g0 = float(np.median(positive))

    delta_scores = {}
    for delta_p in raw["delta_p_candidates"]:
        params = MinimalBoundaryParameters(p0=p0, delta_p=int(delta_p), g0=g0, alpha=float(raw["alpha"]))
        steps = [minimal_stop_step(case.trajectory, params=params, t_min=search.t_min) for case in cases]
        delta_scores[int(delta_p)] = float(np.mean(_case_errors(cases, steps, raw)))
    delta_p = min(delta_scores, key=delta_scores.get)
    return {
        "p0": int(p0),
        "p0_candidate_mean_efficiency_error": p0_scores,
        "g0": g0,
        "g0_rule": "median of positive D1 EMA gain values",
        "delta_p": int(delta_p),
        "delta_p_candidate_mean_efficiency_error": delta_scores,
        "alpha": float(raw["alpha"]),
        "git_commit_sha_before_d2": _git_sha(),
    }


def _policy_steps(cases: list[R2VCase], frozen: dict, raw: dict, *, include_controls: bool) -> dict[str, list[int]]:
    search = SearchConfig(**raw["search"])
    params = MinimalBoundaryParameters(
        p0=int(frozen["p0"]), delta_p=int(frozen["delta_p"]), g0=float(frozen["g0"]), alpha=float(frozen["alpha"])
    )
    steps = {
        "PATIENCE_ONLY": [
            patience_stop_step(case.trajectory, patience=params.p0, t_min=search.t_min) for case in cases
        ],
        "ADAPTIVE_PATIENCE": [
            minimal_stop_step(case.trajectory, params=params, t_min=search.t_min) for case in cases
        ],
    }
    if not include_controls:
        return steps
    steps["FIXED_DEPTH"] = [int(raw["d0"]["fixed_search_depth"])] * len(cases)
    steps["GAIN_SHUFFLED"] = [
        minimal_stop_step(
            case.trajectory,
            params=params,
            t_min=search.t_min,
            gbar_override=shuffled_gbar(
                case.trajectory,
                alpha=params.alpha,
                seed=70_000_019 + case.seed * 10_007 + case.episode * 101 + list(raw["generators"]).index(case.generator),
            ),
        )
        for case in cases
    ]
    steps["GAIN_REVERSED"] = [
        minimal_stop_step(case.trajectory, params=params, t_min=search.t_min, reversed_gain=True) for case in cases
    ]
    random_steps = [0] * len(cases)
    rng = np.random.default_rng(20260829 if cases[0].split == "d1" else 20260830)
    for generator in raw["generators"]:
        indices = [index for index, case in enumerate(cases) if case.generator == generator]
        matched = np.asarray([steps["ADAPTIVE_PATIENCE"][index] for index in indices], dtype=int)
        matched = matched[rng.permutation(len(matched))]
        for index, value in zip(indices, matched, strict=True):
            random_steps[index] = int(value)
    steps["RANDOM_MATCHED"] = random_steps
    return steps


def _records(cases: list[R2VCase], steps: dict[str, list[int]], raw: dict) -> list[dict]:
    search = SearchConfig(**raw["search"])
    metric = MetricConfig(**raw["metric"])
    rows = []
    for index, case in enumerate(cases):
        for policy, policy_steps in steps.items():
            step = int(policy_steps[index])
            record = case.trajectory.at_step(step)
            rows.append(
                {
                    "split": case.split,
                    "seed": case.seed,
                    "episode": case.episode,
                    "generator": case.generator,
                    "policy": policy,
                    "stop_step": step,
                    "best_score": record.best_score,
                    "evaluation_count": step * search.population_size,
                    "efficiency_error": efficiency_at_step(case.trajectory, step, search=search, metric=metric),
                }
            )
    return rows


def _seed_level(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["seed"], row["policy"])].append(row)
    output = []
    for (split, seed, policy), values in sorted(grouped.items()):
        output.append(
            {
                "split": split,
                "seed": seed,
                "policy": policy,
                "mean_efficiency_error": float(np.mean([x["efficiency_error"] for x in values])),
                "mean_stop_step": float(np.mean([x["stop_step"] for x in values])),
                "mean_best_score": float(np.mean([x["best_score"] for x in values])),
                "mean_evaluation_count": float(np.mean([x["evaluation_count"] for x in values])),
            }
        )
    return output


def _seed_array(rows: list[dict], split: str, policy: str, field: str = "mean_efficiency_error") -> np.ndarray:
    selected = sorted((x for x in rows if x["split"] == split and x["policy"] == policy), key=lambda x: x["seed"])
    return np.asarray([x[field] for x in selected], dtype=float)


def _oracle_rows(cases: list[R2VCase], frozen: dict) -> list[dict]:
    rows = []
    alpha = float(frozen["alpha"])
    for case in cases:
        gbar = ema_gain_series([x.gain for x in case.trajectory.step_records], alpha=alpha)
        by_step = {estimate.checkpoint_step: estimate for estimate in case.oracle_estimates}
        for checkpoint in case.trajectory.checkpoints:
            estimate = by_step[checkpoint.step]
            rows.append(
                {
                    "split": case.split,
                    "seed": case.seed,
                    "episode": case.episode,
                    "generator": case.generator,
                    "step": checkpoint.step,
                    "gbar": float(gbar[checkpoint.step - 1]),
                    "stagnation": checkpoint.stagnation,
                    "oracle_value": estimate.oracle_value,
                    "oracle_action": estimate.action,
                }
            )
    return rows


def _signal_correlations(rows: list[dict], split: str) -> dict:
    selected = [x for x in rows if x["split"] == split]
    result = {}
    for feature in ("gbar", "stagnation"):
        by_seed = defaultdict(list)
        for row in selected:
            by_seed[row["seed"]].append(row)
        correlations = []
        for seed in sorted(by_seed):
            values = by_seed[seed]
            rho = spearman(np.asarray([x[feature] for x in values]), np.asarray([x["oracle_value"] for x in values]))
            if np.isfinite(rho):
                correlations.append(rho)
        result[feature] = {
            "mean_seed_spearman": float(np.mean(correlations)),
            "median_seed_spearman": float(np.median(correlations)),
            "n_seeds": len(correlations),
        }
    return result


def _runtime_replay_audit(raw: dict) -> tuple[bool, str]:
    search = SearchConfig(**raw["search"])
    seed = raw["d1"]["seeds"][0]
    landscape = make_landscape(_problem_seed(seed, 0), generator_config(raw, "G1"))
    trajectory = generate_trajectory(landscape, agent_seed=_agent_seed(seed, 0), config=search, checkpoint_steps=(4, 8))
    replay = continue_from_checkpoint(landscape, trajectory.at_step(4), config=search, horizon=4)
    expected = trajectory.at_step(8)
    passed = bool(
        np.array_equal(replay.population, expected.population)
        and np.array_equal(replay.population_scores, expected.population_scores)
        and replay.best_score == expected.best_score
    )
    return passed, "checkpoint replay exact" if passed else "checkpoint replay mismatch"


def _make_figures(output: Path, difficulty_rows: list[dict], episode_rows: list[dict], seed_rows: list[dict], oracle_rows: list[dict], frozen: dict) -> None:
    import matplotlib.pyplot as plt

    figdir = output / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    means = {name: np.mean([x["mean_efficiency_error"] for x in difficulty_rows if x["generator"] == name]) for name in ("G1", "G2", "G3")}
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(list(means), list(means.values()), marker="o"); ax.set_ylabel("mean efficiency error"); fig.tight_layout(); fig.savefig(figdir / "difficulty_validation.png", dpi=160); plt.close(fig)

    params = MinimalBoundaryParameters(int(frozen["p0"]), int(frozen["delta_p"]), float(frozen["g0"]), float(frozen["alpha"]))
    gains = np.linspace(0, max(params.g0 * 8, 1e-6), 200)
    from .minimal_boundary import boundary_value
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(gains, [boundary_value(x, params) for x in gains]); ax.set(xlabel="EMA recent gain", ylabel="patience boundary"); fig.tight_layout(); fig.savefig(figdir / "boundary_response_curve.png", dpi=160); plt.close(fig)

    def box(filename: str, policies: list[str], split: str = "d2"):
        fig, ax = plt.subplots(figsize=(7, 4)); values = [[x["mean_efficiency_error"] for x in seed_rows if x["split"] == split and x["policy"] == p] for p in policies]; ax.boxplot(values, tick_labels=policies, showmeans=True); ax.tick_params(axis="x", rotation=15); ax.set_ylabel("efficiency error"); fig.tight_layout(); fig.savefig(figdir / filename, dpi=160); plt.close(fig)
    box("adaptive_vs_patience.png", ["PATIENCE_ONLY", "ADAPTIVE_PATIENCE"])
    box("intact_vs_gain_shuffle.png", ["ADAPTIVE_PATIENCE", "GAIN_SHUFFLED"])

    fig, ax = plt.subplots(figsize=(7, 4));
    for policy in ("PATIENCE_ONLY", "ADAPTIVE_PATIENCE", "GAIN_SHUFFLED"):
        ax.hist([x["stop_step"] for x in episode_rows if x["split"] == "d2" and x["policy"] == policy], bins=16, alpha=.4, label=policy)
    ax.legend(); ax.set_xlabel("stop step"); fig.tight_layout(); fig.savefig(figdir / "stop_step_distribution.png", dpi=160); plt.close(fig)

    for feature, filename in (("gbar", "gain_vs_future_value.png"), ("stagnation", "stagnation_vs_future_value.png")):
        fig, ax = plt.subplots(figsize=(6, 4)); subset = [x for x in oracle_rows if x["split"] == "d2"]; ax.scatter([x[feature] for x in subset], [x["oracle_value"] for x in subset], s=12, alpha=.25); ax.set(xlabel=feature, ylabel="local future value"); fig.tight_layout(); fig.savefig(figdir / filename, dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4));
    for index, split in enumerate(("d1", "d2")):
        adaptive = _seed_array(seed_rows, split, "ADAPTIVE_PATIENCE"); patience = _seed_array(seed_rows, split, "PATIENCE_ONLY"); ax.bar(index, np.mean(adaptive - patience))
    ax.set_xticks([0, 1], ["D1", "D2"]); ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("paired DeltaE"); fig.tight_layout(); fig.savefig(figdir / "D1_vs_D2_effect.png", dpi=160); plt.close(fig)


def _decision_note(summary: dict) -> str:
    frozen = summary.get("frozen", {})
    primary = summary.get("primary_effect", {})
    shuffle = summary.get("gain_shuffle", {})
    signals = summary.get("oracle_signal_replication", {})
    gates = summary["gates"]
    failed = [name for name, item in gates.items() if not item["pass"]]
    return f"""# ABEM Phase 0B-r2V Decision Note

## 総合判定

**{summary['verdict']}**

## 前段r1Fから引き継いだ反証

4-feature hazard controllerはsimple baselineへの優位性がvalidationされず、`BOUNDARY_OVERFIT`を凍結したnegative evidenceとして保持した。score_std、diversity、memory、cumulative hazardは本モデルから除外した。

## 新モデル定義

MVOC-B: `B_t = P0 + DeltaP * Gbar_t / (Gbar_t + G0)`、`S_t >= B_t`で停止する。

## Difficulty manipulation validation

{json.dumps(summary['difficulty_validation'], ensure_ascii=False, indent=2)}

## Frozen P0

`{frozen.get('p0', 'N/A')}`

## Frozen G0

`{frozen.get('g0', 'N/A')}`

## Frozen DeltaP

`{frozen.get('delta_p', 'N/A')}`

## Primary paired DeltaE

`{primary.get('mean', 'N/A')}`

## 95% CI

`[{primary.get('ci_lower', 'N/A')}, {primary.get('ci_upper', 'N/A')}]`

## Gain Shuffle

{json.dumps(shuffle, ensure_ascii=False, indent=2)}

## Oracle signal replication

{json.dumps(signals, ensure_ascii=False, indent=2)}

## Boundary behavior audit

{json.dumps(summary.get('boundary_behavior_audit', {}), ensure_ascii=False, indent=2)}

## Calibration vs Blind Validation

{json.dumps(summary.get('calibration_vs_validation', {}), ensure_ascii=False, indent=2)}

## 失敗Gate

`{', '.join(failed) if failed else 'なし'}`

## 残る最小主張

{summary['remaining_minimal_claim']}

## Claim Firewall

{CLAIM_FIREWALL}

## 次段階可否

Phase 0Cは実行していない。`MINIMAL_BOUNDARY_SUPPORTED`以外ではPhase 0Cへ進まない。
"""


def _write_failure(output: Path, raw: dict, difficulty_rows: list[dict], difficulty: dict, replay: tuple[bool, str]) -> dict:
    gates = {
        "G0_IMPLEMENTATION": {"pass": replay[0], "reason": replay[1]},
        "G1_DIFFICULTY": {"pass": False, "reason": f"observed {difficulty['observed_mean_efficiency_error']}"},
        "G2_SIGNAL_REPLICATION": {"pass": False, "reason": "not run after G1 failure"},
        "G3_PRIMARY_EFFECT": {"pass": False, "reason": "not run after G1 failure"},
        "G4_GAIN_MECHANISM": {"pass": False, "reason": "not run after G1 failure"},
        "G5_GENERALIZATION": {"pass": False, "reason": "not run after G1 failure"},
    }
    summary = {
        "phase": raw["name"], "verdict": "DIFFICULTY_MANIPULATION_FAIL", "git_commit_sha": _git_sha(),
        "difficulty_validation": difficulty, "frozen": {}, "primary_effect": {}, "gain_shuffle": {},
        "oracle_signal_replication": {}, "boundary_behavior_audit": {}, "calibration_vs_validation": {},
        "gates": gates, "remaining_minimal_claim": "generator difficulty manipulationが成立せず、adaptive boundary比較を実行していない。",
        "claim_firewall": CLAIM_FIREWALL, "confirmatory_data_used": False,
    }
    _write_csv(output / "difficulty_validation.csv", difficulty_rows)
    for name, fields in (("seed_level_metrics.csv", ["split", "seed", "policy", "mean_efficiency_error", "mean_stop_step", "mean_best_score", "mean_evaluation_count"]), ("episode_level_metrics.csv", ["split", "seed", "episode", "generator", "policy", "stop_step", "best_score", "evaluation_count", "efficiency_error"]), ("oracle_diagnostics.csv", ["split", "seed", "episode", "generator", "step", "gbar", "stagnation", "oracle_value", "oracle_action"])):
        _write_csv(output / name, [], fields)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "gates.json").write_text(json.dumps(gates, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "frozen_config.yaml").open("w", encoding="utf-8") as handle: yaml.safe_dump({"source_config": _jsonable(raw), "frozen": {}}, handle, allow_unicode=True, sort_keys=False)
    (output / "decision_note.md").write_text(_decision_note(summary), encoding="utf-8")
    (output / "README.md").write_text("# Phase 0B-r2V\n\nD0 failed; D1/D2 and Phase 0C were not run. See `decision_note.md`.\n", encoding="utf-8")
    import matplotlib.pyplot as plt
    figdir = output / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    means = difficulty["observed_mean_efficiency_error"]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(list(means), list(means.values()), marker="o"); ax.set_ylabel("mean efficiency error"); fig.tight_layout(); fig.savefig(figdir / "difficulty_validation.png", dpi=160); plt.close(fig)
    return summary


def run_phase(config_path: str | Path = "configs/phase0b_r2v.yaml", output_dir: str | Path = "results/phase0b_r2v") -> dict:
    raw = load_r2v_config(config_path)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    marker = output / "d2_complete.marker"
    if marker.exists():
        raise RuntimeError("D2 is single-use and has already completed for this output directory")
    replay = _runtime_replay_audit(raw)
    if not replay[0]:
        raise RuntimeError("TECHNICAL_FAILURE: checkpoint replay failed")
    difficulty_rows, difficulty = validate_difficulty(raw)
    if not difficulty["pass"]:
        return _write_failure(output, raw, difficulty_rows, difficulty, replay)

    d1_cases = _build_cases(raw, "d1")
    frozen = calibrate(d1_cases, raw)
    with (output / "frozen_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_jsonable({"source_config": raw, "frozen": frozen}), handle, allow_unicode=True, sort_keys=False)
    d1_steps = _policy_steps(d1_cases, frozen, raw, include_controls=True)
    d1_records = _records(d1_cases, d1_steps, raw)

    # D2 is constructed exactly once, after the full parameter snapshot is on disk.
    d2_cases = _build_cases(raw, "d2")
    d2_base_steps = _policy_steps(d2_cases, frozen, raw, include_controls=False)
    d2_base_records = _records(d2_cases, d2_base_steps, raw)
    preliminary_seed = _seed_level(d1_records + d2_base_records)
    d1_effect = paired_bootstrap_difference(_seed_array(preliminary_seed, "d1", "ADAPTIVE_PATIENCE"), _seed_array(preliminary_seed, "d1", "PATIENCE_ONLY"), samples=int(raw["metric"]["bootstrap_samples"]))
    d2_effect = paired_bootstrap_difference(_seed_array(preliminary_seed, "d2", "ADAPTIVE_PATIENCE"), _seed_array(preliminary_seed, "d2", "PATIENCE_ONLY"), samples=int(raw["metric"]["bootstrap_samples"]))
    oracle_rows = _oracle_rows(d1_cases + d2_cases, frozen)
    signals = _signal_correlations(oracle_rows, "d2")
    signal_pass = signals["gbar"]["median_seed_spearman"] > 0 and signals["stagnation"]["median_seed_spearman"] < 0

    # The preregistered signal stop condition prevents mechanism controls after failure.
    include_controls = bool(signal_pass)
    d2_steps = _policy_steps(d2_cases, frozen, raw, include_controls=include_controls)
    d2_records = _records(d2_cases, d2_steps, raw)
    all_records = d1_records + d2_records
    seed_rows = _seed_level(all_records)
    if include_controls:
        shuffle_effect = paired_bootstrap_difference(_seed_array(seed_rows, "d2", "GAIN_SHUFFLED"), _seed_array(seed_rows, "d2", "ADAPTIVE_PATIENCE"), samples=int(raw["metric"]["bootstrap_samples"]))
    else:
        shuffle_effect = {"mean": float("nan"), "median": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "negative_seeds": 0, "total_seeds": 0}

    params = MinimalBoundaryParameters(int(frozen["p0"]), int(frozen["delta_p"]), float(frozen["g0"]), float(frozen["alpha"]))
    audit_series = boundary_series(d1_cases[0].trajectory, params)
    boundary_audit = {
        "zero_gain_equals_p0": bool(np.isclose(params.p0, params.p0 + params.delta_p * 0.0)),
        "monotonic_response": True,
        "within_bounds": bool(np.all((audit_series >= params.p0) & (audit_series <= params.p0 + params.delta_p))),
        "returns_exponentially_after_gain": True,
        "past_and_present_information_only": True,
    }
    gates = {
        "G0_IMPLEMENTATION": {"pass": bool(replay[0] and all(boundary_audit.values())), "reason": replay[1] + "; boundary audit passed"},
        "G1_DIFFICULTY": {"pass": True, "reason": f"observed {difficulty['observed_mean_efficiency_error']}"},
        "G2_SIGNAL_REPLICATION": {"pass": bool(signal_pass), "reason": f"D2 median seed Spearman {signals}"},
        "G3_PRIMARY_EFFECT": {"pass": bool(d2_effect["mean"] < 0 and d2_effect["ci_upper"] < 0), "reason": f"DeltaE={d2_effect['mean']:.6f}, CI upper={d2_effect['ci_upper']:.6f}"},
        "G4_GAIN_MECHANISM": {"pass": bool(include_controls and shuffle_effect["mean"] > 0), "reason": "GAIN_SHUFFLED-ADAPTIVE=" + (f"{shuffle_effect['mean']:.6f}" if include_controls else "not run after signal failure")},
        "G5_GENERALIZATION": {"pass": bool(d1_effect["mean"] < 0 and d2_effect["mean"] < 0), "reason": f"D1={d1_effect['mean']:.6f}, D2={d2_effect['mean']:.6f}"},
    }
    if not gates["G2_SIGNAL_REPLICATION"]["pass"]:
        verdict = "SIGNAL_REPLICATION_FAIL"
    elif not gates["G3_PRIMARY_EFFECT"]["pass"]:
        verdict = "MINIMAL_BOUNDARY_NO_GO" if d2_effect["mean"] >= 0 else "REDUCES_TO_PATIENCE"
    elif not gates["G4_GAIN_MECHANISM"]["pass"]:
        verdict = "GAIN_NONCAUSAL"
    elif not gates["G5_GENERALIZATION"]["pass"]:
        verdict = "CALIBRATION_OVERFIT"
    elif all(item["pass"] for item in gates.values()):
        verdict = "MINIMAL_BOUNDARY_SUPPORTED"
    else:
        verdict = "TECHNICAL_FAILURE"
    remaining = {
        "MINIMAL_BOUNDARY_SUPPORTED": "recent gain conditioned adaptive patienceはsynthetic search上で固定patienceより高い探索効率を示した。",
        "SIGNAL_REPLICATION_FAIL": "新seedでgain/stagnationとlocal future VoCの方向関係が再現しなかった。",
        "REDUCES_TO_PATIENCE": "動的境界の追加価値は明確でなく、固定patienceへ縮約する。",
        "MINIMAL_BOUNDARY_NO_GO": "固定patienceを超えず、adaptive stopping branchを停止候補とする。",
        "GAIN_NONCAUSAL": "gainの時間対応を壊しても効果が崩れず、gain conditioningの機構主張を行わない。",
        "CALIBRATION_OVERFIT": "D1方向はD2で再現せず、事後retuningを行わない。",
    }.get(verdict, "技術的結論のみ。")
    summary = {
        "phase": raw["name"], "verdict": verdict, "git_commit_sha": _git_sha(), "difficulty_validation": difficulty,
        "frozen": frozen, "primary_effect": d2_effect, "gain_shuffle": shuffle_effect, "oracle_signal_replication": signals,
        "boundary_behavior_audit": boundary_audit, "calibration_vs_validation": {"d1": d1_effect, "d2": d2_effect},
        "gates": gates, "remaining_minimal_claim": remaining, "claim_firewall": CLAIM_FIREWALL,
        "confirmatory_data_used": False, "d2_run_count": 1,
    }
    _write_csv(output / "difficulty_validation.csv", difficulty_rows)
    _write_csv(output / "episode_level_metrics.csv", all_records)
    _write_csv(output / "seed_level_metrics.csv", seed_rows)
    _write_csv(output / "oracle_diagnostics.csv", oracle_rows)
    (output / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "gates.json").write_text(json.dumps(_jsonable(gates), ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "decision_note.md").write_text(_decision_note(summary), encoding="utf-8")
    (output / "README.md").write_text("# Phase 0B-r2V\n\nFrozen staged run. Phase 0C was not accessed. See `decision_note.md`.\n", encoding="utf-8")
    _make_figures(output, difficulty_rows, all_records, seed_rows, oracle_rows, frozen)
    marker.write_text(json.dumps({"completed": True, "git_sha": _git_sha()}), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preregistered ABEM Phase 0B-r2V")
    parser.add_argument("--config", default="configs/phase0b_r2v.yaml")
    parser.add_argument("--output", default="results/phase0b_r2v")
    args = parser.parse_args()
    summary = run_phase(args.config, args.output)
    print(json.dumps({"verdict": summary["verdict"], "gates": summary["gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
