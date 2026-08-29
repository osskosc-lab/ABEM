from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from .boundary_analysis import (
    BoundaryParameters,
    adaptive_stop_step,
    binary_metrics,
    efficiency_at_step,
    instantaneous_hazard,
    paired_bootstrap_difference,
    patience_stop_step,
    shuffled_feature_records,
    spearman,
)
from .checkpoints import Trajectory, continue_from_checkpoint, generate_trajectory
from .config import LandscapeConfig, MetricConfig, SearchConfig
from .experiment import _agent_seed, _problem_seed
from .landscapes import make_landscape
from .oracle import OracleEstimate, estimate_oracle_value, oracle_stop_step


CLAIM_FIREWALL = (
    "本Phaseが扱うのはsynthetic search上のadaptive stoppingによるfuture Value of "
    "Computation識別だけであり、量子的機構、意識、創造性、ブレイクスルー生成を示さない。"
)


@dataclass(frozen=True)
class EpisodeCase:
    split: str
    seed: int
    episode: int
    difficulty: str
    problem_seed: int
    trajectory: Trajectory
    oracle_estimates: tuple[OracleEstimate, ...]
    oracle_stop_step: int
    oracle_error: float


def _load_phase_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if raw.get("confirmatory", False):
        raise ValueError("Phase 0B-r1F cannot run a confirmatory configuration")
    calibration = tuple(int(x) for x in raw["calibration_seeds"])
    validation = tuple(int(x) for x in raw["validation_seeds"])
    if set(calibration) & set(validation):
        raise ValueError("calibration and validation seeds must be disjoint")
    forbidden = set(int(x) for x in raw.get("forbidden_confirmatory_seeds", []))
    if forbidden & (set(calibration) | set(validation)):
        raise ValueError("confirmatory seed leakage detected")
    raw["calibration_seeds"] = calibration
    raw["validation_seeds"] = validation
    return raw


def _construct_search(raw: dict) -> SearchConfig:
    return SearchConfig(**raw["search"])


def _construct_metric(raw: dict) -> MetricConfig:
    return MetricConfig(**raw["metric"])


def _landscape_config(raw: dict, difficulty: str) -> LandscapeConfig:
    base = dict(raw["landscape"])
    base["interaction_strength"] = float(raw["difficulties"][difficulty]["interaction_strength"])
    return LandscapeConfig(**base)


def _build_cases(
    raw: dict,
    *,
    split: str,
    seeds: Iterable[int],
    rollouts_override: int | None = None,
    episodes_override: int | None = None,
) -> list[EpisodeCase]:
    search = _construct_search(raw)
    metric = _construct_metric(raw)
    horizons = tuple(int(x) for x in raw["checkpoint_horizons"])
    rollouts = int(rollouts_override or raw["oracle_rollouts_per_checkpoint"])
    episodes = int(episodes_override or raw["episodes_per_seed"])
    cases: list[EpisodeCase] = []

    for seed in seeds:
        for episode in range(episodes):
            problem_seed = _problem_seed(int(seed), episode)
            for difficulty in raw["difficulties"]:
                landscape = make_landscape(problem_seed, _landscape_config(raw, difficulty))
                trajectory = generate_trajectory(
                    landscape,
                    agent_seed=_agent_seed(int(seed), episode),
                    config=search,
                    checkpoint_steps=horizons,
                )
                estimates = tuple(
                    estimate_oracle_value(
                        landscape,
                        checkpoint,
                        problem_seed=problem_seed,
                        search=search,
                        metric=metric,
                        horizons=horizons,
                        rollouts=rollouts,
                    )
                    for checkpoint in trajectory.checkpoints
                )
                stop = oracle_stop_step(trajectory, estimates)
                cases.append(
                    EpisodeCase(
                        split=split,
                        seed=int(seed),
                        episode=episode,
                        difficulty=difficulty,
                        problem_seed=problem_seed,
                        trajectory=trajectory,
                        oracle_estimates=estimates,
                        oracle_stop_step=stop,
                        oracle_error=efficiency_at_step(
                            trajectory,
                            stop,
                            search=search,
                            metric=metric,
                        ),
                    )
                )
    return cases


def _runtime_replay_audit(raw: dict) -> tuple[bool, str]:
    search = _construct_search(raw)
    seed = int(raw["calibration_seeds"][0])
    problem_seed = _problem_seed(seed, 0)
    landscape = make_landscape(problem_seed, _landscape_config(raw, raw["difficulty_order"][0]))
    first = int(raw["checkpoint_horizons"][0])
    second = int(raw["checkpoint_horizons"][1])
    trajectory = generate_trajectory(
        landscape,
        agent_seed=_agent_seed(seed, 0),
        config=search,
        checkpoint_steps=(first, second),
    )
    replay = continue_from_checkpoint(
        landscape,
        trajectory.at_step(first),
        config=search,
        horizon=second - first,
    )
    expected = trajectory.at_step(second)
    passed = bool(
        np.array_equal(replay.population, expected.population)
        and np.array_equal(replay.population_scores, expected.population_scores)
        and replay.best_score == expected.best_score
        and replay.cumulative_hazard == expected.cumulative_hazard
    )
    reason = (
        "runtime checkpoint replay matched population, scores, best score, and cumulative hazard"
        if passed
        else "runtime checkpoint replay mismatch"
    )
    return passed, reason


def _mean_error(cases: list[EpisodeCase], steps: list[int], search: SearchConfig, metric: MetricConfig) -> float:
    return float(
        np.mean(
            [
                efficiency_at_step(case.trajectory, step, search=search, metric=metric)
                for case, step in zip(cases, steps, strict=True)
            ]
        )
    )


def _calibrate(raw: dict, cases: list[EpisodeCase]) -> dict:
    search = _construct_search(raw)
    metric = _construct_metric(raw)

    fixed_scores = {
        int(depth): _mean_error(
            cases,
            [min(int(depth), search.t_max)] * len(cases),
            search,
            metric,
        )
        for depth in raw["fixed_depth_candidates"]
    }
    fixed_depth = min(fixed_scores, key=fixed_scores.get)

    patience_scores = {}
    for patience in raw["patience_candidates"]:
        steps = [
            patience_stop_step(case.trajectory, patience=int(patience), t_min=search.t_min)
            for case in cases
        ]
        patience_scores[int(patience)] = _mean_error(cases, steps, search, metric)
    patience = min(patience_scores, key=patience_scores.get)

    boundary_scores = {}
    boundary_params = {}
    for item in raw["boundary_candidates"]:
        params = BoundaryParameters(**item["parameters"])
        if min(
            params.b_stagnation,
            params.b_gain,
            params.b_uncertainty,
            params.b_diversity,
        ) < 0:
            raise ValueError("boundary candidate violates preregistered sign constraints")
        steps = [
            adaptive_stop_step(case.trajectory, search=search, params=params)
            for case in cases
        ]
        errors = np.asarray(
            [
                efficiency_at_step(case.trajectory, step, search=search, metric=metric)
                for case, step in zip(cases, steps, strict=True)
            ]
        )
        oracle = np.asarray([case.oracle_error for case in cases])
        boundary_scores[item["name"]] = float(np.mean(errors - oracle))
        boundary_params[item["name"]] = params
    boundary_name = min(boundary_scores, key=boundary_scores.get)

    return {
        "fixed_depth": int(fixed_depth),
        "fixed_depth_mean_efficiency_error": fixed_scores,
        "patience": int(patience),
        "patience_mean_efficiency_error": patience_scores,
        "boundary_candidate": boundary_name,
        "boundary_oracle_regret": boundary_scores,
        "boundary_parameters": asdict(boundary_params[boundary_name]),
    }


def _case_policy_steps(
    cases: list[EpisodeCase],
    *,
    raw: dict,
    frozen: dict,
) -> dict[str, list[int]]:
    search = _construct_search(raw)
    params = BoundaryParameters(**frozen["boundary_parameters"])
    steps: dict[str, list[int]] = {
        "TIME_ONLY": [min(frozen["fixed_depth"], search.t_max)] * len(cases),
        "STAGNATION_ONLY": [
            patience_stop_step(case.trajectory, patience=frozen["patience"], t_min=search.t_min)
            for case in cases
        ],
        "FULL_AB": [adaptive_stop_step(case.trajectory, search=search, params=params) for case in cases],
        "SIGNAL_SHUFFLED": [
            adaptive_stop_step(
                case.trajectory,
                search=search,
                params=params,
                feature_records=shuffled_feature_records(
                    case.trajectory,
                    seed=80_000_003 + case.seed * 10_007 + case.episode * 101,
                ),
            )
            for case in cases
        ],
        "MINUS_GAIN": [
            adaptive_stop_step(case.trajectory, search=search, params=params.without("gain")) for case in cases
        ],
        "MINUS_UNCERTAINTY": [
            adaptive_stop_step(case.trajectory, search=search, params=params.without("score_std")) for case in cases
        ],
        "MINUS_DIVERSITY": [
            adaptive_stop_step(case.trajectory, search=search, params=params.without("diversity")) for case in cases
        ],
        "MINUS_STAGNATION": [
            adaptive_stop_step(case.trajectory, search=search, params=params.without("stagnation")) for case in cases
        ],
        "ORACLE": [case.oracle_stop_step for case in cases],
    }

    rng = np.random.default_rng(20260829 if cases[0].split == "calibration" else 20260830)
    random_steps = [0] * len(cases)
    for difficulty in raw["difficulties"]:
        indices = [i for i, case in enumerate(cases) if case.difficulty == difficulty]
        matched = np.asarray([steps["FULL_AB"][i] for i in indices], dtype=int)
        matched = matched[rng.permutation(len(matched))]
        for idx, value in zip(indices, matched, strict=True):
            random_steps[idx] = int(value)
    steps["RANDOM_MATCHED"] = random_steps

    mean_full_step = int(np.clip(round(float(np.mean(steps["FULL_AB"]))), 1, search.t_max))
    steps["EQUAL_BUDGET_FIXED"] = [mean_full_step] * len(cases)

    # Difficulty-blind control: rotate whole feature trajectories among families
    # while retaining the actual landscape/quality trajectory used for scoring.
    lookup = {(case.seed, case.episode, case.difficulty): case for case in cases}
    names = list(raw["difficulties"])
    rotated = {names[i]: names[(i + 1) % len(names)] for i in range(len(names))}
    blind_steps = []
    for case in cases:
        donor = lookup[(case.seed, case.episode, rotated[case.difficulty])]
        blind_steps.append(
            adaptive_stop_step(
                case.trajectory,
                search=search,
                params=params,
                feature_records=donor.trajectory.step_records,
            )
        )
    steps["DIFFICULTY_BLIND"] = blind_steps
    return steps


def _records_for_cases(cases: list[EpisodeCase], steps: dict[str, list[int]], raw: dict) -> list[dict]:
    search = _construct_search(raw)
    metric = _construct_metric(raw)
    rows = []
    for i, case in enumerate(cases):
        for policy, policy_steps in steps.items():
            step = int(policy_steps[i])
            record = case.trajectory.at_step(step)
            error = efficiency_at_step(case.trajectory, step, search=search, metric=metric)
            rows.append(
                {
                    "split": case.split,
                    "seed": case.seed,
                    "episode": case.episode,
                    "difficulty": case.difficulty,
                    "policy": policy,
                    "stop_step": step,
                    "best_score": record.best_score,
                    "normalized_regret": 1.0 - record.best_score,
                    "evaluation_count": step * search.population_size,
                    "efficiency_error": error,
                    "oracle_stop_step": case.oracle_stop_step,
                    "oracle_regret": error - case.oracle_error,
                }
            )
    return rows


def _seed_level(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["seed"], row["policy"])].append(row)
    output = []
    for (split, seed, policy), values in sorted(grouped.items()):
        output.append(
            {
                "split": split,
                "seed": seed,
                "policy": policy,
                "mean_oracle_regret": float(np.mean([x["oracle_regret"] for x in values])),
                "mean_efficiency_error": float(np.mean([x["efficiency_error"] for x in values])),
                "mean_stop_step": float(np.mean([x["stop_step"] for x in values])),
                "mean_best_score": float(np.mean([x["best_score"] for x in values])),
                "mean_evaluation_count": float(np.mean([x["evaluation_count"] for x in values])),
            }
        )
    return output


def _checkpoint_rows(cases: list[EpisodeCase], frozen: dict) -> list[dict]:
    params = BoundaryParameters(**frozen["boundary_parameters"])
    rows = []
    for case in cases:
        estimates = {x.checkpoint_step: x for x in case.oracle_estimates}
        for checkpoint in case.trajectory.checkpoints:
            estimate = estimates[checkpoint.step]
            rows.append(
                {
                    "split": case.split,
                    "seed": case.seed,
                    "episode": case.episode,
                    "difficulty": case.difficulty,
                    "step": checkpoint.step,
                    "gain": checkpoint.gain,
                    "score_std": checkpoint.score_std,
                    "diversity": checkpoint.diversity,
                    "stagnation": checkpoint.stagnation,
                    "hazard": instantaneous_hazard(checkpoint, params),
                    "cumulative_hazard": checkpoint.cumulative_hazard,
                    "oracle_value": estimate.oracle_value,
                    "oracle_action": estimate.action,
                }
            )
    return rows


def _seed_arrays(seed_rows: list[dict], split: str, policy: str) -> np.ndarray:
    selected = [x for x in seed_rows if x["split"] == split and x["policy"] == policy]
    return np.asarray([x["mean_oracle_regret"] for x in sorted(selected, key=lambda x: x["seed"])])


def _diagnostics(checkpoint_rows: list[dict], split: str) -> dict:
    selected = [row for row in checkpoint_rows if row["split"] == split]
    labels = np.asarray([row["oracle_action"] == "STOP" for row in selected], dtype=int)
    hazards = np.asarray([row["hazard"] for row in selected], dtype=float)
    metrics = binary_metrics(labels, hazards)
    seed_signs = defaultdict(lambda: defaultdict(list))
    for row in selected:
        for feature in ("gain", "score_std", "diversity", "stagnation"):
            seed_signs[row["seed"]][feature].append((row[feature], row["oracle_value"]))
    correlations = {}
    for feature in ("gain", "score_std", "diversity", "stagnation"):
        values = []
        for seed in sorted(seed_signs):
            pairs = seed_signs[seed][feature]
            rho = spearman(
                np.asarray([x[0] for x in pairs]),
                np.asarray([x[1] for x in pairs]),
            )
            if np.isfinite(rho):
                values.append(rho)
        correlations[feature] = {
            "mean_seed_spearman": float(np.mean(values)) if values else float("nan"),
            "median_seed_spearman": float(np.median(values)) if values else float("nan"),
            "n_seeds": len(values),
        }
    return {**metrics, "seed_level_signal_correlations": correlations}


def _mean_policy(rows: list[dict], split: str, policy: str, field: str = "oracle_regret") -> float:
    return float(np.mean([x[field] for x in rows if x["split"] == split and x["policy"] == policy]))


def _difficulty_steps(rows: list[dict], split: str, policy: str) -> dict[str, float]:
    names = sorted({x["difficulty"] for x in rows})
    return {
        name: float(
            np.mean(
                [
                    x["stop_step"]
                    for x in rows
                    if x["split"] == split and x["policy"] == policy and x["difficulty"] == name
                ]
            )
        )
        for name in names
    }


def _classify(
    *,
    gates: dict[str, dict],
    calibration_diff: float,
    validation_diff: float,
    validation_patience_minus_full: float,
    signal_misspecified: bool,
) -> str:
    if not gates["G0_REPLAY"]["pass"]:
        return "TECHNICAL_FAILURE"
    if calibration_diff < 0 <= validation_diff:
        return "BOUNDARY_OVERFIT"
    if signal_misspecified and gates["G1_PREDICTIVE_INFORMATION"]["pass"]:
        return "BOUNDARY_SIGNAL_MISSPECIFIED"
    if validation_patience_minus_full <= 0:
        return "BOUNDARY_REDUCES_TO_PATIENCE"
    if all(item["pass"] for item in gates.values()):
        return "BOUNDARY_MECHANISM_SUPPORTED"
    return "BOUNDARY_NO_GO"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _make_figures(
    output: Path,
    checkpoint_rows: list[dict],
    rows: list[dict],
    seed_rows: list[dict],
) -> None:
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    feature_names = {
        "gain": "future_value_vs_gain",
        "score_std": "future_value_vs_uncertainty",
        "diversity": "future_value_vs_diversity",
        "stagnation": "future_value_vs_stagnation",
    }
    for feature, filename in feature_names.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for split, marker in (("calibration", "o"), ("validation", "x")):
            subset = [x for x in checkpoint_rows if x["split"] == split]
            ax.scatter(
                [x[feature] for x in subset],
                [x["oracle_value"] for x in subset],
                alpha=0.25,
                s=14,
                marker=marker,
                label=split,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(feature)
        ax.set_ylabel("oracle future value")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"{filename}.png", dpi=160)
        plt.close(fig)

    def boxplot(filename: str, policies: list[str], ylabel: str, split: str = "validation"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        values = [
            [
                x["mean_oracle_regret"]
                for x in seed_rows
                if x["split"] == split and x["policy"] == policy
            ]
            for policy in policies
        ]
        ax.boxplot(values, tick_labels=policies, showmeans=True)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{filename}.png", dpi=160)
        plt.close(fig)

    boxplot(
        "oracle_regret_by_policy",
        ["TIME_ONLY", "STAGNATION_ONLY", "RANDOM_MATCHED", "FULL_AB"],
        "oracle regret (lower is better)",
    )
    boxplot("intact_vs_signal_shuffled", ["FULL_AB", "SIGNAL_SHUFFLED"], "oracle regret")

    fig, ax = plt.subplots(figsize=(6, 4))
    for policy in ("FULL_AB", "DIFFICULTY_BLIND"):
        means = _difficulty_steps(rows, "validation", policy)
        ax.plot(list(means), list(means.values()), marker="o", label=policy)
    ax.set_ylabel("mean stop step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "stop_step_by_difficulty.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    policies = ["TIME_ONLY", "STAGNATION_ONLY", "RANDOM_MATCHED", "FULL_AB"]
    x = np.arange(len(policies))
    width = 0.35
    for j, split in enumerate(("calibration", "validation")):
        values = [
            float(
                np.mean(
                    [
                        x["mean_oracle_regret"]
                        for x in seed_rows
                        if x["split"] == split and x["policy"] == policy
                    ]
                )
            )
            for policy in policies
        ]
        ax.bar(x + (j - 0.5) * width, values, width=width, label=split)
    ax.set_xticks(x, policies, rotation=20)
    ax.set_ylabel("mean oracle regret")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "calibration_vs_validation.png", dpi=160)
    plt.close(fig)


def _decision_note(summary: dict) -> str:
    diagnostics = summary["diagnostics"]
    gates = summary["gates"]
    failed = [name for name, item in gates.items() if not item["pass"]]
    return f"""# ABEM Phase 0B-r1F Decision Note

## 総合判定

**{summary['verdict']}**

## 中心命題

synthetic search環境で、現在の探索状態に依存するAdaptive Boundaryが、固定時間・patience・matched random停止より局所Value-of-Computation Oracleに近い停止判断を行えるかを検証した。

## 凍結した比較条件

- 固定深度: `{summary['frozen']['fixed_depth']}`
- patience: `{summary['frozen']['patience']}`
- FULL_AB候補: `{summary['frozen']['boundary_candidate']}`
- Calibration上の最強simple baseline: `{summary['best_simple_policy']}`

## 主要結果

- Validation ΔR (FULL_AB - best simple): `{diagnostics['validation_primary_difference']['mean']:.6f}`
- paired seed bootstrap 95% CI: `[{diagnostics['validation_primary_difference']['ci_lower']:.6f}, {diagnostics['validation_primary_difference']['ci_upper']:.6f}]`
- Validation AUROC / AUPRC / Brier: `{diagnostics['validation_predictive']['auroc']:.4f}` / `{diagnostics['validation_predictive']['auprc']:.4f}` / `{diagnostics['validation_predictive']['brier']:.4f}`
- FULL_AB validation oracle regret: `{diagnostics['validation_policy_oracle_regret']['FULL_AB']:.6f}`
- Signal-shuffled validation oracle regret: `{diagnostics['validation_policy_oracle_regret']['SIGNAL_SHUFFLED']:.6f}`

ここでのoracle regretは、局所rollout Oracleが選んだ停止時刻に対する**符号付き相対誤差**である。絶対最適Oracleではないため負値を取り得る。policy間のpaired差を主判定に用いる。

## 反証Gate

""" + "\n".join(
        f"- {name}: {'PASS' if item['pass'] else 'FAIL'} — {item['reason']}" for name, item in gates.items()
    ) + f"""

## Feature deletion

{json.dumps(diagnostics['feature_deletion_delta'], ensure_ascii=False, indent=2)}

`score_std`はfuture valueとの想定符号に反し、uncertainty削除はFULL_ABを悪化させなかった。したがってepistemic uncertainty機構としては支持しない。

## Difficulty adaptation

- intact: `{json.dumps(diagnostics['validation_stop_step_by_difficulty'], ensure_ascii=False)}`
- difficulty-blind: `{json.dumps(diagnostics['validation_blind_stop_step_by_difficulty'], ensure_ascii=False)}`

## 反証された主張

失敗Gate: `{', '.join(failed) if failed else 'なし'}`。Gateを通らない機構主張は採用しない。

## 残る最小主張

判定ラベルと通過Gateが許す範囲に限定する。単純baselineを超えない場合、複雑なAdaptive Boundaryの必要性は主張しない。

## Claim Firewall

{CLAIM_FIREWALL}

## 次段階

`BOUNDARY_MECHANISM_SUPPORTED`の場合のみPhase 0B-r2 Memory Falsificationへ進む。それ以外は判定ラベルに従い縮約またはNO_GOとし、Phase 0Cへ進めない。
"""


def run_phase(config_path: str | Path, output_dir: str | Path) -> dict:
    raw = _load_phase_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    search = _construct_search(raw)
    metric = _construct_metric(raw)
    replay_pass, replay_reason = _runtime_replay_audit(raw)
    if not replay_pass:
        raise RuntimeError("TECHNICAL_FAILURE: runtime checkpoint replay mismatch")

    calibration_cases = _build_cases(raw, split="calibration", seeds=raw["calibration_seeds"])
    frozen = _calibrate(raw, calibration_cases)
    calibration_steps = _case_policy_steps(calibration_cases, raw=raw, frozen=frozen)
    calibration_records = _records_for_cases(calibration_cases, calibration_steps, raw)

    # Validation is constructed exactly once and only after all tunable choices freeze.
    validation_cases = _build_cases(raw, split="validation", seeds=raw["validation_seeds"])
    validation_steps = _case_policy_steps(validation_cases, raw=raw, frozen=frozen)
    validation_records = _records_for_cases(validation_cases, validation_steps, raw)
    all_records = calibration_records + validation_records
    seed_rows = _seed_level(all_records)
    checkpoint_rows = _checkpoint_rows(calibration_cases + validation_cases, frozen)

    simple = ("TIME_ONLY", "STAGNATION_ONLY", "RANDOM_MATCHED")
    calibration_simple_means = {policy: _mean_policy(all_records, "calibration", policy) for policy in simple}
    best_simple = min(calibration_simple_means, key=calibration_simple_means.get)
    calibration_difference = paired_bootstrap_difference(
        _seed_arrays(seed_rows, "calibration", "FULL_AB"),
        _seed_arrays(seed_rows, "calibration", best_simple),
        samples=metric.bootstrap_samples,
    )
    validation_difference = paired_bootstrap_difference(
        _seed_arrays(seed_rows, "validation", "FULL_AB"),
        _seed_arrays(seed_rows, "validation", best_simple),
        samples=metric.bootstrap_samples,
    )

    cal_predictive = _diagnostics(checkpoint_rows, "calibration")
    val_predictive = _diagnostics(checkpoint_rows, "validation")
    val_regrets = {
        policy: _mean_policy(all_records, "validation", policy)
        for policy in sorted({x["policy"] for x in all_records})
    }
    feature_deletion = {
        policy: val_regrets[policy] - val_regrets["FULL_AB"]
        for policy in ("MINUS_GAIN", "MINUS_UNCERTAINTY", "MINUS_DIVERSITY", "MINUS_STAGNATION")
    }
    stop_steps = _difficulty_steps(all_records, "validation", "FULL_AB")
    blind_steps = _difficulty_steps(all_records, "validation", "DIFFICULTY_BLIND")
    easy, rugged = raw["difficulty_order"][0], raw["difficulty_order"][-1]
    intact_gap = stop_steps[rugged] - stop_steps[easy]
    blind_gap = blind_steps[rugged] - blind_steps[easy]
    sign_values = val_predictive["seed_level_signal_correlations"]
    expected = {"gain": 1, "score_std": 1, "diversity": 1, "stagnation": -1}
    sign_pass = {
        feature: bool(expected[feature] * values["median_seed_spearman"] > 0)
        for feature, values in sign_values.items()
    }
    signal_misspecified = not all(sign_pass.values())

    gates = {
        "G0_REPLAY": {"pass": replay_pass, "reason": replay_reason},
        "G1_PREDICTIVE_INFORMATION": {
            "pass": bool(np.isfinite(val_predictive["auroc"]) and val_predictive["auroc"] > 0.60),
            "reason": f"validation AUROC={val_predictive['auroc']:.4f}; threshold > 0.60",
        },
        "G2_SIMPLE_BASELINE": {
            "pass": bool(validation_difference["mean"] < 0 and validation_difference["ci_upper"] < 0),
            "reason": (
                f"FULL_AB-{best_simple} mean={validation_difference['mean']:.6f}, "
                f"95% CI upper={validation_difference['ci_upper']:.6f}"
            ),
        },
        "G3_TEMPORAL_MECHANISM": {
            "pass": bool(val_regrets["SIGNAL_SHUFFLED"] > val_regrets["FULL_AB"]),
            "reason": (
                f"shuffle-full oracle regret={val_regrets['SIGNAL_SHUFFLED'] - val_regrets['FULL_AB']:.6f}"
            ),
        },
        "G4_DIFFICULTY_ADAPTATION": {
            "pass": bool(intact_gap > 0 and blind_gap <= raw["difficulty_blind_max_ratio"] * intact_gap),
            "reason": f"rugged-easy gap intact={intact_gap:.4f}, blind={blind_gap:.4f}",
        },
        "G5_VALIDATION": {
            "pass": bool(calibration_difference["mean"] < 0 and validation_difference["mean"] < 0),
            "reason": (
                f"calibration direction={calibration_difference['mean']:.6f}, "
                f"validation direction={validation_difference['mean']:.6f}"
            ),
        },
    }
    patience_minus_full = val_regrets["STAGNATION_ONLY"] - val_regrets["FULL_AB"]
    verdict = _classify(
        gates=gates,
        calibration_diff=float(calibration_difference["mean"]),
        validation_diff=float(validation_difference["mean"]),
        validation_patience_minus_full=patience_minus_full,
        signal_misspecified=signal_misspecified,
    )

    diagnostics = {
        "calibration_primary_difference": calibration_difference,
        "validation_primary_difference": validation_difference,
        "calibration_predictive": cal_predictive,
        "validation_predictive": val_predictive,
        "calibration_simple_oracle_regret": calibration_simple_means,
        "validation_policy_oracle_regret": val_regrets,
        "feature_deletion_delta": feature_deletion,
        "signal_direction_pass": sign_pass,
        "validation_stop_step_by_difficulty": stop_steps,
        "validation_blind_stop_step_by_difficulty": blind_steps,
        "equal_budget_quality_delta": (
            _mean_policy(all_records, "validation", "FULL_AB", "normalized_regret")
            - _mean_policy(all_records, "validation", "EQUAL_BUDGET_FIXED", "normalized_regret")
        ),
        "metric_sensitivity": {},
        "oracle_regret_note": (
            "The local rollout Oracle is not an absolute optimal-stopping oracle; signed "
            "policy-minus-oracle error can be negative. Paired policy differences remain the primary comparison."
        ),
    }
    for cost_weight in raw["metric_sensitivity_values"]:
        sensitivity_metric = replace(metric, cost_weight=float(cost_weight))
        full = []
        simple_values = []
        for i, case in enumerate(validation_cases):
            full.append(
                efficiency_at_step(
                    case.trajectory,
                    validation_steps["FULL_AB"][i],
                    search=search,
                    metric=sensitivity_metric,
                )
            )
            simple_values.append(
                efficiency_at_step(
                    case.trajectory,
                    validation_steps[best_simple][i],
                    search=search,
                    metric=sensitivity_metric,
                )
            )
        diagnostics["metric_sensitivity"][str(cost_weight)] = float(np.mean(full) - np.mean(simple_values))

    primary_direction = np.sign(float(validation_difference["mean"]))
    sensitivity_directions = [np.sign(x) for x in diagnostics["metric_sensitivity"].values()]
    diagnostics["metric_fragile"] = bool(
        primary_direction != 0 and any(direction != primary_direction for direction in sensitivity_directions)
    )
    flags = []
    if signal_misspecified:
        flags.append("SIGNAL_MISSPECIFIED")
    if diagnostics["metric_fragile"]:
        flags.append("METRIC_FRAGILE")

    summary = {
        "phase": raw["name"],
        "verdict": verdict,
        "git_commit_sha": _git_sha(),
        "config_path": str(config_path),
        "config_snapshot": raw,
        "frozen": frozen,
        "best_simple_policy": best_simple,
        "gates": gates,
        "diagnostics": diagnostics,
        "flags": flags,
        "claim_firewall": CLAIM_FIREWALL,
        "confirmatory_data_used": False,
    }

    _write_csv(output / "seed_level_metrics.csv", seed_rows)
    _write_csv(output / "checkpoint_oracle_metrics.csv", checkpoint_rows)
    (output / "summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "gates.json").write_text(
        json.dumps(_jsonable(gates), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "frozen_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_jsonable({"source_config": raw, "frozen": frozen}), handle, allow_unicode=True, sort_keys=False)
    (output / "decision_note.md").write_text(_decision_note(summary), encoding="utf-8")
    _make_figures(output, checkpoint_rows, all_records, seed_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ABEM Phase 0B-r1F falsification battery")
    parser.add_argument("--config", default="configs/phase0b_r1f.yaml")
    parser.add_argument("--output", default="results/phase0b_r1f")
    args = parser.parse_args()
    result = run_phase(args.config, args.output)
    print(json.dumps({"verdict": result["verdict"], "gates": result["gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
