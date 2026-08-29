from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from .boundary_analysis import paired_bootstrap_difference
from .checkpoints import generate_trajectory
from .config import LandscapeConfig, MetricConfig, SearchConfig
from .experiment import _agent_seed, _problem_seed
from .landscapes import make_landscape
from .metrics import efficiency_error


GATE_STATUSES = frozenset({"PASS", "FAIL", "NOT_EVALUATED"})
CLAIM_FIREWALL = (
    "本PhaseはABEM固定探索kernelに対するsynthetic difficulty gradientだけを評価する。"
    "Adaptive Boundary、MVOC-B、Memory、量子機構、AI意識・創造性の有効性を示さない。"
)


def gate(status: str, reason: str) -> dict[str, str]:
    if status not in GATE_STATUSES:
        raise ValueError(f"invalid gate status: {status}")
    return {"status": status, "reason": reason}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(x) for x in value]
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


def load_r2g_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if raw.get("confirmatory", False):
        raise ValueError("r2G cannot run a confirmatory configuration")
    for stage in ("screening", "replication"):
        raw[stage]["seeds"] = tuple(int(x) for x in raw[stage]["seeds"])
    screening = set(raw["screening"]["seeds"])
    replication = set(raw["replication"]["seeds"])
    forbidden = set(int(x) for x in raw["forbidden_r2v_seeds"])
    forbidden |= set(range(int(raw["forbidden_confirmatory_seed_min"]), int(raw["forbidden_confirmatory_seed_max"]) + 1))
    if screening & replication:
        raise ValueError("screening and replication seeds must be disjoint")
    if forbidden & (screening | replication):
        raise ValueError("r2V or Phase 0C seed leakage detected")
    required_search = {
        "population_size": 24,
        "elite_size": 6,
        "mutation_rate": 0.08,
        "memory_bias": 0.0,
        "fixed_depth": 32,
        "t_min": 4,
        "t_max": 64,
    }
    for name, expected in required_search.items():
        if raw["search"][name] != expected:
            raise ValueError(f"frozen search kernel changed: {name}")
    return raw


def level_config(raw: dict, family: str, level: str) -> LandscapeConfig:
    family_config = raw["families"][family]
    values = dict(raw["landscape"])
    values.update(family_config["fixed"])
    values.update(family_config["levels"][level])
    return LandscapeConfig(**values)


def _target_signature(target: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(target, dtype=np.int8).tobytes()).hexdigest()[:16]


def implementation_audit(raw: dict) -> tuple[bool, dict]:
    search = SearchConfig(**raw["search"])
    targets = []
    target_scores = []
    for family, family_config in raw["families"].items():
        for level in family_config["levels"]:
            landscape = make_landscape(98_765, level_config(raw, family, level))
            targets.append(landscape.target)
            target_scores.append(float(landscape.evaluate(landscape.target)[0]))
    target_matched = all(np.array_equal(targets[0], item) for item in targets[1:])
    normalized = bool(np.allclose(target_scores, 1.0, atol=1e-12))

    landscape = make_landscape(12_345, level_config(raw, "FAMILY_C_CONFLICTING_INTERACTIONS", "C4"))
    kwargs = dict(agent_seed=54_321, config=search, checkpoint_steps=(search.fixed_depth,))
    left = generate_trajectory(landscape, **kwargs)
    right = generate_trajectory(landscape, **kwargs)
    replay = all(
        (
            np.array_equal(left.at_step(search.fixed_depth).population, right.at_step(search.fixed_depth).population),
            np.array_equal(left.at_step(search.fixed_depth).population_scores, right.at_step(search.fixed_depth).population_scores),
            left.at_step(search.fixed_depth).best_score == right.at_step(search.fixed_depth).best_score,
        )
    )
    details = {
        "target_matched_across_all_levels": bool(target_matched),
        "target_normalized_score_is_one": normalized,
        "iid_replay_exact": bool(replay),
        "search_kernel_frozen": True,
        "controller_evaluated": False,
        "target_information_exposed_to_agent": False,
    }
    passed = bool(
        target_matched
        and normalized
        and replay
        and details["search_kernel_frozen"]
        and not details["controller_evaluated"]
        and not details["target_information_exposed_to_agent"]
    )
    return passed, details


def _run_stage(raw: dict, stage: str, selection: dict[str, list[str]] | None = None) -> list[dict]:
    search = SearchConfig(**raw["search"])
    metric = MetricConfig(**raw["metric"])
    threshold = float(raw["success_score_threshold"])
    rows: list[dict] = []
    families = selection or {name: list(item["levels"]) for name, item in raw["families"].items()}
    for seed in raw[stage]["seeds"]:
        for episode in range(int(raw[stage]["episodes_per_seed"])):
            problem_seed = _problem_seed(seed, episode)
            agent_seed = _agent_seed(seed, episode)
            expected_target = None
            for family, levels in families.items():
                for level in levels:
                    landscape = make_landscape(problem_seed, level_config(raw, family, level))
                    if expected_target is None:
                        expected_target = landscape.target.copy()
                    elif not np.array_equal(expected_target, landscape.target):
                        raise RuntimeError("TECHNICAL_FAILURE: target matching broke within paired case")
                    trajectory = generate_trajectory(
                        landscape,
                        agent_seed=agent_seed,
                        config=search,
                        checkpoint_steps=(search.fixed_depth,),
                    )
                    record = trajectory.at_step(search.fixed_depth)
                    regret = float(1.0 - record.best_score)
                    rows.append(
                        {
                            "stage": stage,
                            "seed": int(seed),
                            "episode": episode,
                            "family": family,
                            "level": level,
                            "problem_seed": problem_seed,
                            "agent_seed": agent_seed,
                            "target_signature": _target_signature(landscape.target),
                            "terminal_regret": regret,
                            "efficiency_error": efficiency_error(record.best_score, search.fixed_depth, search, metric),
                            "best_score": record.best_score,
                            "success_at_threshold": int(record.best_score >= threshold),
                            "reached_global_optimum": int(np.isclose(record.best_score, 1.0, atol=1e-12)),
                            "final_population_diversity": record.diversity,
                            "target_normalized_score": float(landscape.evaluate(landscape.target)[0]),
                        }
                    )
    return rows


SEED_FIELDS = [
    "stage", "seed", "family", "level", "mean_terminal_regret", "median_terminal_regret",
    "mean_efficiency_error", "mean_best_score", "success_rate_at_threshold",
    "global_optimum_success_rate", "mean_final_population_diversity", "episodes",
]


def _seed_level(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["stage"], row["seed"], row["family"], row["level"])].append(row)
    output = []
    for (stage, seed, family, level), values in sorted(grouped.items()):
        output.append(
            {
                "stage": stage,
                "seed": seed,
                "family": family,
                "level": level,
                "mean_terminal_regret": float(np.mean([x["terminal_regret"] for x in values])),
                "median_terminal_regret": float(np.median([x["terminal_regret"] for x in values])),
                "mean_efficiency_error": float(np.mean([x["efficiency_error"] for x in values])),
                "mean_best_score": float(np.mean([x["best_score"] for x in values])),
                "success_rate_at_threshold": float(np.mean([x["success_at_threshold"] for x in values])),
                "global_optimum_success_rate": float(np.mean([x["reached_global_optimum"] for x in values])),
                "mean_final_population_diversity": float(np.mean([x["final_population_diversity"] for x in values])),
                "episodes": len(values),
            }
        )
    return output


def _level_array(seed_rows: list[dict], family: str, level: str) -> np.ndarray:
    selected = sorted((x for x in seed_rows if x["family"] == family and x["level"] == level), key=lambda x: x["seed"])
    return np.asarray([x["mean_terminal_regret"] for x in selected], dtype=float)


def _candidate_stats(seed_rows: list[dict], family: str, levels: list[str], samples: int) -> dict:
    arrays = {level: _level_array(seed_rows, family, level) for level in levels}
    means = {level: float(np.mean(values)) for level, values in arrays.items()}
    medians = {level: float(np.median(values)) for level, values in arrays.items()}
    adjacent = [means[right] - means[left] for left, right in zip(levels, levels[1:])]
    effect = paired_bootstrap_difference(arrays[levels[-1]], arrays[levels[0]], samples=samples)
    direction_count = int(np.sum(arrays[levels[-1]] >= arrays[levels[0]]))
    direction_fraction = float(direction_count / len(arrays[levels[0]]))
    monotonic = bool(all(value > 0 for value in adjacent))
    passed = bool(monotonic and effect["ci_lower"] > 0 and direction_fraction >= 0.70)
    return {
        "levels": levels,
        "mean_terminal_regret": means,
        "median_terminal_regret": medians,
        "adjacent_mean_differences": adjacent,
        "monotonic_increase": monotonic,
        "hardest_minus_easiest": effect,
        "direction_consistent_seed_count": direction_count,
        "direction_consistent_seed_fraction": direction_fraction,
        "development_gate_pass": passed,
        "monotonicity_margin": float(min(adjacent)) if adjacent else float("nan"),
    }


def screen_families(seed_rows: list[dict], raw: dict) -> tuple[dict, dict | None]:
    samples = int(raw["bootstrap_samples"])
    results = {}
    passing = []
    for family, config in raw["families"].items():
        levels = list(config["levels"])
        candidates = []
        for width in range(len(levels), 2, -1):
            for start in range(0, len(levels) - width + 1):
                candidates.append(_candidate_stats(seed_rows, family, levels[start : start + width], samples))
        valid = [item for item in candidates if item["development_gate_pass"]]
        best = max(
            valid,
            key=lambda item: (
                len(item["levels"]), item["monotonicity_margin"],
                item["hardest_minus_easiest"]["ci_lower"], item["hardest_minus_easiest"]["mean"],
            ),
            default=None,
        )
        rng = np.random.default_rng(20_260_829 + int(config["complexity_rank"]))
        full_means = [float(np.mean(_level_array(seed_rows, family, level))) for level in levels]
        shuffled = np.asarray(full_means)[rng.permutation(len(full_means))]
        results[family] = {
            "all_level_statistics": _candidate_stats(seed_rows, family, levels, samples),
            "evaluated_level_windows": candidates,
            "selected_development_window": best,
            "shuffled_level_label_monotonic": bool(np.all(np.diff(shuffled) > 0)),
        }
        if best is not None:
            passing.append(
                {
                    "family": family,
                    "levels": best["levels"],
                    "statistics": best,
                    "complexity_rank": int(config["complexity_rank"]),
                }
            )
    selected = max(
        passing,
        key=lambda item: (
            len(item["levels"]), item["statistics"]["monotonicity_margin"],
            item["statistics"]["hardest_minus_easiest"]["ci_lower"],
            item["statistics"]["hardest_minus_easiest"]["mean"], -item["complexity_rank"],
        ),
        default=None,
    )
    return results, selected


def _replication_stats(seed_rows: list[dict], selected: dict, raw: dict) -> dict:
    stats = _candidate_stats(
        seed_rows,
        selected["family"],
        list(selected["levels"]),
        int(raw["bootstrap_samples"]),
    )
    stats["blind_replication_pass"] = bool(
        stats["hardest_minus_easiest"]["mean"] > 0
        and stats["hardest_minus_easiest"]["ci_lower"] > 0
        and stats["direction_consistent_seed_fraction"] >= 0.70
    )
    return stats


def _difficulty_labels(levels: list[str]) -> dict[str, str]:
    if len(levels) == 3:
        labels = ["EASY", "MEDIUM", "HARD"]
    elif len(levels) == 4:
        labels = ["EASY", "MEDIUM_LOW", "MEDIUM_HIGH", "HARD"]
    else:
        raise ValueError("difficulty labels require three or four frozen levels")
    return dict(zip(levels, labels, strict=True))


def _make_figures(output: Path, screening_seed: list[dict], replication_seed: list[dict], raw: dict, selected: dict | None, labels: dict[str, str]) -> None:
    import matplotlib.pyplot as plt

    figdir = output / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "FAMILY_A_DECEPTION_ONLY": "family_A_difficulty_curve.png",
        "FAMILY_B_INTERACTION_DENSITY_ONLY": "family_B_difficulty_curve.png",
        "FAMILY_C_CONFLICTING_INTERACTIONS": "family_C_difficulty_curve.png",
    }
    for family, filename in filenames.items():
        levels = list(raw["families"][family]["levels"])
        means = [float(np.mean(_level_array(screening_seed, family, level))) for level in levels]
        standard_errors = [float(np.std(_level_array(screening_seed, family, level), ddof=1) / np.sqrt(len(raw["screening"]["seeds"]))) for level in levels]
        fig, ax = plt.subplots(figsize=(6, 4)); ax.errorbar(levels, means, yerr=standard_errors, marker="o", capsize=3); ax.set_ylabel("terminal normalized regret"); ax.set_title(family); fig.tight_layout(); fig.savefig(figdir / filename, dpi=160); plt.close(fig)

    def placeholder(path: Path, message: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 4)); ax.axis("off"); ax.text(.5, .5, message, ha="center", va="center"); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

    if not selected or not replication_seed:
        placeholder(figdir / "frozen_family_blind_replication.png", "NOT EVALUATED\nNo family frozen")
        placeholder(figdir / "hardest_vs_easiest_paired.png", "NOT EVALUATED\nNo blind replication")
        return
    family = selected["family"]
    levels = list(selected["levels"])
    screen_means = [float(np.mean(_level_array(screening_seed, family, level))) for level in levels]
    blind_means = [float(np.mean(_level_array(replication_seed, family, level))) for level in levels]
    x = np.arange(len(levels)); width = .35
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(x-width/2, screen_means, width, label="screening"); ax.bar(x+width/2, blind_means, width, label="blind replication"); ax.set_xticks(x, [labels.get(level, level) for level in levels]); ax.set_ylabel("terminal normalized regret"); ax.legend(); fig.tight_layout(); fig.savefig(figdir / "frozen_family_blind_replication.png", dpi=160); plt.close(fig)

    easy = _level_array(replication_seed, family, levels[0]); hard = _level_array(replication_seed, family, levels[-1])
    fig, ax = plt.subplots(figsize=(6, 5));
    for index in range(len(easy)): ax.plot([0, 1], [easy[index], hard[index]], color="gray", alpha=.5)
    ax.scatter(np.zeros(len(easy)), easy, label=labels.get(levels[0], levels[0])); ax.scatter(np.ones(len(hard)), hard, label=labels.get(levels[-1], levels[-1])); ax.set_xticks([0, 1], ["easiest", "hardest"]); ax.set_ylabel("seed mean terminal regret"); ax.legend(); fig.tight_layout(); fig.savefig(figdir / "hardest_vs_easiest_paired.png", dpi=160); plt.close(fig)


def _decision_note(summary: dict) -> str:
    screening = summary["family_screening"]
    selected = summary.get("selected_family")
    replication = summary.get("blind_replication")
    failed = [name for name, value in summary["gates"].items() if value["status"] == "FAIL"]
    return f"""# ABEM Phase 0B-r2G Decision Note

## 総合判定

**{summary['verdict']}**

## PR #3から引き継いだnegative result

PR #3の`DIFFICULTY_MANIPULATION_FAIL`を保持した。MVOC-Bは未評価であり、r2V artifactsは変更していない。

## 今回の修正理由

固定探索kernelに対するdifficulty gradientをcontrollerと分離して同定する。固定budgetのため主指標は`1 - F*_32`とした。

## Family A結果

{json.dumps(screening['FAMILY_A_DECEPTION_ONLY'], ensure_ascii=False, indent=2)}

## Family B結果

{json.dumps(screening['FAMILY_B_INTERACTION_DENSITY_ONLY'], ensure_ascii=False, indent=2)}

## Family C結果

{json.dumps(screening['FAMILY_C_CONFLICTING_INTERACTIONS'], ensure_ascii=False, indent=2)}

## 採用または不採用理由

{json.dumps(selected, ensure_ascii=False, indent=2)}

## Freeze時点

`{summary.get('freeze_git_sha', 'NOT_EVALUATED')}`。blind replication前にfamilyとlevelsを保存した。

## Blind replication

{json.dumps(replication, ensure_ascii=False, indent=2)}

## Difficulty labelsの可否

{json.dumps(summary.get('difficulty_labels', {}), ensure_ascii=False, indent=2)}

## 失敗Gate

`{', '.join(failed) if failed else 'なし'}`

## 残る最小主張

{summary['remaining_minimal_claim']}

## MVOC-B実験へ進めるか

この実行内ではMVOC-Bを評価していない。`DIFFICULTY_FAMILY_VALIDATED`の場合のみ、別の事前登録Phaseを設計する資格が得られる。

## Claim Firewall

{CLAIM_FIREWALL}
"""


def run_phase(config_path: str | Path = "configs/phase0b_r2g.yaml", output_dir: str | Path = "results/phase0b_r2g") -> dict:
    raw = load_r2g_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    replication_marker = output / "replication_complete.marker"
    if replication_marker.exists():
        raise RuntimeError("R2G-C blind replication is single-use for this output directory")

    audit_pass, audit = implementation_audit(raw)
    if not audit_pass:
        raise RuntimeError(f"TECHNICAL_FAILURE: {audit}")
    screening_rows = _run_stage(raw, "screening")
    screening_seed = _seed_level(screening_rows)
    family_results, selected = screen_families(screening_seed, raw)

    gates = {
        "G0_IMPLEMENTATION": gate("PASS", f"normalization, target matching, replay, seed and kernel audits passed: {audit}"),
        "G1_FAMILY_SCREEN": gate("PASS" if selected else "FAIL", "at least one development family passed" if selected else "no development family passed the monotonic paired gate"),
        "G2_FREEZE": gate("NOT_EVALUATED", "stopped after G1 failure"),
        "G3_BLIND_REPLICATION": gate("NOT_EVALUATED", "stopped before blind replication"),
        "G4_LABEL_VALIDATION": gate("NOT_EVALUATED", "difficulty labels are not assigned before successful replication"),
    }
    freeze = {
        "phase": raw["name"],
        "status": "FROZEN" if selected else "NOT_EVALUATED",
        "selected_family": selected["family"] if selected else None,
        "selected_levels": selected["levels"] if selected else [],
        "level_configs": {level: _jsonable(level_config(raw, selected["family"], level).__dict__) for level in selected["levels"]} if selected else {},
        "screening_statistics": selected["statistics"] if selected else {},
        "git_commit_sha_before_blind_replication": _git_sha() if selected else None,
        "difficulty_labels": "NOT_ASSIGNED_BEFORE_BLIND_REPLICATION",
    }
    with (output / "frozen_generator.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_jsonable(freeze), handle, allow_unicode=True, sort_keys=False)

    replication_rows: list[dict] = []
    replication_seed: list[dict] = []
    replication_stats = None
    labels: dict[str, str] = {}
    if selected:
        gates["G2_FREEZE"] = gate("PASS", f"family and levels frozen at git SHA {freeze['git_commit_sha_before_blind_replication']}")
        selection = {selected["family"]: list(selected["levels"])}
        replication_rows = _run_stage(raw, "replication", selection)
        replication_marker.write_text(json.dumps({"completed": True, "git_sha": _git_sha()}), encoding="utf-8")
        replication_seed = _seed_level(replication_rows)
        replication_stats = _replication_stats(replication_seed, selected, raw)
        if replication_stats["blind_replication_pass"]:
            gates["G3_BLIND_REPLICATION"] = gate("PASS", f"paired blind gate passed: {replication_stats['hardest_minus_easiest']}")
            labels = _difficulty_labels(list(selected["levels"]))
            gates["G4_LABEL_VALIDATION"] = gate("PASS", "labels assigned only after the blind gate passed")
        else:
            gates["G3_BLIND_REPLICATION"] = gate("FAIL", f"paired blind gate failed: {replication_stats}")
            gates["G4_LABEL_VALIDATION"] = gate("NOT_EVALUATED", "blind replication failed; no difficulty labels assigned")

    if gates["G1_FAMILY_SCREEN"]["status"] == "FAIL":
        verdict = "NO_VALID_DIFFICULTY_FAMILY"
        remaining = "development seedsで単調かつpaired CIを通るgenerator familyを同定できなかった。"
    elif gates["G2_FREEZE"]["status"] != "PASS":
        verdict = "TECHNICAL_FAILURE"
        remaining = "freeze protocolを完了できなかったため科学結論を出さない。"
    elif gates["G3_BLIND_REPLICATION"]["status"] == "FAIL":
        verdict = "DIFFICULTY_NOT_REPLICATED"
        remaining = "developmentで選択したdifficulty gradientは独立seedで再現せず、generator overfitとして停止する。"
    elif all(item["status"] == "PASS" for item in gates.values()):
        verdict = "DIFFICULTY_FAMILY_VALIDATED"
        remaining = "ABEM固定探索kernelに対してblind replicationされたsynthetic difficulty gradientを同定した。"
    else:
        verdict = "TECHNICAL_FAILURE"
        remaining = "技術Gateが完了していないため科学結論を出さない。"

    summary = {
        "phase": raw["name"],
        "verdict": verdict,
        "git_commit_sha": _git_sha(),
        "previous_pr_3_verdict_retained": "DIFFICULTY_MANIPULATION_FAIL",
        "previous_r2v_artifacts_modified": False,
        "primary_metric": "terminal_normalized_regret_at_fixed_depth_32",
        "implementation_audit": audit,
        "family_screening": family_results,
        "selected_family": selected,
        "freeze_git_sha": freeze["git_commit_sha_before_blind_replication"],
        "blind_replication": replication_stats,
        "difficulty_labels": labels,
        "gates": gates,
        "remaining_minimal_claim": remaining,
        "mvoc_b_evaluated": False,
        "phase0c_data_used": False,
        "claim_firewall": CLAIM_FIREWALL,
    }
    _write_csv(output / "screening_seed_level.csv", screening_seed, SEED_FIELDS)
    _write_csv(output / "replication_seed_level.csv", replication_seed, SEED_FIELDS)
    _write_csv(output / "generator_metrics.csv", screening_rows + replication_rows)
    (output / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "gates.json").write_text(json.dumps(gates, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "decision_note.md").write_text(_decision_note(summary), encoding="utf-8")
    (output / "README.md").write_text(
        "# Phase 0B-r2G\n\nGenerator-only screening and blind replication. MVOC-B and Phase 0C were not run. See `decision_note.md`.\n",
        encoding="utf-8",
    )
    _make_figures(output, screening_seed, replication_seed, raw, selected, labels)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ABEM Phase 0B-r2G generator identification")
    parser.add_argument("--config", default="configs/phase0b_r2g.yaml")
    parser.add_argument("--output", default="results/phase0b_r2g")
    args = parser.parse_args()
    summary = run_phase(args.config, args.output)
    print(json.dumps({"verdict": summary["verdict"], "gates": summary["gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
