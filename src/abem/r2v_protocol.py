from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml


class ProtocolViolation(RuntimeError):
    pass


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _as_set(values: Iterable[int]) -> set[int]:
    return {int(x) for x in values}


def validate_r2v_protocol(
    config_path: str | Path = "configs/phase0b_r2v.yaml",
    *,
    repo_root: str | Path = ".",
) -> dict:
    root = Path(repo_root)
    raw = load_yaml(root / config_path)

    if raw.get("confirmatory", False):
        raise ProtocolViolation("r2V must remain non-confirmatory")
    if raw.get("execution_lock") != "D0_ONLY":
        raise ProtocolViolation("experiment ground must remain D0_ONLY before D0 validation")
    if not raw.get("forbid_confirmatory_access", False):
        raise ProtocolViolation("confirmatory access guard must be enabled")

    parts = raw["seed_partitions"]
    d0 = _as_set(parts["D0_DIFFICULTY"])
    d1 = _as_set(parts["D1_CALIBRATION"])
    d2 = _as_set(parts["D2_BLIND_VALIDATION"])
    prior = _as_set(raw.get("prior_phase_seeds", []))

    named = {"D0": d0, "D1": d1, "D2": d2, "PRIOR": prior}
    names = list(named)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            overlap = named[left_name] & named[right_name]
            if overlap:
                raise ProtocolViolation(
                    f"seed overlap {left_name}/{right_name}: {sorted(overlap)}"
                )

    confirmatory_path = root / raw["confirmatory_config"]
    confirmatory = load_yaml(confirmatory_path)
    if not confirmatory.get("confirmatory", False):
        raise ProtocolViolation("referenced Phase 0C config is not marked confirmatory")
    confirmatory_seeds = _as_set(confirmatory["seeds"])
    development = d0 | d1 | d2 | prior
    overlap = development & confirmatory_seeds
    if overlap:
        raise ProtocolViolation(f"confirmatory seed leakage: {sorted(overlap)}")

    search = raw["search"]
    if float(search.get("memory_bias", 0.0)) != 0.0:
        raise ProtocolViolation("memory must remain disabled in r2V")

    boundary = raw["minimal_boundary"]
    if float(boundary["alpha"]) != 0.5:
        raise ProtocolViolation("alpha is preregistered at 0.5")
    if set(boundary["features"]) != {"stagnation", "recent_gain"}:
        raise ProtocolViolation("r2V may use only stagnation and recent_gain")
    forbidden = {"score_std", "diversity"}
    if not forbidden.issubset(set(boundary["forbidden_features"])):
        raise ProtocolViolation("r1F-rejected features must stay forbidden")
    if bool(boundary.get("cumulative_hazard", True)):
        raise ProtocolViolation("cumulative hazard must remain disabled in r2V")

    d0_cfg = raw["D0_difficulty_validation"]
    if d0_cfg.get("status") != "UNVALIDATED":
        raise ProtocolViolation("difficulty labels must remain unvalidated before D0")

    return {
        "status": "READY_FOR_D0_ONLY",
        "D0_seeds": sorted(d0),
        "D1_seeds": sorted(d1),
        "D2_seeds": sorted(d2),
        "confirmatory_seeds": sorted(confirmatory_seeds),
    }


def assert_stage_allowed(
    stage: str,
    *,
    d0_passed: bool = False,
    d1_frozen: bool = False,
) -> None:
    stage = stage.upper()
    if stage == "D0":
        return
    if stage == "D1":
        if not d0_passed:
            raise ProtocolViolation("D1 is locked until D0 difficulty validation passes")
        return
    if stage == "D2":
        if not d0_passed or not d1_frozen:
            raise ProtocolViolation("D2 is locked until D0 PASS and D1 freeze")
        return
    if stage in {"PHASE0C", "0C", "CONFIRMATORY"}:
        raise ProtocolViolation("Phase 0C is forbidden during r2V preparation/validation")
    raise ProtocolViolation(f"unknown stage: {stage}")
