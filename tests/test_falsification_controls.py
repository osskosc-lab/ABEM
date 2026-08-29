from copy import deepcopy

import numpy as np
import pytest
import yaml

from abem.boundary_analysis import shuffled_feature_records
from abem.falsification import (
    _build_cases,
    _calibrate,
    _case_policy_steps,
    _load_phase_config,
    _records_for_cases,
    _runtime_replay_audit,
    _seed_level,
)


def _tiny_raw():
    raw = _load_phase_config("configs/phase0b_r1f.yaml")
    raw = deepcopy(raw)
    raw["landscape"].update(dimension=8, block_size=4, interaction_edges=2)
    raw["search"].update(population_size=8, elite_size=2, t_max=12, t_min=2)
    raw["checkpoint_horizons"] = [2, 4, 8]
    raw["fixed_depth_candidates"] = [4, 8, 12]
    raw["patience_candidates"] = [2, 4]
    raw["metric"]["bootstrap_samples"] = 50
    return raw


def test_signal_shuffle_preserves_marginals():
    raw = _tiny_raw()
    case = _build_cases(raw, split="calibration", seeds=(100,), rollouts_override=1, episodes_override=1)[0]
    shuffled = shuffled_feature_records(case.trajectory, seed=44)
    for feature in ("gain", "score_std", "diversity", "stagnation"):
        before = sorted(getattr(x, feature) for x in case.trajectory.step_records)
        after = sorted(getattr(x, feature) for x in shuffled)
        assert np.allclose(before, after)


def test_runtime_replay_gate():
    passed, reason = _runtime_replay_audit(_tiny_raw())
    assert passed, reason


def test_random_matched_stop_budget():
    raw = _tiny_raw()
    cases = _build_cases(raw, split="calibration", seeds=(100, 101), rollouts_override=1, episodes_override=1)
    frozen = _calibrate(raw, cases)
    steps = _case_policy_steps(cases, raw=raw, frozen=frozen)
    for difficulty in raw["difficulties"]:
        indices = [i for i, case in enumerate(cases) if case.difficulty == difficulty]
        assert sorted(steps["RANDOM_MATCHED"][i] for i in indices) == sorted(
            steps["FULL_AB"][i] for i in indices
        )


def test_seed_level_pairing():
    raw = _tiny_raw()
    cases = _build_cases(raw, split="calibration", seeds=(100, 101), rollouts_override=1, episodes_override=1)
    frozen = _calibrate(raw, cases)
    rows = _records_for_cases(cases, _case_policy_steps(cases, raw=raw, frozen=frozen), raw)
    seed_rows = _seed_level(rows)
    policies = {x["policy"] for x in seed_rows}
    for policy in policies:
        assert {x["seed"] for x in seed_rows if x["policy"] == policy} == {100, 101}


def test_no_confirmatory_seed_access(tmp_path):
    raw = yaml.safe_load(open("configs/phase0b_r1f.yaml", encoding="utf-8"))
    assert set(raw["calibration_seeds"] + raw["validation_seeds"]) == set(range(100, 110))
    raw["confirmatory"] = True
    path = tmp_path / "forbidden.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="confirmatory"):
        _load_phase_config(path)
