from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from abem.r2v_protocol import ProtocolViolation, assert_stage_allowed, validate_r2v_protocol


def test_r2v_protocol_is_ready_for_d0_only():
    result = validate_r2v_protocol()
    assert result["status"] == "READY_FOR_D0_ONLY"
    assert set(result["D0_seeds"]).isdisjoint(result["D1_seeds"])
    assert set(result["D1_seeds"]).isdisjoint(result["D2_seeds"])
    assert set(result["D0_seeds"]).isdisjoint(result["confirmatory_seeds"])
    assert set(result["D1_seeds"]).isdisjoint(result["confirmatory_seeds"])
    assert set(result["D2_seeds"]).isdisjoint(result["confirmatory_seeds"])


def test_stage_lock_order():
    assert_stage_allowed("D0")
    with pytest.raises(ProtocolViolation, match="D1 is locked"):
        assert_stage_allowed("D1")
    assert_stage_allowed("D1", d0_passed=True)
    with pytest.raises(ProtocolViolation, match="D2 is locked"):
        assert_stage_allowed("D2", d0_passed=True)
    assert_stage_allowed("D2", d0_passed=True, d1_frozen=True)
    with pytest.raises(ProtocolViolation, match="Phase 0C is forbidden"):
        assert_stage_allowed("PHASE0C", d0_passed=True, d1_frozen=True)


def test_prior_r1f_seed_reuse_is_rejected(tmp_path: Path):
    raw = yaml.safe_load(Path("configs/phase0b_r2v.yaml").read_text(encoding="utf-8"))
    raw = deepcopy(raw)
    raw["seed_partitions"]["D0_DIFFICULTY"][0] = 100
    config = tmp_path / "r2v.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    confirmatory = tmp_path / "confirmatory.yaml"
    confirmatory.write_text(Path("configs/confirmatory.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    raw["confirmatory_config"] = "confirmatory.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolViolation, match="seed overlap"):
        validate_r2v_protocol(config.name, repo_root=tmp_path)


def test_confirmatory_overlap_is_rejected(tmp_path: Path):
    raw = yaml.safe_load(Path("configs/phase0b_r2v.yaml").read_text(encoding="utf-8"))
    raw = deepcopy(raw)
    raw["seed_partitions"]["D2_BLIND_VALIDATION"][0] = 1000
    raw["confirmatory_config"] = "confirmatory.yaml"
    (tmp_path / "r2v.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    (tmp_path / "confirmatory.yaml").write_text(
        Path("configs/confirmatory.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(ProtocolViolation, match="confirmatory seed leakage"):
        validate_r2v_protocol("r2v.yaml", repo_root=tmp_path)


def test_rejected_features_cannot_return(tmp_path: Path):
    raw = yaml.safe_load(Path("configs/phase0b_r2v.yaml").read_text(encoding="utf-8"))
    raw = deepcopy(raw)
    raw["minimal_boundary"]["features"] = ["stagnation", "recent_gain", "score_std"]
    raw["confirmatory_config"] = "confirmatory.yaml"
    (tmp_path / "r2v.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    (tmp_path / "confirmatory.yaml").write_text(
        Path("configs/confirmatory.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(ProtocolViolation, match="only stagnation and recent_gain"):
        validate_r2v_protocol("r2v.yaml", repo_root=tmp_path)
