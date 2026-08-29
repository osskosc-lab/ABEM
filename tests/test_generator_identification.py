from copy import deepcopy

import numpy as np

from abem.generator_identification import (
    _run_stage,
    _seed_level,
    implementation_audit,
    level_config,
    load_r2g_config,
    screen_families,
)


def test_each_family_changes_only_one_primary_axis():
    raw = load_r2g_config("configs/phase0b_r2g.yaml")
    expected = {
        "FAMILY_A_DECEPTION_ONLY": "deceptive_bonus",
        "FAMILY_B_INTERACTION_DENSITY_ONLY": "interaction_edges",
        "FAMILY_C_CONFLICTING_INTERACTIONS": "decoy_compatible_edge_fraction",
    }
    for family, axis in expected.items():
        levels = list(raw["families"][family]["levels"])
        left = level_config(raw, family, levels[0]).__dict__
        right = level_config(raw, family, levels[-1]).__dict__
        changed = {name for name in left if left[name] != right[name]}
        assert changed == {axis}


def test_seed_splits_exclude_r2v_and_confirmatory_data():
    raw = load_r2g_config("configs/phase0b_r2g.yaml")
    used = set(raw["screening"]["seeds"]) | set(raw["replication"]["seeds"])
    assert not used & set(raw["forbidden_r2v_seeds"])
    assert not used & set(range(1000, 1030))


def test_implementation_audit_passes():
    passed, details = implementation_audit(load_r2g_config("configs/phase0b_r2g.yaml"))
    assert passed, details


def test_tiny_stage_is_exactly_reproducible_and_paired():
    raw = deepcopy(load_r2g_config("configs/phase0b_r2g.yaml"))
    raw["screening"]["seeds"] = (300,)
    raw["screening"]["episodes_per_seed"] = 1
    raw["search"].update(population_size=8, elite_size=2, fixed_depth=4, t_max=8, t_min=2)
    raw["landscape"].update(dimension=8, block_size=4)
    selection = {"FAMILY_A_DECEPTION_ONLY": ["A1", "A2", "A3"]}
    left = _run_stage(raw, "screening", selection)
    right = _run_stage(raw, "screening", selection)
    assert left == right
    assert len({row["target_signature"] for row in left}) == 1
    assert all(row["problem_seed"] == left[0]["problem_seed"] for row in left)
    assert all(row["agent_seed"] == left[0]["agent_seed"] for row in left)


def test_screening_selects_monotonic_family_from_seed_level_data():
    raw = load_r2g_config("configs/phase0b_r2g.yaml")
    rows = []
    for seed in raw["screening"]["seeds"]:
        for family, config in raw["families"].items():
            for index, level in enumerate(config["levels"]):
                value = 0.01 + 0.02 * index if family == "FAMILY_A_DECEPTION_ONLY" else 0.02
                rows.append({
                    "stage": "screening", "seed": seed, "family": family, "level": level,
                    "mean_terminal_regret": value, "median_terminal_regret": value,
                    "mean_efficiency_error": value + .125, "mean_best_score": 1-value,
                    "success_rate_at_threshold": 0.0, "global_optimum_success_rate": 0.0,
                    "mean_final_population_diversity": .5, "episodes": 16,
                })
    _, selected = screen_families(rows, raw)
    assert selected is not None
    assert selected["family"] == "FAMILY_A_DECEPTION_ONLY"
    assert selected["levels"] == ["A1", "A2", "A3", "A4"]
