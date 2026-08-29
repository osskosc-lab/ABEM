from copy import deepcopy

import numpy as np

from abem.r2v_simulation import _policy_steps, load_r2v_config


def test_d0_d1_d2_and_confirmatory_seeds_are_disjoint():
    raw = load_r2v_config("configs/phase0b_r2v.yaml")
    d0, d1, d2 = (set(raw[name]["seeds"]) for name in ("d0", "d1", "d2"))
    assert not (d0 & d1 or d0 & d2 or d1 & d2)
    assert not set(range(1000, 1030)) & (d0 | d1 | d2)


def test_config_fixes_search_kernel_and_excludes_removed_features():
    raw = load_r2v_config("configs/phase0b_r2v.yaml")
    assert raw["search"]["population_size"] == 24
    assert raw["search"]["elite_size"] == 6
    assert raw["search"]["mutation_rate"] == 0.08
    assert raw["search"]["memory_bias"] == 0.0
    assert set(raw["delta_p_candidates"]) == {4, 8, 16}
    assert raw["alpha"] == 0.5
