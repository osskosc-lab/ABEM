from copy import deepcopy

from abem.difficulty_validation import validate_difficulty
from abem.r2v_simulation import load_r2v_config


def test_difficulty_validation_is_reproducible_and_seed_level():
    raw = deepcopy(load_r2v_config("configs/phase0b_r2v.yaml"))
    raw["d0"]["seeds"] = (200, 201)
    raw["d0"]["episodes_per_seed"] = 1
    raw["d0"]["fixed_search_depth"] = 4
    raw["search"].update(population_size=8, elite_size=2, t_max=8, t_min=2)
    raw["landscape"].update(dimension=8, block_size=4)
    raw["generators"] = {
        "G1": {"deceptive_bonus": 0.55, "interaction_edges": 1, "interaction_strength": 0.1},
        "G2": {"deceptive_bonus": 0.72, "interaction_edges": 2, "interaction_strength": 0.3},
        "G3": {"deceptive_bonus": 0.88, "interaction_edges": 3, "interaction_strength": 0.5},
    }
    left, left_summary = validate_difficulty(raw)
    right, right_summary = validate_difficulty(raw)
    assert left == right
    assert left_summary == right_summary
    assert len(left) == 2 * 3
