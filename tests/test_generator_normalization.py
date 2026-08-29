from dataclasses import replace
from itertools import product

import numpy as np

from abem.generator_identification import level_config, load_r2g_config
from abem.landscapes import make_landscape


def test_targets_are_matched_and_normalized_across_all_levels():
    raw = load_r2g_config("configs/phase0b_r2g.yaml")
    landscapes = [
        make_landscape(9123, level_config(raw, family, level))
        for family, family_config in raw["families"].items()
        for level in family_config["levels"]
    ]
    assert all(np.array_equal(landscapes[0].target, item.target) for item in landscapes[1:])
    assert all(np.isclose(item.evaluate(item.target)[0], 1.0) for item in landscapes)


def test_conflicting_interaction_target_is_unique_enumerated_optimum():
    raw = load_r2g_config("configs/phase0b_r2g.yaml")
    config = replace(
        level_config(raw, "FAMILY_C_CONFLICTING_INTERACTIONS", "C4"),
        dimension=8,
        block_size=4,
        interaction_edges=6,
    )
    landscape = make_landscape(44, config)
    population = np.asarray(list(product((0, 1), repeat=8)), dtype=np.int8)
    scores = landscape.evaluate(population)
    target_index = int(np.flatnonzero(np.all(population == landscape.target, axis=1))[0])
    assert np.isclose(scores[target_index], 1.0)
    assert np.sum(np.isclose(scores, 1.0)) == 1


def test_decoy_compatible_edges_are_not_xor_aliases():
    raw = load_r2g_config("configs/phase0b_r2g.yaml")
    base = make_landscape(55, level_config(raw, "FAMILY_C_CONFLICTING_INTERACTIONS", "C1"))
    conflict = make_landscape(55, level_config(raw, "FAMILY_C_CONFLICTING_INTERACTIONS", "C4"))
    assert base.edges == conflict.edges
    assert not base.decoy_compatible_edges
    assert conflict.decoy_compatible_edges
    assert np.array_equal(base.target, conflict.target)
    assert np.isclose(base.evaluate(base.target)[0], conflict.evaluate(conflict.target)[0])
