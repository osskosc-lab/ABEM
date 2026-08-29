import numpy as np

from abem.config import LandscapeConfig
from abem.landscapes import make_landscape


def test_target_is_known_global_optimum():
    cfg = LandscapeConfig(dimension=16, block_size=4, interaction_edges=6)
    landscape = make_landscape(42, cfg)
    score = landscape.evaluate(landscape.target)[0]
    assert np.isclose(score, 1.0)


def test_deceptive_solution_is_suboptimal():
    cfg = LandscapeConfig(dimension=16, block_size=4, interaction_edges=6)
    landscape = make_landscape(42, cfg)
    decoy = np.concatenate(list(landscape.decoys))
    score = landscape.evaluate(decoy)[0]
    assert 0.0 <= score < 1.0


def test_landscape_is_seed_reproducible():
    cfg = LandscapeConfig(dimension=16, block_size=4, interaction_edges=6)
    a = make_landscape(7, cfg)
    b = make_landscape(7, cfg)
    assert np.array_equal(a.target, b.target)
    assert a.edges == b.edges
