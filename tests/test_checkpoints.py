import numpy as np

from abem.checkpoints import continue_from_checkpoint, generate_trajectory
from abem.config import LandscapeConfig, SearchConfig
from abem.landscapes import make_landscape


class BlackBoxLandscape:
    dimension = 8

    def evaluate(self, population):
        return np.mean(np.asarray(population), axis=1)


def _search():
    return SearchConfig(population_size=8, elite_size=2, t_max=12, fixed_depth=4)


def test_same_seed_same_checkpoint_replay():
    search = _search()
    landscape = make_landscape(17, LandscapeConfig(dimension=8, block_size=4, interaction_edges=2))
    trajectory = generate_trajectory(landscape, agent_seed=23, config=search, checkpoint_steps=(4, 8))
    replay = continue_from_checkpoint(landscape, trajectory.at_step(4), config=search, horizon=4)
    expected = trajectory.at_step(8)
    assert np.array_equal(replay.population, expected.population)
    assert np.array_equal(replay.population_scores, expected.population_scores)
    assert replay.best_score == expected.best_score
    assert replay.cumulative_hazard == expected.cumulative_hazard


def test_checkpoint_restore_exactness():
    search = _search()
    landscape = make_landscape(19, LandscapeConfig(dimension=8, block_size=4, interaction_edges=2))
    trajectory = generate_trajectory(landscape, agent_seed=29, config=search, checkpoint_steps=(4,))
    checkpoint = trajectory.checkpoints[0]
    restored = continue_from_checkpoint(landscape, checkpoint, config=search, horizon=0)
    assert restored is checkpoint
    assert np.array_equal(restored.best_candidate, checkpoint.best_candidate)


def test_problem_rng_agent_rng_separation():
    search = _search()
    cfg = LandscapeConfig(dimension=8, block_size=4, interaction_edges=2)
    left = make_landscape(31, cfg)
    right = make_landscape(32, cfg)
    left_trace = generate_trajectory(left, agent_seed=41, config=search, checkpoint_steps=(1,))
    right_trace = generate_trajectory(right, agent_seed=41, config=search, checkpoint_steps=(1,))
    assert not np.array_equal(left.target, right.target)
    assert np.array_equal(left_trace.at_step(1).population, right_trace.at_step(1).population)


def test_no_target_leakage_in_checkpoint_kernel():
    trajectory = generate_trajectory(BlackBoxLandscape(), agent_seed=7, config=_search(), checkpoint_steps=(4,))
    assert trajectory.at_step(4).best_score >= 0.0
