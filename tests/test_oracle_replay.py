import numpy as np

from abem.checkpoints import continue_from_checkpoint, generate_trajectory
from abem.config import LandscapeConfig, MetricConfig, SearchConfig
from abem.landscapes import make_landscape
from abem.metrics import efficiency_error
from abem.oracle import estimate_oracle_value, rollout_seed


def _fixture():
    search = SearchConfig(population_size=8, elite_size=2, t_max=12, fixed_depth=4)
    landscape = make_landscape(101, LandscapeConfig(dimension=8, block_size=4, interaction_edges=2))
    trajectory = generate_trajectory(landscape, agent_seed=103, config=search, checkpoint_steps=(4,))
    return search, landscape, trajectory


def test_oracle_future_seed_independence():
    search, landscape, trajectory = _fixture()
    checkpoint = trajectory.at_step(4)
    seed_a = rollout_seed(101, 4, 4, 0)
    seed_b = rollout_seed(101, 4, 4, 1)
    assert seed_a != seed_b != 103
    a = continue_from_checkpoint(landscape, checkpoint, config=search, horizon=4, future_seed=seed_a)
    b = continue_from_checkpoint(landscape, checkpoint, config=search, horizon=4, future_seed=seed_a)
    assert np.array_equal(a.population, b.population)


def test_oracle_estimate_is_reproducible():
    search, landscape, trajectory = _fixture()
    kwargs = dict(
        problem_seed=101,
        search=search,
        metric=MetricConfig(cost_weight=0.25),
        horizons=(4, 8),
        rollouts=3,
    )
    a = estimate_oracle_value(landscape, trajectory.at_step(4), **kwargs)
    b = estimate_oracle_value(landscape, trajectory.at_step(4), **kwargs)
    assert a == b


def test_metric_recomputation_consistency():
    search, _, trajectory = _fixture()
    record = trajectory.at_step(4)
    metric = MetricConfig(cost_weight=0.25)
    expected = (1.0 - record.best_score) + 0.25 * 4 / search.t_max
    assert np.isclose(efficiency_error(record.best_score, 4, search, metric), expected)
