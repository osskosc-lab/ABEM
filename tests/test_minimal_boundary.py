from dataclasses import replace

import numpy as np

from abem.checkpoints import generate_trajectory
from abem.config import LandscapeConfig, SearchConfig
from abem.landscapes import make_landscape
from abem.minimal_boundary import (
    MinimalBoundaryParameters,
    boundary_value,
    ema_gain_series,
    minimal_stop_step,
    shuffled_gbar,
)


def _fixture():
    search = SearchConfig(population_size=8, elite_size=2, t_min=2, t_max=12, memory_bias=0.0)
    landscape = make_landscape(31, LandscapeConfig(dimension=8, block_size=4, interaction_edges=2))
    return search, generate_trajectory(landscape, agent_seed=41, config=search, checkpoint_steps=(4, 8))


def test_boundary_equals_p0_when_gain_zero():
    params = MinimalBoundaryParameters(p0=8, delta_p=16, g0=0.1)
    assert boundary_value(0.0, params) == 8


def test_boundary_monotonic_and_bounded():
    params = MinimalBoundaryParameters(p0=8, delta_p=16, g0=0.1)
    values = np.asarray([boundary_value(x, params) for x in np.linspace(0, 10, 1000)])
    assert np.all(np.diff(values) >= 0)
    assert np.all(values >= params.p0)
    assert np.all(values <= params.p0 + params.delta_p)


def test_ema_gain_reproducibility_and_decay():
    a = ema_gain_series([0, 1, 0, 0], alpha=0.5)
    b = ema_gain_series([0, 1, 0, 0], alpha=0.5)
    assert np.array_equal(a, b)
    assert np.allclose(a, [0, 0.5, 0.25, 0.125])


def test_gain_shuffle_preserves_ema_marginal():
    _, trajectory = _fixture()
    before = ema_gain_series([x.gain for x in trajectory.step_records], alpha=0.5)
    after = shuffled_gbar(trajectory, alpha=0.5, seed=73)
    assert np.allclose(np.sort(before), np.sort(after))


def test_stop_rule_ignores_future_records():
    search, trajectory = _fixture()
    params = MinimalBoundaryParameters(p0=2, delta_p=4, g0=0.01)
    stop = minimal_stop_step(trajectory, params=params, t_min=search.t_min)
    changed = list(trajectory.step_records)
    for index in range(stop, len(changed)):
        changed[index] = replace(changed[index], gain=1e6, stagnation=0)
    assert minimal_stop_step(replace(trajectory, step_records=tuple(changed)), params=params, t_min=search.t_min) == stop
