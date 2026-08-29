import numpy as np

from abem.agents import run_episode
from abem.config import ExperimentConfig, LandscapeConfig, SearchConfig
from abem.experiment import run_experiment


class BlackBoxLandscape:
    """Exposes only dimension + evaluate; no target/optimum internals."""

    dimension = 8

    def evaluate(self, population):
        pop = np.asarray(population)
        return np.mean(pop, axis=1)


def test_agent_does_not_require_hidden_target_access():
    cfg = SearchConfig(population_size=8, elite_size=2, t_max=8, fixed_depth=4)
    result = run_episode(
        BlackBoxLandscape(),
        rng=np.random.default_rng(123),
        config=cfg,
        memory=None,
        adaptive_boundary=False,
    )
    assert 0.0 <= result.best_score <= 1.0
    assert result.steps == 4


def test_experiment_is_exactly_reproducible():
    cfg = ExperimentConfig(
        name="test",
        seeds=(3, 4),
        episodes_per_seed=3,
        confirmatory=False,
        landscape=LandscapeConfig(dimension=16, block_size=4, interaction_edges=4),
        search=SearchConfig(population_size=10, elite_size=3, t_max=10, fixed_depth=5),
    )
    a = run_experiment(cfg)
    b = run_experiment(cfg)
    assert a["per_seed_efficiency_error"] == b["per_seed_efficiency_error"]
    assert a["records"] == b["records"]
    assert a["verdict"] == "NON_CONFIRMATORY_DO_NOT_INTERPRET"
