import numpy as np

from abem.boundary import AdaptiveBoundary
from abem.config import MemoryConfig, SearchConfig
from abem.memory import SearchMemory


def test_stagnation_accumulates_more_hazard_than_promising_search():
    cfg = SearchConfig()
    stalled = AdaptiveBoundary(cfg)
    promising = AdaptiveBoundary(cfg)

    for t in range(8):
        stalled.step(gain=0.0, uncertainty=0.01, diversity=0.01, stagnation=t)
        promising.step(gain=0.05, uncertainty=0.2, diversity=0.4, stagnation=0)

    assert stalled.cumulative_hazard > promising.cumulative_hazard


def test_memory_update_is_bounded_and_nonzero():
    cfg = MemoryConfig(learning_rate=0.5, decay=0.1, clip=0.25)
    memory = SearchMemory.zeros(4, cfg)
    candidates = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int8)
    scores = np.array([1.0, 0.0])
    memory.update(candidates, scores)
    assert np.any(np.abs(memory.weights) > 0)
    assert np.max(np.abs(memory.weights)) <= cfg.clip + 1e-12


def test_zero_advantage_does_not_write_directional_memory():
    cfg = MemoryConfig()
    memory = SearchMemory.zeros(4, cfg)
    candidates = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int8)
    scores = np.array([0.5, 0.5])
    memory.update(candidates, scores)
    assert np.allclose(memory.weights, 0.0)
