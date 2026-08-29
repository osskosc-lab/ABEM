from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LandscapeConfig


@dataclass(frozen=True)
class DeceptiveModularLandscape:
    target: np.ndarray
    decoys: np.ndarray
    edges: tuple[tuple[int, int, int], ...]
    decoy_compatible_edges: frozenset[tuple[int, int]]
    config: LandscapeConfig

    @property
    def dimension(self) -> int:
        return int(self.target.size)

    @property
    def n_blocks(self) -> int:
        return self.dimension // self.config.block_size

    @property
    def max_raw_score(self) -> float:
        if not self.edges:
            return self.n_blocks * self.config.optimum_bonus
        target_compatible = len(self.edges) - len(self.decoy_compatible_edges)
        target_interaction = self.config.interaction_strength * target_compatible / len(self.edges)
        return self.n_blocks * self.config.optimum_bonus + target_interaction

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        pop = np.asarray(population, dtype=np.int8)
        if pop.ndim == 1:
            pop = pop[None, :]
        if pop.shape[1] != self.dimension:
            raise ValueError("candidate dimension does not match landscape")

        scores = np.zeros(pop.shape[0], dtype=float)
        bs = self.config.block_size
        for b in range(self.n_blocks):
            sl = slice(b * bs, (b + 1) * bs)
            block = pop[:, sl]
            target = self.target[sl]
            decoy = self.decoys[b]
            is_target = np.all(block == target, axis=1)
            is_decoy = np.all(block == decoy, axis=1)
            partial = np.mean(block == target, axis=1) * 0.45
            scores += np.where(
                is_target,
                self.config.optimum_bonus,
                np.where(is_decoy, self.config.deceptive_bonus, partial),
            )

        if self.edges:
            satisfied = np.zeros(pop.shape[0], dtype=float)
            for i, j, parity in self.edges:
                if (i, j) in self.decoy_compatible_edges:
                    # Target and its bitwise-complement decoy have identical XOR
                    # parity.  A conflicting edge must therefore reward the exact
                    # two-bit decoy motif rather than an indistinguishable parity.
                    decoy_i = int(1 - self.target[i])
                    decoy_j = int(1 - self.target[j])
                    satisfied += ((pop[:, i] == decoy_i) & (pop[:, j] == decoy_j)).astype(float)
                else:
                    satisfied += ((pop[:, i] ^ pop[:, j]) == parity).astype(float)
            scores += self.config.interaction_strength * satisfied / len(self.edges)

        return np.clip(scores / self.max_raw_score, 0.0, 1.0)


def make_landscape(seed: int, config: LandscapeConfig) -> DeceptiveModularLandscape:
    if config.dimension % config.block_size != 0:
        raise ValueError("dimension must be divisible by block_size")
    if not 0.0 <= config.decoy_compatible_edge_fraction <= 1.0:
        raise ValueError("decoy_compatible_edge_fraction must be within [0, 1]")
    if config.interaction_edges == 0 and config.decoy_compatible_edge_fraction != 0.0:
        raise ValueError("decoy-compatible interactions require at least one edge")

    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, size=config.dimension, dtype=np.int8)
    n_blocks = config.dimension // config.block_size
    decoys = np.empty((n_blocks, config.block_size), dtype=np.int8)

    for b in range(n_blocks):
        sl = slice(b * config.block_size, (b + 1) * config.block_size)
        decoys[b] = 1 - target[sl]

    possible = [(i, j) for i in range(config.dimension) for j in range(i + 1, config.dimension)]
    rng.shuffle(possible)
    edges = []
    for i, j in possible[: config.interaction_edges]:
        parity = int(target[i] ^ target[j])
        edges.append((i, j, parity))

    conflict_count = int(round(config.decoy_compatible_edge_fraction * len(edges)))
    decoy_compatible_edges = frozenset((i, j) for i, j, _ in edges[:conflict_count])
    if edges and conflict_count:
        # A non-target candidate must lose at least this much block score.  The
        # maximum interaction advantage over the target must be strictly smaller,
        # which keeps the target as the unique analytically known optimum.
        best_partial = 0.45 * (config.block_size - 1) / config.block_size
        minimum_block_loss = min(
            config.optimum_bonus - config.deceptive_bonus,
            config.optimum_bonus - best_partial,
        )
        interaction_advantage = config.interaction_strength * conflict_count / len(edges)
        if interaction_advantage >= minimum_block_loss:
            raise ValueError("conflicting interaction strength can displace the known target optimum")

    return DeceptiveModularLandscape(
        target=target,
        decoys=decoys,
        edges=tuple(edges),
        decoy_compatible_edges=decoy_compatible_edges,
        config=config,
    )
