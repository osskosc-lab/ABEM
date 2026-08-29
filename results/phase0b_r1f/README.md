# Phase 0B-r1F artifacts

Frozen falsification run for adaptive stopping with memory disabled (`memory_bias: 0.0`).

Reproduce with:

```bash
python -m abem.falsification \
  --config configs/phase0b_r1f.yaml \
  --output results/phase0b_r1f
```

The independent statistical unit is the seed. `checkpoint_oracle_metrics.csv`
is diagnostic data and checkpoints must not be treated as independent samples.

The local rollout Oracle is not an absolute optimal-stopping oracle. The stored
`oracle_regret` is therefore a signed policy-minus-local-Oracle error and can be
negative; paired policy differences are the primary comparison.

Phase 0C confirmatory seeds were not used.
