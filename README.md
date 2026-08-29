# ABEM — Adaptive Boundary Exploration with Memory

ABEM is a falsification-first toy research program for testing whether a **history-dependent adaptive stopping boundary** can improve out-of-distribution search efficiency relative to a tuned fixed-depth baseline.

## Phase 0-r0 claim

> A history-dependent adaptive stopping boundary improves OOD search efficiency relative to a frozen fixed-depth baseline.

The goal function is fixed. Only the **search termination condition** is allowed to adapt.

## Conditions

- `BASE`: fixed boundary, no memory
- `AB`: adaptive boundary, no memory
- `MEM`: fixed boundary, memory
- `ABEM`: adaptive boundary, memory
- `MEMORY_SHUFFLED`: adaptive boundary with episode-memory correspondence destroyed
- `BOUNDARY_CLAMP`: ABEM memory with the boundary clamped to the fixed baseline depth

## Primary metric

For each episode:

`E = normalized_regret + cost_weight * normalized_cost`

Lower is better. The pre-registered support threshold is:

`mean(E_ABEM) / mean(E_BASE) <= 0.90`

with a paired bootstrap 95% upper confidence bound below `1.0`.

## Verdicts

- `PASS_COMBINED`
- `PASS_BOUNDARY_ONLY`
- `PASS_MEMORY_ONLY`
- `NO_GO`

Mechanistic attribution additionally depends on degradation under `MEMORY_SHUFFLED` and `BOUNDARY_CLAMP`.

## Phase structure

- **0A Smoke**: implementation/reproducibility checks only
- **0B Pilot**: calibrate and then freeze parameters
- **0C Confirmatory**: 30 fresh paired seeds, no OOD tuning

## Quick start

```bash
python -m pip install -e .[dev]
pytest -q
python -m abem.experiment --config configs/smoke.yaml
```

## Claim firewall

A successful Phase 0 supports only the statement that adaptive stopping and/or history bias improved search efficiency on the specified synthetic landscapes. It does **not** establish quantum decoherence control, consciousness, creativity, or autonomous breakthrough generation.
