# ABEM Phase 0B-r2V preregistration

## Purpose

Test a reduced stopping controller after the negative Phase 0B-r1F result. This is a new model, not a rescue tuning pass.

## Minimal controller

Only two signals are admissible:

- stagnation since the last best-score improvement
- recent gain represented by an EMA

The planned boundary is

`B_t = P0 + DeltaP * Gbar_t / (Gbar_t + G0)`

with `alpha = 0.5` frozen before simulation.

Removed from the primary controller: `score_std`, diversity, memory, and cumulative hazard.

## Stage isolation

- D0 difficulty validation: seeds 200-205
- D1 calibration: seeds 210-217
- D2 blind validation: seeds 220-231
- Phase 0B-r1F seeds 100-109: forbidden for model selection
- Phase 0C seeds 1000-1029: forbidden during r2V

The statistical unit remains the seed.

## D0 gate

Generator candidates are not called EASY/MEDIUM/HARD until fixed-depth search empirically establishes the preregistered difficulty ordering. If D0 fails, adaptive-boundary simulation stops without interpreting a difficulty effect.

## Freeze order

1. Run D0 only.
2. If D0 passes, calibrate fixed `P0` on D1.
3. Compute `G0` using the preregistered median-positive-EMA rule on D1.
4. Select one `DeltaP` from `{4, 8, 16}` on D1 only.
5. Save frozen config plus generating Git SHA.
6. Open D2 once.
7. No post-D2 retuning.

## Primary metric

`E = (1 - F*_tau) + 0.25 * tau / T_max`

Primary paired effect:

`DeltaE = E_ADAPTIVE - E_PATIENCE_ONLY`

Support requires `mean(DeltaE) < 0` and upper 95% paired seed-bootstrap CI `< 0` on D2.

## Main falsification

Gain-time alignment is destroyed while preserving the gain marginal distribution. If the apparent benefit survives gain shuffle, the gain-conditioned mechanism is not supported.

## Claim firewall

This experiment concerns stopping control in synthetic search only. It does not test quantum decoherence, consciousness, creativity, or breakthrough generation.
