# Phase 0B-r2V experiment ground

Status: **PREPARED / NOT EXECUTED**

This directory is reserved for the next ABEM experiment. No scientific result has been generated here yet.

## Stage lock

- D0 difficulty validation: allowed
- D1 calibration: locked until D0 passes
- D2 blind validation: locked until D1 parameters are frozen
- Phase 0C confirmatory: forbidden

## Required freeze order

1. Validate the D0 generator ordering with fixed-depth search only.
2. If and only if D0 passes, calibrate `P0`, derive `G0`, and choose one `DeltaP` on D1.
3. Write a frozen config and the generating Git SHA before opening D2.
4. Run D2 once for scientific evaluation. Do not retune on D2.
5. Do not access Phase 0C seeds during r2V.

## Inherited negative evidence

Phase 0B-r1F is retained as a negative result. r2V must not restore `score_std`, diversity, memory, or cumulative-hazard tuning to rescue the previous controller.
