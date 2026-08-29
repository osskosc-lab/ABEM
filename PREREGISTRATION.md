# ABEM Phase 0-r0 Preregistration

## Primary hypothesis

A history-dependent adaptive stopping boundary improves OOD search efficiency relative to a frozen fixed-depth baseline.

## Immutable semantic constraint

The task objective / landscape score is never changed by the agent. ABEM may adapt only the search process and stopping boundary.

## Primary metric

For each episode:

`E = normalized_regret + 0.25 * normalized_cost`

with `normalized_regret = 1 - best_score` and `normalized_cost = steps / T_max`.

The confirmatory unit of analysis is the **seed-level mean** across episodes.

## Primary support gate

Let `r = mean(E_ABEM) / mean(E_BASE)` over paired seed means.

Support requires both:

1. `r <= 0.90`
2. paired bootstrap 95% upper CI for `r` is `< 1.0`

## Conditions

- BASE: fixed depth, no memory
- AB: adaptive boundary, no memory
- MEM: fixed depth, online dissipative memory
- ABEM: adaptive boundary + online dissipative memory
- MEMORY_SHUFFLED: adaptive boundary using the same ABEM pre-episode memory-state collection with episode correspondence deranged
- BOUNDARY_CLAMP: fixed boundary using the correctly matched ABEM pre-episode memory state

## Mechanism gates

A mechanism is considered supported only if its matched ablation worsens mean efficiency error by at least the frozen `mechanism_margin`.

- Memory mechanism: `MEMORY_SHUFFLED >= ABEM + margin`
- Boundary mechanism: `BOUNDARY_CLAMP >= ABEM + margin`

## Phase sequence

### 0A Smoke

5 seeds. Implementation/reproducibility only. No scientific interpretation.

### 0B Pilot

10 seeds. Calibration is permitted only here. The fixed-depth BASE candidate set is `{8, 16, 32, 64}`. Search-boundary and memory hyperparameters may be adjusted during this phase, but every final value must be frozen before Phase 0C is opened.

### 0C Confirmatory

30 fresh seeds under a pre-specified OOD shift (stronger coupling and more interaction edges). No tuning after viewing confirmatory outcomes.

## Freeze procedure

Before running Phase 0C:

1. run all unit tests;
2. run `python -m abem.calibrate --config configs/pilot.yaml`;
3. choose and record the fixed-depth baseline using pilot results only;
4. freeze all `search`, `memory`, and `metric` fields in `configs/confirmatory.yaml`;
5. commit the frozen configuration before executing Phase 0C;
6. do not edit the confirmatory configuration in response to Phase 0C outcomes.

## Verdict labels

- PASS_COMBINED
- PASS_BOUNDARY_ONLY
- PASS_MEMORY_ONLY
- NO_GO

Smoke and Pilot always emit `NON_CONFIRMATORY_DO_NOT_INTERPRET` regardless of apparent performance.

## Claim firewall

A positive Phase 0 result supports only an algorithmic statement about synthetic search efficiency and matched mechanism controls. It does not establish physical decoherence control, quantum advantage, consciousness, creativity, intention, or autonomous scientific breakthrough generation.
