# ABEM Phase 0B-r1F Decision Note

## 総合判定

**BOUNDARY_OVERFIT**

## 中心命題

synthetic search環境で、現在の探索状態に依存するAdaptive Boundaryが、固定時間・patience・matched random停止より局所Value-of-Computation Oracleに近い停止判断を行えるかを検証した。

## 凍結した比較条件

- 固定深度: `32`
- patience: `8`
- FULL_AB候補: `CONSERVATIVE`
- Calibration上の最強simple baseline: `RANDOM_MATCHED`

## 主要結果

- Validation ΔR (FULL_AB - best simple): `0.004072`
- paired seed bootstrap 95% CI: `[-0.000826, 0.009119]`
- Validation AUROC / AUPRC / Brier: `0.7850` / `0.5229` / `0.1159`
- FULL_AB validation oracle regret: `-0.005568`
- Signal-shuffled validation oracle regret: `0.028772`

ここでのoracle regretは、局所rollout Oracleが選んだ停止時刻に対する**符号付き相対誤差**である。絶対最適Oracleではないため負値を取り得る。policy間のpaired差を主判定に用いる。

## 反証Gate

- G0_REPLAY: PASS — runtime checkpoint replay matched population, scores, best score, and cumulative hazard
- G1_PREDICTIVE_INFORMATION: PASS — validation AUROC=0.7850; threshold > 0.60
- G2_SIMPLE_BASELINE: FAIL — FULL_AB-RANDOM_MATCHED mean=0.004072, 95% CI upper=0.009119
- G3_TEMPORAL_MECHANISM: PASS — shuffle-full oracle regret=0.034341
- G4_DIFFICULTY_ADAPTATION: FAIL — rugged-easy gap intact=-0.6667, blind=-0.7708
- G5_VALIDATION: FAIL — calibration direction=-0.001118, validation direction=0.004072

## Feature deletion

{
  "MINUS_GAIN": 0.000454487407382664,
  "MINUS_UNCERTAINTY": -0.000894695230462479,
  "MINUS_DIVERSITY": -0.000723054584353307,
  "MINUS_STAGNATION": 0.02757869448983368
}

`score_std`はfuture valueとの想定符号に反し、uncertainty削除はFULL_ABを悪化させなかった。したがってepistemic uncertainty機構としては支持しない。

## Difficulty adaptation

- intact: `{"EASY": 29.229166666666668, "MEDIUM": 30.0, "RUGGED": 28.5625}`
- difficulty-blind: `{"EASY": 30.0, "MEDIUM": 28.5625, "RUGGED": 29.229166666666668}`

## 反証された主張

失敗Gate: `G2_SIMPLE_BASELINE, G4_DIFFICULTY_ADAPTATION, G5_VALIDATION`。Gateを通らない機構主張は採用しない。

## 残る最小主張

判定ラベルと通過Gateが許す範囲に限定する。単純baselineを超えない場合、複雑なAdaptive Boundaryの必要性は主張しない。

## Claim Firewall

本Phaseが扱うのはsynthetic search上のadaptive stoppingによるfuture Value of Computation識別だけであり、量子的機構、意識、創造性、ブレイクスルー生成を示さない。

## 次段階

`BOUNDARY_MECHANISM_SUPPORTED`の場合のみPhase 0B-r2 Memory Falsificationへ進む。それ以外は判定ラベルに従い縮約またはNO_GOとし、Phase 0Cへ進めない。
