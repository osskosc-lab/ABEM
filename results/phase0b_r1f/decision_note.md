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

## F1–F10反証診断

- F1 Time-only: validation oracle-relative error `-0.008634`でFULL_ABより低く、FULL_ABは経過時間baselineを超えない。
- F2 Stagnation-only: `-0.000984`でFULL_ABより高い。単純patienceへの完全縮約は、この比較だけからは成立しない。
- F3 Random matched: `-0.009641`でFULL_ABより低く、state-dependent停止の追加価値はvalidationで支持されない。
- F4 Temporal shuffle: shuffleによる増分 `0.034341`。時間対応には情報がある。
- F5 Feature deletion: uncertaintyとdiversityの削除は悪化を生まず、必要機構として主張しない。stagnation削除は大幅に悪化した。
- F6 Sign test: `{"gain": true, "score_std": false, "diversity": true, "stagnation": true}`。`score_std`はSIGNAL_MISSPECIFIED。
- F7 Equal budget: FULL_AB − fixedのnormalized regret差 `0.002786`で、同計算量のfixed searchより品質が悪い。
- F8 Difficulty adaptation: EASY→RUGGEDで停止時刻が増えず、事前期待に反する。
- F9 Difficulty-blind permutation: blind policyのvalidation oracle-relative error `-0.006032`はFULL_ABを悪化させず、context依存性を支持しない。
- F10 Metric robustness: `{"0.1": 0.00407215310903869, "0.5": 0.00407215310903869}`。方向反転はなく`METRIC_FRAGILE`ではない。

## Difficulty adaptation

- intact: `{"EASY": 29.229166666666668, "MEDIUM": 30.0, "RUGGED": 28.5625}`
- difficulty-blind: `{"EASY": 30.0, "MEDIUM": 28.5625, "RUGGED": 29.229166666666668}`

## 反証された主張

失敗Gate: `G2_SIMPLE_BASELINE, G4_DIFFICULTY_ADAPTATION, G5_VALIDATION`。Gateを通らない機構主張は採用しない。

## 残る最小主張

checkpoint信号はOracle STOP/CONTINUEをある程度予測し、時間shuffleで性能が崩れるため、探索履歴に予測情報が存在することまでは残る。ただしその情報を現在のhazard式で停止利得へ変換できず、複雑なAdaptive Boundaryの必要性は支持されない。

## Claim Firewall

本Phaseが扱うのはsynthetic search上のadaptive stoppingによるfuture Value of Computation識別だけであり、量子的機構、意識、創造性、ブレイクスルー生成を示さない。

## 次段階

`BOUNDARY_MECHANISM_SUPPORTED`の場合のみPhase 0B-r2 Memory Falsificationへ進む。それ以外は判定ラベルに従い縮約またはNO_GOとし、Phase 0Cへ進めない。
