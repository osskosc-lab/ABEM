# ABEM Phase 0B-r2G Decision Note

## 総合判定

**NO_VALID_DIFFICULTY_FAMILY**

## PR #3から引き継いだnegative result

PR #3の`DIFFICULTY_MANIPULATION_FAIL`を保持した。MVOC-Bは未評価であり、r2V artifactsは変更していない。

## 今回の修正理由

固定探索kernelに対するdifficulty gradientをcontrollerと分離して同定する。固定budgetのため主指標は`1 - F*_32`とした。

## Family A結果

{
  "all_level_statistics": {
    "levels": [
      "A1",
      "A2",
      "A3",
      "A4"
    ],
    "mean_terminal_regret": {
      "A1": 0.09043945312499999,
      "A2": 0.08848632812499997,
      "A3": 0.08318359375000003,
      "A4": 0.06287109374999995
    },
    "median_terminal_regret": {
      "A1": 0.09042968749999998,
      "A2": 0.08779296874999998,
      "A3": 0.08833007812500003,
      "A4": 0.06538085937499996
    },
    "adjacent_mean_differences": [
      -0.001953125000000014,
      -0.005302734374999937,
      -0.02031250000000008
    ],
    "monotonic_increase": false,
    "hardest_minus_easiest": {
      "mean": -0.02756835937500004,
      "median": -0.024755859375000036,
      "ci_lower": -0.03859545898437504,
      "ci_upper": -0.016523437500000036,
      "negative_seeds": 9,
      "total_seeds": 10
    },
    "direction_consistent_seed_count": 1,
    "direction_consistent_seed_fraction": 0.1,
    "development_gate_pass": false,
    "monotonicity_margin": -0.02031250000000008
  },
  "evaluated_level_windows": [
    {
      "levels": [
        "A1",
        "A2",
        "A3",
        "A4"
      ],
      "mean_terminal_regret": {
        "A1": 0.09043945312499999,
        "A2": 0.08848632812499997,
        "A3": 0.08318359375000003,
        "A4": 0.06287109374999995
      },
      "median_terminal_regret": {
        "A1": 0.09042968749999998,
        "A2": 0.08779296874999998,
        "A3": 0.08833007812500003,
        "A4": 0.06538085937499996
      },
      "adjacent_mean_differences": [
        -0.001953125000000014,
        -0.005302734374999937,
        -0.02031250000000008
      ],
      "monotonic_increase": false,
      "hardest_minus_easiest": {
        "mean": -0.02756835937500004,
        "median": -0.024755859375000036,
        "ci_lower": -0.03859545898437504,
        "ci_upper": -0.016523437500000036,
        "negative_seeds": 9,
        "total_seeds": 10
      },
      "direction_consistent_seed_count": 1,
      "direction_consistent_seed_fraction": 0.1,
      "development_gate_pass": false,
      "monotonicity_margin": -0.02031250000000008
    },
    {
      "levels": [
        "A1",
        "A2",
        "A3"
      ],
      "mean_terminal_regret": {
        "A1": 0.09043945312499999,
        "A2": 0.08848632812499997,
        "A3": 0.08318359375000003
      },
      "median_terminal_regret": {
        "A1": 0.09042968749999998,
        "A2": 0.08779296874999998,
        "A3": 0.08833007812500003
      },
      "adjacent_mean_differences": [
        -0.001953125000000014,
        -0.005302734374999937
      ],
      "monotonic_increase": false,
      "hardest_minus_easiest": {
        "mean": -0.0072558593749999555,
        "median": -0.0051269531249999445,
        "ci_lower": -0.018896728515624957,
        "ci_upper": 0.0040246582031250285,
        "negative_seeds": 6,
        "total_seeds": 10
      },
      "direction_consistent_seed_count": 4,
      "direction_consistent_seed_fraction": 0.4,
      "development_gate_pass": false,
      "monotonicity_margin": -0.005302734374999937
    },
    {
      "levels": [
        "A2",
        "A3",
        "A4"
      ],
      "mean_terminal_regret": {
        "A2": 0.08848632812499997,
        "A3": 0.08318359375000003,
        "A4": 0.06287109374999995
      },
      "median_terminal_regret": {
        "A2": 0.08779296874999998,
        "A3": 0.08833007812500003,
        "A4": 0.06538085937499996
      },
      "adjacent_mean_differences": [
        -0.005302734374999937,
        -0.02031250000000008
      ],
      "monotonic_increase": false,
      "hardest_minus_easiest": {
        "mean": -0.025615234375000028,
        "median": -0.02329101562500005,
        "ci_lower": -0.033916259765625026,
        "ci_upper": -0.018701171875000026,
        "negative_seeds": 10,
        "total_seeds": 10
      },
      "direction_consistent_seed_count": 0,
      "direction_consistent_seed_fraction": 0.0,
      "development_gate_pass": false,
      "monotonicity_margin": -0.02031250000000008
    }
  ],
  "selected_development_window": null,
  "shuffled_level_label_monotonic": false
}

## Family B結果

{
  "all_level_statistics": {
    "levels": [
      "B1",
      "B2",
      "B3",
      "B4"
    ],
    "mean_terminal_regret": {
      "B1": 0.08588671875000004,
      "B2": 0.09468559451219508,
      "B3": 0.0908403201219512,
      "B4": 0.0866150067750677
    },
    "median_terminal_regret": {
      "B1": 0.08433593750000004,
      "B2": 0.09306402439024386,
      "B3": 0.09058689024390242,
      "B4": 0.09092564363143624
    },
    "adjacent_mean_differences": [
      0.00879887576219504,
      -0.0038452743902438746,
      -0.0042253133468835025
    ],
    "monotonic_increase": false,
    "hardest_minus_easiest": {
      "mean": 0.0007282880250676568,
      "median": 0.002786008426490412,
      "ci_lower": -0.007878721788194534,
      "ci_upper": 0.0085581684557079,
      "negative_seeds": 5,
      "total_seeds": 10
    },
    "direction_consistent_seed_count": 5,
    "direction_consistent_seed_fraction": 0.5,
    "development_gate_pass": false,
    "monotonicity_margin": -0.0042253133468835025
  },
  "evaluated_level_windows": [
    {
      "levels": [
        "B1",
        "B2",
        "B3",
        "B4"
      ],
      "mean_terminal_regret": {
        "B1": 0.08588671875000004,
        "B2": 0.09468559451219508,
        "B3": 0.0908403201219512,
        "B4": 0.0866150067750677
      },
      "median_terminal_regret": {
        "B1": 0.08433593750000004,
        "B2": 0.09306402439024386,
        "B3": 0.09058689024390242,
        "B4": 0.09092564363143624
      },
      "adjacent_mean_differences": [
        0.00879887576219504,
        -0.0038452743902438746,
        -0.0042253133468835025
      ],
      "monotonic_increase": false,
      "hardest_minus_easiest": {
        "mean": 0.0007282880250676568,
        "median": 0.002786008426490412,
        "ci_lower": -0.007878721788194534,
        "ci_upper": 0.0085581684557079,
        "negative_seeds": 5,
        "total_seeds": 10
      },
      "direction_consistent_seed_count": 5,
      "direction_consistent_seed_fraction": 0.5,
      "development_gate_pass": false,
      "monotonicity_margin": -0.0042253133468835025
    },
    {
      "levels": [
        "B1",
        "B2",
        "B3"
      ],
      "mean_terminal_regret": {
        "B1": 0.08588671875000004,
        "B2": 0.09468559451219508,
        "B3": 0.0908403201219512
      },
      "median_terminal_regret": {
        "B1": 0.08433593750000004,
        "B2": 0.09306402439024386,
        "B3": 0.09058689024390242
      },
      "adjacent_mean_differences": [
        0.00879887576219504,
        -0.0038452743902438746
      ],
      "monotonic_increase": false,
      "hardest_minus_easiest": {
        "mean": 0.004953601371951169,
        "median": 0.002788919588414575,
        "ci_lower": -0.0020226181402439427,
        "ci_upper": 0.01248348418445116,
        "negative_seeds": 2,
        "total_seeds": 10
      },
      "direction_consistent_seed_count": 8,
      "direction_consistent_seed_fraction": 0.8,
      "development_gate_pass": false,
      "monotonicity_margin": -0.0038452743902438746
    },
    {
      "levels": [
        "B2",
        "B3",
        "B4"
      ],
      "mean_terminal_regret": {
        "B2": 0.09468559451219508,
        "B3": 0.0908403201219512,
        "B4": 0.0866150067750677
      },
      "median_terminal_regret": {
        "B2": 0.09306402439024386,
        "B3": 0.09058689024390242,
        "B4": 0.09092564363143624
      },
      "adjacent_mean_differences": [
        -0.0038452743902438746,
        -0.0042253133468835025
      ],
      "monotonic_increase": false,
      "hardest_minus_easiest": {
        "mean": -0.008070587737127388,
        "median": -0.008757833672086733,
        "ci_lower": -0.015004462017276454,
        "ci_upper": -0.000972539803523063,
        "negative_seeds": 7,
        "total_seeds": 10
      },
      "direction_consistent_seed_count": 3,
      "direction_consistent_seed_fraction": 0.3,
      "development_gate_pass": false,
      "monotonicity_margin": -0.0042253133468835025
    }
  ],
  "selected_development_window": null,
  "shuffled_level_label_monotonic": false
}

## Family C結果

{
  "all_level_statistics": {
    "levels": [
      "C1",
      "C2",
      "C3",
      "C4"
    ],
    "mean_terminal_regret": {
      "C1": 0.08960606060606062,
      "C2": 0.08418893129770996,
      "C3": 0.09017884615384619,
      "C4": 0.0881291989664083
    },
    "median_terminal_regret": {
      "C1": 0.08915404040404043,
      "C2": 0.08480279898218833,
      "C3": 0.08990384615384622,
      "C4": 0.09026485788113697
    },
    "adjacent_mean_differences": [
      -0.005417129308350657,
      0.00598991485613623,
      -0.0020496471874378863
    ],
    "monotonic_increase": false,
    "hardest_minus_easiest": {
      "mean": -0.0014768616396523214,
      "median": 0.0028224835564951978,
      "ci_lower": -0.01023709148167723,
      "ci_upper": 0.0072458028688043305,
      "negative_seeds": 4,
      "total_seeds": 10
    },
    "direction_consistent_seed_count": 6,
    "direction_consistent_seed_fraction": 0.6,
    "development_gate_pass": false,
    "monotonicity_margin": -0.005417129308350657
  },
  "evaluated_level_windows": [
    {
      "levels": [
        "C1",
        "C2",
        "C3",
        "C4"
      ],
      "mean_terminal_regret": {
        "C1": 0.08960606060606062,
        "C2": 0.08418893129770996,
        "C3": 0.09017884615384619,
        "C4": 0.0881291989664083
      },
      "median_terminal_regret": {
        "C1": 0.08915404040404043,
        "C2": 0.08480279898218833,
        "C3": 0.08990384615384622,
        "C4": 0.09026485788113697
      },
      "adjacent_mean_differences": [
        -0.005417129308350657,
        0.00598991485613623,
        -0.0020496471874378863
      ],
      "monotonic_increase": false,
      "hardest_minus_easiest": {
        "mean": -0.0014768616396523214,
        "median": 0.0028224835564951978,
        "ci_lower": -0.01023709148167723,
        "ci_upper": 0.0072458028688043305,
        "negative_seeds": 4,
        "total_seeds": 10
      },
      "direction_consistent_seed_count": 6,
      "direction_consistent_seed_fraction": 0.6,
      "development_gate_pass": false,
      "monotonicity_margin": -0.005417129308350657
    },
    {
      "levels": [
        "C1",
        "C2",
        "C3"
      ],
      "mean_terminal_regret": {
        "C1": 0.08960606060606062,
        "C2": 0.08418893129770996,
        "C3": 0.09017884615384619
      },
      "median_terminal_regret": {
        "C1": 0.08915404040404043,
        "C2": 0.08480279898218833,
        "C3": 0.08990384615384622
      },
      "adjacent_mean_differences": [
        -0.005417129308350657,
        0.00598991485613623
      ],
      "monotonic_increase": false,
      "hardest_minus_easiest": {
        "mean": 0.0005727855477855662,
        "median": -0.0018881118881119013,
        "ci_lower": -0.00594579181235429,
        "ci_upper": 0.007866393987956482,
        "negative_seeds": 6,
        "total_seeds": 10
      },
      "direction_consistent_seed_count": 4,
      "direction_consistent_seed_fraction": 0.4,
      "development_gate_pass": false,
      "monotonicity_margin": -0.005417129308350657
    },
    {
      "levels": [
        "C2",
        "C3",
        "C4"
      ],
      "mean_terminal_regret": {
        "C2": 0.08418893129770996,
        "C3": 0.09017884615384619,
        "C4": 0.0881291989664083
      },
      "median_terminal_regret": {
        "C2": 0.08480279898218833,
        "C3": 0.08990384615384622,
        "C4": 0.09026485788113697
      },
      "adjacent_mean_differences": [
        0.00598991485613623,
        -0.0020496471874378863
      ],
      "monotonic_increase": false,
      "hardest_minus_easiest": {
        "mean": 0.003940267668698348,
        "median": 0.006209810245182162,
        "ci_lower": -0.005209191742114912,
        "ci_upper": 0.011943862555969785,
        "negative_seeds": 2,
        "total_seeds": 10
      },
      "direction_consistent_seed_count": 8,
      "direction_consistent_seed_fraction": 0.8,
      "development_gate_pass": false,
      "monotonicity_margin": -0.0020496471874378863
    }
  ],
  "selected_development_window": null,
  "shuffled_level_label_monotonic": true
}

## 採用または不採用理由

{
  "selected_family": null,
  "decision": "No family was adopted: every full or contiguous three-plus-level window failed the monotonic paired development gate."
}

## Freeze時点

`NOT_EVALUATED`。G1失敗のためfamily/levelsはfreezeしていない。

## Blind replication

null

## Difficulty labelsの可否

{}

## 失敗Gate

`G1_FAMILY_SCREEN`

## 残る最小主張

development seedsで単調かつpaired CIを通るgenerator familyを同定できなかった。

## MVOC-B実験へ進めるか

この実行内ではMVOC-Bを評価していない。`DIFFICULTY_FAMILY_VALIDATED`の場合のみ、別の事前登録Phaseを設計する資格が得られる。

## Claim Firewall

本PhaseはABEM固定探索kernelに対するsynthetic difficulty gradientだけを評価する。Adaptive Boundary、MVOC-B、Memory、量子機構、AI意識・創造性の有効性を示さない。
