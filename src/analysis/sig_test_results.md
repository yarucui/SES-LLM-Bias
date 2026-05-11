# Significance Test Results

Random seed: 42 | Bootstrap iterations: 10,000 | FDR correction: Benjamini-Hochberg

## Direction 1 — Verbalized Sampling (paired bootstrap, n_boot=10 000)

| Comparison | n_pairs | mean_diff | 95% CI | p_raw | p_fdr | sig |
|---|---|---|---|---|---|---|
| vs vs baseline | 525 | -0.2352 | [-0.2502, -0.2200] | <0.001 | <0.001 | *** |
| dv vs baseline | 526 | -0.2122 | [-0.2276, -0.1966] | <0.001 | <0.001 | *** |
| hight vs baseline | 526 | -0.0157 | [-0.0263, -0.0051] | 0.003 | 0.004 | ** |
| vs vs hight | 524 | -0.2197 | [-0.2344, -0.2050] | <0.001 | <0.001 | *** |
| dv vs hight | 525 | -0.1963 | [-0.2119, -0.1806] | <0.001 | <0.001 | *** |
| vs vs dv | 524 | -0.0227 | [-0.0329, -0.0127] | <0.001 | <0.001 | *** |

## Direction 2 — Oracle vs Advisor Framing (paired bootstrap, n_boot=10 000)

| Comparison | n_pairs | mean_diff | 95% CI | p_raw | p_fdr | sig |
|---|---|---|---|---|---|---|
| oracle vs advisor | 2470 | -0.0151 | [-0.0253, -0.0045] | 0.005 | 0.005 | ** |
| oracle vs personal | 2444 | -0.0259 | [-0.0358, -0.0158] | <0.001 | <0.001 | *** |
| advisor vs personal | 2442 | -0.0116 | [-0.0196, -0.0037] | 0.004 | 0.005 | ** |

## Direction 2 — Cross-domain (Mann-Whitney U)

| Comparison | n_pairs | mean_diff | 95% CI | p_raw | p_fdr | sig |
|---|---|---|---|---|---|---|
| health vs other domains | 48 vs 7347 | -0.1820 | — | 0.005 | 0.005 | ** |

## Direction 2 — Cross-trade-off (Mann-Whitney U)

| Comparison | n_pairs | mean_diff | 95% CI | p_raw | p_fdr | sig |
|---|---|---|---|---|---|---|
| short_term_vs_long_term vs aggressive_vs_conservative | 120 vs 60 | -0.5400 | — | <0.001 | <0.001 | *** |

## Summary

All Direction 1 verbalized-sampling conditions show significantly lower abs_gap than baseline (FDR-corrected), confirming the headline finding that entropy collapse in step 5 is decoding-driven: verbalized sampling (VS) reduces abs_gap by 0.2352 on average.
High-temperature forced choice also reduces abs_gap vs baseline (mean_diff = -0.0157, p_fdr 0.004), indicating temperature is a secondary driver alongside output format.
For Direction 2, oracle framing significantly differs from advisor framing (mean_diff = -0.0151, p_fdr 0.005), and oracle vs personal is likewise significant (p_fdr <0.001).
The health-domain posts differ significantly from other domains in abs_gap (Mann-Whitney p_fdr 0.005), and the short_term_vs_long_term trade-off type differs significantly from aggressive_vs_conservative (p_fdr <0.001).
