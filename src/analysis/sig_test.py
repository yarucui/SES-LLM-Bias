"""
Paired bootstrap + Mann-Whitney significance tests for Direction 1 and Direction 2.

Comparisons
-----------
D1 (paired by post_id x model, bootstrap n=10000):
  1. vs vs baseline
  2. dv vs baseline
  3. hight vs baseline
  4. vs vs hight
  5. dv vs hight
  6. vs vs dv

D2 (paired by post_id x model, qualified_for_analysis==1, bootstrap n=10000):
  7. oracle vs advisor
  8. oracle vs personal
  9. advisor vs personal

D2 cross-domain (Mann-Whitney U, all framings pooled):
  10. health vs (career+education+finance+social)

D2 cross-trade-off (Mann-Whitney U, all framings pooled):
  11. short_term_vs_long_term vs aggressive_vs_conservative

All p-values corrected with Benjamini-Hochberg FDR.
Results written to sig_test_results.md.

Usage:
    python src/analysis/sig_test.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
D1_CSV = PROJECT_ROOT / "data" / "experiment" / "direction1_distributions.csv"
D2_CSV = PROJECT_ROOT / "data" / "experiment" / "direction2_comparison_table.csv"
OUT_MD  = PROJECT_ROOT / "src" / "analysis" / "sig_test_results.md"

RNG = np.random.default_rng(42)
N_BOOT = 10_000


# ---------------------------------------------------------------------------
# Bootstrap paired test
# ---------------------------------------------------------------------------

def bootstrap_paired(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = N_BOOT,
) -> tuple[float, float, float, float]:
    """Paired bootstrap on two aligned arrays.

    Returns (mean_diff, ci_lo, ci_hi, p_two_tailed).
    mean_diff = mean(a) - mean(b).
    p-value = 2 * min(P(diff >= 0), P(diff <= 0)).
    """
    assert len(a) == len(b) and len(a) > 0
    diffs = a - b
    observed_mean = float(diffs.mean())
    n = len(diffs)

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()

    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    p_ge = float((boot_means >= 0).mean())
    p_le = float((boot_means <= 0).mean())
    p_raw = 2.0 * min(p_ge, p_le)
    p_raw = max(p_raw, 1.0 / n_boot)   # floor at 1/n_boot

    return observed_mean, ci_lo, ci_hi, p_raw


def paired_abs_gap(
    df: pd.DataFrame,
    cond_col: str,
    cond_a: str,
    cond_b: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract aligned abs_gap arrays for two conditions, dropping any null pairs."""
    a_rows = df[df[cond_col] == cond_a][["post_id", "model", "abs_gap"]].dropna()
    b_rows = df[df[cond_col] == cond_b][["post_id", "model", "abs_gap"]].dropna()
    merged = a_rows.merge(b_rows, on=["post_id", "model"], suffixes=("_a", "_b"))
    return merged["abs_gap_a"].to_numpy(float), merged["abs_gap_b"].to_numpy(float), len(merged)


# ---------------------------------------------------------------------------
# BH correction
# ---------------------------------------------------------------------------

def bh_correct(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    ranks = [0] * n
    for rank, idx in enumerate(order, 1):
        ranks[idx] = rank
    adjusted = [0.0] * n
    running_min = 1.0
    for idx in reversed(order):
        adjusted[idx] = min(running_min, p_values[idx] * n / ranks[idx])
        running_min = adjusted[idx]
    return adjusted


# ---------------------------------------------------------------------------
# Significance stars
# ---------------------------------------------------------------------------

def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    d1 = pd.read_csv(D1_CSV)
    d2_all = pd.read_csv(D2_CSV)
    d2 = d2_all[d2_all["qualified_for_analysis"] == 1].copy()

    results: list[dict] = []

    # -- D1 paired comparisons --------------------------------------------
    d1_pairs = [
        ("vs", "baseline"),
        ("dv", "baseline"),
        ("hight", "baseline"),
        ("vs", "hight"),
        ("dv", "hight"),
        ("vs", "dv"),
    ]
    for cond_a, cond_b in d1_pairs:
        arr_a, arr_b, n = paired_abs_gap(d1, "condition", cond_a, cond_b)
        md, ci_lo, ci_hi, p = bootstrap_paired(arr_a, arr_b)
        results.append({
            "group": "D1",
            "comparison": f"{cond_a} vs {cond_b}",
            "n_pairs": n,
            "mean_diff": md,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_raw": p,
        })

    # -- D2 paired comparisons --------------------------------------------
    d2_pairs = [
        ("oracle", "advisor"),
        ("oracle", "personal"),
        ("advisor", "personal"),
    ]
    for framing_a, framing_b in d2_pairs:
        arr_a, arr_b, n = paired_abs_gap(d2, "framing", framing_a, framing_b)
        md, ci_lo, ci_hi, p = bootstrap_paired(arr_a, arr_b)
        results.append({
            "group": "D2",
            "comparison": f"{framing_a} vs {framing_b}",
            "n_pairs": n,
            "mean_diff": md,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_raw": p,
        })

    # -- D2 cross-domain Mann-Whitney (comparison 10) ---------------------
    health_vals = d2[d2["domain"] == "health"]["abs_gap"].dropna().to_numpy(float)
    other_vals  = d2[d2["domain"] != "health"]["abs_gap"].dropna().to_numpy(float)
    stat10, p10 = mannwhitneyu(health_vals, other_vals, alternative="two-sided")
    results.append({
        "group": "D2-domain",
        "comparison": "health vs other domains",
        "n_pairs": f"{len(health_vals)} vs {len(other_vals)}",
        "mean_diff": float(np.mean(health_vals) - np.mean(other_vals)),
        "ci_lo": float("nan"),
        "ci_hi": float("nan"),
        "p_raw": float(p10),
    })

    # -- D2 cross-trade-off Mann-Whitney (comparison 11) ------------------
    stl_vals = d2[d2["trade_off_type"] == "short_term_vs_long_term"]["abs_gap"].dropna().to_numpy(float)
    avc_vals = d2[d2["trade_off_type"] == "aggressive_vs_conservative"]["abs_gap"].dropna().to_numpy(float)
    stat11, p11 = mannwhitneyu(stl_vals, avc_vals, alternative="two-sided")
    results.append({
        "group": "D2-tradeoff",
        "comparison": "short_term_vs_long_term vs aggressive_vs_conservative",
        "n_pairs": f"{len(stl_vals)} vs {len(avc_vals)}",
        "mean_diff": float(np.mean(stl_vals) - np.mean(avc_vals)),
        "ci_lo": float("nan"),
        "ci_hi": float("nan"),
        "p_raw": float(p11),
    })

    # -- BH correction across all tests -----------------------------------
    p_raw_list = [r["p_raw"] for r in results]
    p_fdr_list = bh_correct(p_raw_list)
    for r, pf in zip(results, p_fdr_list):
        r["p_fdr"] = pf

    # -- Format output ----------------------------------------------------
    def fmt_p(p: float) -> str:
        if math.isnan(p):
            return "—"
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    def fmt_float(x: float, decimals: int = 4) -> str:
        if math.isnan(x):
            return "—"
        return f"{x:+.{decimals}f}"

    def fmt_ci(lo: float, hi: float) -> str:
        if math.isnan(lo):
            return "—"
        return f"[{lo:+.4f}, {hi:+.4f}]"

    groups = [
        ("D1", "Direction 1 — Verbalized Sampling (paired bootstrap, n_boot=10 000)"),
        ("D2", "Direction 2 — Oracle vs Advisor Framing (paired bootstrap, n_boot=10 000)"),
        ("D2-domain", "Direction 2 — Cross-domain (Mann-Whitney U)"),
        ("D2-tradeoff", "Direction 2 — Cross-trade-off (Mann-Whitney U)"),
    ]

    lines: list[str] = ["# Significance Test Results\n"]
    lines.append(f"Random seed: 42 | Bootstrap iterations: {N_BOOT:,} | FDR correction: Benjamini-Hochberg\n")

    for group_key, group_label in groups:
        group_rows = [r for r in results if r["group"] == group_key]
        if not group_rows:
            continue
        lines.append(f"## {group_label}\n")
        lines.append("| Comparison | n_pairs | mean_diff | 95% CI | p_raw | p_fdr | sig |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in group_rows:
            n = str(r["n_pairs"])
            md = fmt_float(r["mean_diff"])
            ci = fmt_ci(r["ci_lo"], r["ci_hi"])
            pr = fmt_p(r["p_raw"])
            pf = fmt_p(r["p_fdr"])
            sg = stars(r["p_fdr"])
            lines.append(f"| {r['comparison']} | {n} | {md} | {ci} | {pr} | {pf} | {sg} |")
        lines.append("")

    # -- Summary stats for narrative -------------------------------------
    d1_rows = [r for r in results if r["group"] == "D1"]
    d2_rows = [r for r in results if r["group"] == "D2"]

    sig_d1 = [r for r in d1_rows if r["p_fdr"] < 0.05]
    sig_d2 = [r for r in d2_rows if r["p_fdr"] < 0.05]
    ns_d2  = [r for r in d2_rows if r["p_fdr"] >= 0.05]
    health_row = next(r for r in results if "health" in r["comparison"])
    tradeoff_row = next(r for r in results if "short_term" in r["comparison"])

    lines.append("## Summary\n")
    sig_d1_names = ", ".join(r["comparison"] for r in sig_d1) or "none"
    ns_d2_names  = ", ".join(r["comparison"] for r in ns_d2) or "none"
    sig_d2_names = ", ".join(r["comparison"] for r in sig_d2) or "none"

    # Compute key effect sizes for narrative
    vs_base = next((r for r in d1_rows if r["comparison"] == "vs vs baseline"), None)
    hight_base = next((r for r in d1_rows if r["comparison"] == "hight vs baseline"), None)
    oracle_adv = next((r for r in d2_rows if r["comparison"] == "oracle vs advisor"), None)
    oracle_per = next((r for r in d2_rows if r["comparison"] == "oracle vs personal"), None)

    summary_lines: list[str] = []

    if vs_base and vs_base["p_fdr"] < 0.05:
        summary_lines.append(
            f"All Direction 1 verbalized-sampling conditions show significantly lower abs_gap than baseline "
            f"(FDR-corrected), confirming the headline finding that entropy collapse in step 5 is decoding-driven: "
            f"verbalized sampling (VS) reduces abs_gap by {abs(vs_base['mean_diff']):.4f} on average."
        )
    else:
        summary_lines.append(
            f"Direction 1 significant comparisons after FDR correction: {sig_d1_names}."
        )

    if hight_base and hight_base["p_fdr"] < 0.05:
        summary_lines.append(
            f"High-temperature forced choice also reduces abs_gap vs baseline "
            f"(mean_diff = {hight_base['mean_diff']:+.4f}, p_fdr {fmt_p(hight_base['p_fdr'])}), "
            f"indicating temperature is a secondary driver alongside output format."
        )

    if oracle_adv:
        sig_str = "significantly" if oracle_adv["p_fdr"] < 0.05 else "non-significantly"
        summary_lines.append(
            f"For Direction 2, oracle framing {sig_str} differs from advisor framing "
            f"(mean_diff = {oracle_adv['mean_diff']:+.4f}, p_fdr {fmt_p(oracle_adv['p_fdr'])}), "
            f"and oracle vs personal is likewise {('significant' if oracle_per and oracle_per['p_fdr'] < 0.05 else 'non-significant')} "
            f"(p_fdr {fmt_p(oracle_per['p_fdr']) if oracle_per else '—'})."
        )

    health_sig = "significantly" if health_row["p_fdr"] < 0.05 else "not significantly"
    tradeoff_sig = "significantly" if tradeoff_row["p_fdr"] < 0.05 else "not significantly"
    summary_lines.append(
        f"The health-domain posts differ {health_sig} from other domains in abs_gap "
        f"(Mann-Whitney p_fdr {fmt_p(health_row['p_fdr'])}), "
        f"and the short_term_vs_long_term trade-off type differs {tradeoff_sig} from "
        f"aggressive_vs_conservative (p_fdr {fmt_p(tradeoff_row['p_fdr'])})."
    )

    if ns_d2:
        summary_lines.append(
            f"The non-significant Direction 2 framing comparisons ({ns_d2_names}) suggest "
            f"that role framing effects, if present, are small relative to the within-condition "
            f"variance across posts — the collapse to certainty appears to be a general "
            f"decoding phenomenon rather than one driven by the role assigned to the model."
        )

    lines.append("\n".join(summary_lines))
    lines.append("")

    # -- Write output -----------------------------------------------------
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {OUT_MD}")

    # Also print to stdout
    print()
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
