"""
Interim *exploratory* statistical analysis -- Llama + GPT only.

WARNING: the tests in this script are exploratory. They are run on partial
data (2 of 3 preregistered study models) to estimate effect sizes and
inform the decision on whether to run Claude Sonnet 4.6. The
preregistered confirmatory analysis is a separate run on the full
three-model dataset after data collection completes; only that run
produces confirmatory p-values for the thesis.

This script does NOT apply Benjamini-Hochberg FDR. The preregistration
reserves BH-FDR (q=0.05) for the seven primary hypotheses on the full
dataset. Correcting here would double-count.

Companion to src/analysis/interim_check.py (descriptive figures). This
script writes its own HTML and TXT reports and does not touch the
descriptive outputs.

Tests implemented:
    1. H1a  -- mean abs_gap > 0                  (1-sample t-test, pooled + per-model)
    2. H1b  -- entropy collapse on ambiguous     (one-sided paired diff, pooled + per-model)
    3/4/5. H2a/H2b/H2c -- moderator regression   (statsmodels mixedlm, random intercept on post_id)
    6. H3   -- cross-model abs_gap difference    (paired t-test, gpt vs llama)
    7. RQ-c -- cross-model agreement             (descriptive: % agreement + Pearson r)
    8. RQ3  -- stability decomposition           (one-way ANOVA of llm_entropy by consensus_level)

Outputs:
    data/analysis/interim_stats_report.html
    data/analysis/interim_stats_summary.txt
    logs/interim_stats.log

Usage:
    python src/analysis/interim_stats.py
    python src/analysis/interim_stats.py --skip-html
"""

from __future__ import annotations

import argparse
import html
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Paths + logging
# ---------------------------------------------------------------------------
ANALYSIS_DIR = config.DATA_DIR / "analysis"
REPORT_PATH = ANALYSIS_DIR / "interim_stats_report.html"
SUMMARY_PATH = ANALYSIS_DIR / "interim_stats_summary.txt"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "interim_stats.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("interim_stats")

COMPARISON_CSV = config.EXPERIMENT_DIR / "comparison_table.csv"
MODELS = ["gpt", "llama"]

STRONG_DISCLAIMER_HTML = """
<div class='disclaimer-strong' style='background: #ffd6d6; border: 2px solid #c00;
     padding: 1em; border-radius: 6px; margin: 1em 0;'>
<b>WARNING -- INTERIM EXPLORATORY ANALYSIS.</b><br>
This report contains formal statistical tests run on partial data
(2 of 3 preregistered models). These p-values are NOT confirmatory.
The preregistered confirmatory analysis will be run on the full
3-model dataset after data collection completes.<br>
Do not cite these results as final.
</div>
<p><b>No multiple-comparison correction is applied to interim tests.</b>
The preregistration commits to BH-FDR (q=0.05) on the seven primary
hypotheses run on the full 3-model dataset.</p>
"""

STRONG_DISCLAIMER_TXT = (
    "WARNING -- INTERIM EXPLORATORY ANALYSIS\n"
    "Tests below run on partial data (2 of 3 preregistered models).\n"
    "These p-values are NOT confirmatory. The preregistered confirmatory\n"
    "analysis runs on the full 3-model dataset after data collection.\n"
    "No multiple-comparison correction is applied here; the preregistration\n"
    "reserves BH-FDR (q=0.05) for the seven primary hypotheses on the full dataset.\n"
)

# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def cohens_d_label(d: float) -> str:
    if np.isnan(d):
        return "n/a"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def fmt_p(p: float) -> str:
    """Standard stats-report p-value formatting."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.3f}"


def fmt(x, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        if math.isnan(x):
            return "n/a"
    except (TypeError, ValueError):
        return str(x)
    return f"{x:.{digits}f}"


def cohens_d_ci(d: float, n: int, paired: bool = False,
                conf: float = 0.95) -> tuple[float, float]:
    """Approximate 95% CI for Cohen's d via the Hedges/Olkin asymptotic SE.

    Works for one-sample, paired, and within-subject designs when n is the
    effective sample size (pairs, not total observations).
    """
    if n <= 1 or np.isnan(d):
        return (float("nan"), float("nan"))
    se = math.sqrt(1.0 / n + (d ** 2) / (2 * n))
    z = stats.norm.ppf(0.5 + conf / 2)
    return (d - z * se, d + z * se)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    if not COMPARISON_CSV.exists():
        raise FileNotFoundError(
            f"comparison_table.csv not found at {COMPARISON_CSV}. Run step5 first."
        )
    df = pd.read_csv(COMPARISON_CSV)
    log.info("Loaded %d rows from %s", len(df), COMPARISON_CSV.name)

    df = df[df["qualified_for_analysis"] == 1].copy()
    log.info("After qualified_for_analysis==1 filter: %d rows", len(df))

    df = df[df["model"].isin(MODELS)].copy()
    log.info("After model filter (%s): %d rows", ", ".join(MODELS), len(df))

    if df.empty:
        raise RuntimeError("No rows remain after filtering. Run step5 first.")

    for model in MODELS:
        n = (df["model"] == model).sum()
        log.info("  %s: %d qualified rows", model, n)

    # Preregistration-update warnings the user asked for.
    log.warning(
        "Preregistration update check: the SES binary contrast "
        "(low vs non-low) used below must be filed as a preregistration "
        "update before Claude data is collected, if not already done."
    )
    log.warning(
        "Preregistration update check: trade_off_type is excluded from "
        "the moderator regression because 92%% of pilot posts fall into "
        "risk_vs_stability, making it near-constant. This exclusion should "
        "also be filed as a preregistration update if not already done."
    )
    return df


# ---------------------------------------------------------------------------
# Test 1 -- H1a: mean abs_gap > 0
# ---------------------------------------------------------------------------
def test_h1a(df: pd.DataFrame) -> dict:
    log.info("[Test 1 -- H1a] 1-sample t-test: abs_gap vs popmean=0")

    results = {"label": "H1a -- mean abs_gap > 0 (1-sample t-test vs 0)",
               "pooled": None, "per_model": {}}

    pooled = df["abs_gap"].dropna()
    if len(pooled) < 30:
        log.warning("[Test 1] pooled N=%d < 30; power limited", len(pooled))
    t_stat, p = stats.ttest_1samp(pooled, popmean=0)
    mean, sd = pooled.mean(), pooled.std(ddof=1)
    d = mean / sd if sd > 0 else float("nan")
    lo, hi = cohens_d_ci(d, len(pooled))
    results["pooled"] = {
        "n": int(len(pooled)), "mean": float(mean), "sd": float(sd),
        "t": float(t_stat), "df": int(len(pooled) - 1), "p": float(p),
        "d": float(d), "d_ci": (float(lo), float(hi)), "label": cohens_d_label(d),
    }
    log.info("  pooled: n=%d, mean=%.3f, t=%.2f, p=%s, d=%.2f (%s)",
             len(pooled), mean, t_stat, fmt_p(p), d, cohens_d_label(d))

    for m in MODELS:
        sub = df[df["model"] == m]["abs_gap"].dropna()
        if len(sub) < 30:
            log.warning("[Test 1 %s] N=%d < 30; power limited", m, len(sub))
        if len(sub) < 2:
            results["per_model"][m] = {"n": int(len(sub)), "error": "N<2"}
            continue
        t_stat, p = stats.ttest_1samp(sub, popmean=0)
        mean, sd = sub.mean(), sub.std(ddof=1)
        d = mean / sd if sd > 0 else float("nan")
        lo, hi = cohens_d_ci(d, len(sub))
        results["per_model"][m] = {
            "n": int(len(sub)), "mean": float(mean), "sd": float(sd),
            "t": float(t_stat), "df": int(len(sub) - 1), "p": float(p),
            "d": float(d), "d_ci": (float(lo), float(hi)),
            "label": cohens_d_label(d),
        }
        log.info("  %s: n=%d, mean=%.3f, t=%.2f, p=%s, d=%.2f (%s)",
                 m, len(sub), mean, t_stat, fmt_p(p), d, cohens_d_label(d))

    return results


# ---------------------------------------------------------------------------
# Test 2 -- H1b: entropy collapse on ambiguous stratum (one-sided paired)
# ---------------------------------------------------------------------------
def _one_sided_p(t_stat: float, p_two_sided: float, expect_positive: bool = True) -> float:
    """Convert two-sided p to one-sided given expected direction."""
    if expect_positive:
        return p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    return p_two_sided / 2 if t_stat < 0 else 1 - p_two_sided / 2


def _h1b_core(df_amb: pd.DataFrame, label: str) -> dict:
    diff = (df_amb["human_entropy"] - df_amb["llm_entropy"]).dropna()
    if len(diff) < 2:
        log.warning("[Test 2 %s] N=%d < 2; skipping", label, len(diff))
        return {"n": int(len(diff)), "error": "N<2"}
    if len(diff) < 30:
        log.warning("[Test 2 %s] N=%d < 30; power limited", label, len(diff))
    t_stat, p_two = stats.ttest_1samp(diff, popmean=0)
    p_one = _one_sided_p(t_stat, p_two, expect_positive=True)
    mean, sd = diff.mean(), diff.std(ddof=1)
    d = mean / sd if sd > 0 else float("nan")
    lo, hi = cohens_d_ci(d, len(diff))
    out = {"n": int(len(diff)), "mean_diff": float(mean), "sd_diff": float(sd),
           "t": float(t_stat), "df": int(len(diff) - 1),
           "p_two_sided": float(p_two), "p_one_sided": float(p_one),
           "d": float(d), "d_ci": (float(lo), float(hi)),
           "label": cohens_d_label(d)}
    log.info("  %s: n=%d, mean_diff=%.3f, t=%.2f, p_one=%s, d=%.2f (%s)",
             label, len(diff), mean, t_stat, fmt_p(p_one), d, cohens_d_label(d))
    return out


def test_h1b(df: pd.DataFrame) -> dict:
    log.info("[Test 2 -- H1b] Paired diff (human_entropy - llm_entropy) "
             "on ambiguous stratum; one-sided expecting positive")
    results = {"label": "H1b -- entropy collapse on ambiguous posts "
                        "(one-sided paired, H_human - H_llm > 0)",
               "pooled": None, "per_model": {}}

    amb = df[df["consensus_level"] == "ambiguous"]
    results["pooled"] = _h1b_core(amb, "pooled")
    for m in MODELS:
        sub = amb[amb["model"] == m]
        results["per_model"][m] = _h1b_core(sub, m)
    return results


# ---------------------------------------------------------------------------
# Tests 3/4/5 -- H2a/H2b/H2c: mixed-effects regression
# ---------------------------------------------------------------------------
def _mixedlm_summary(res) -> dict:
    """Extract coefficients + SE + p-values from a fitted mixedlm result."""
    params = res.params
    se = res.bse
    pvals = res.pvalues
    ci = res.conf_int()
    rows = []
    for name in params.index:
        row = {
            "term": name,
            "coef": float(params[name]),
            "se": float(se[name]),
            "p": float(pvals[name]) if not np.isnan(pvals[name]) else float("nan"),
            "ci_lo": float(ci.loc[name, 0]),
            "ci_hi": float(ci.loc[name, 1]),
        }
        rows.append(row)
    # Random-intercept variance (mixedlm exposes it via cov_re).
    try:
        var_re = float(res.cov_re.iloc[0, 0])
    except Exception:
        var_re = float("nan")
    resid_var = float(getattr(res, "scale", float("nan")))
    return {"terms": rows, "var_post_id": var_re, "resid_var": resid_var}


def _approx_marginal_r2(res) -> float:
    """Crude marginal R^2: fixed-effect variance / (fixed + random + residual)."""
    try:
        fitted = res.fittedvalues
        var_fixed = float(np.var(fitted, ddof=1))
        var_re = float(res.cov_re.iloc[0, 0])
        var_res = float(res.scale)
        denom = var_fixed + var_re + var_res
        return var_fixed / denom if denom > 0 else float("nan")
    except Exception:
        return float("nan")


def _fit_h2_regression(df: pd.DataFrame, ses_term: str) -> tuple[dict, str]:
    """Fit the H2 mixed-effects regression with a given SES parameterisation.

    ses_term is either 'C(ses_level, Treatment(reference="mid"))' or
    'ses_strain' (the binary contrast).
    """
    # statsmodels requires a clean frame (no NaN in model columns).
    needed = ["abs_gap", "reversibility", "time_horizon",
              "domain", "model", "post_id"]
    d = df.dropna(subset=needed).copy()
    if "ses_strain" in ses_term:
        d["ses_strain"] = (d["ses_level"] == "low").astype(int)
    else:
        d = d.dropna(subset=["ses_level"])
    log.info("  N=%d rows entering regression (%s)", len(d), ses_term)

    formula = (
        "abs_gap ~ C(reversibility, Treatment(reference='reversible')) "
        "+ C(time_horizon, Treatment(reference='short')) "
        f"+ {ses_term} "
        "+ C(domain, Treatment(reference='education')) "
        "+ C(model, Treatment(reference='gpt'))"
    )
    md = smf.mixedlm(formula, d, groups=d["post_id"])
    res = md.fit(reml=True, method="lbfgs")
    summary = _mixedlm_summary(res)
    summary["marginal_r2_approx"] = _approx_marginal_r2(res)
    summary["formula"] = formula
    summary["n"] = int(len(d))
    summary["converged"] = bool(res.converged)
    return summary, formula


def test_h2_abc(df: pd.DataFrame) -> dict:
    log.info("[Tests 3/4/5 -- H2a/H2b/H2c] Mixed-effects regression on abs_gap")
    log.info("  (trade_off_type excluded: 92%% single-category in pilot)")

    results = {"label": "H2 moderator regression (mixed-effects, random intercept on post_id)"}

    log.info("  [H2 full] SES as 3-level factor (low / mid / high)")
    full, full_formula = _fit_h2_regression(
        df, "C(ses_level, Treatment(reference='mid'))"
    )
    results["full"] = full

    log.info("  [H2c binary] SES collapsed to binary (low vs non-low)")
    binary, binary_formula = _fit_h2_regression(df, "ses_strain")
    results["binary"] = binary

    return results


# ---------------------------------------------------------------------------
# Test 6 -- H3: paired t-test abs_gap gpt vs llama
# ---------------------------------------------------------------------------
def test_h3(df: pd.DataFrame) -> dict:
    log.info("[Test 6 -- H3] Paired t-test: abs_gap gpt vs llama (by post_id)")
    pivot = df.pivot_table(
        index="post_id", columns="model", values="abs_gap", aggfunc="mean"
    ).dropna()
    results = {"label": "H3 -- cross-model abs_gap difference (paired t-test, gpt vs llama)",
               "n": int(len(pivot))}
    if not {"gpt", "llama"}.issubset(pivot.columns):
        log.warning("[Test 6] Need both gpt and llama; skipping")
        results["error"] = "both models required"
        return results
    if len(pivot) < 30:
        log.warning("[Test 6] n=%d paired posts < 30; power limited", len(pivot))
    if len(pivot) < 2:
        results["error"] = "N<2"
        return results

    diff = pivot["gpt"] - pivot["llama"]
    t_stat, p = stats.ttest_rel(pivot["gpt"], pivot["llama"])
    mean, sd = diff.mean(), diff.std(ddof=1)
    d = mean / sd if sd > 0 else float("nan")
    lo, hi = cohens_d_ci(d, len(diff))
    results.update({
        "mean_diff": float(mean), "sd_diff": float(sd),
        "t": float(t_stat), "df": int(len(diff) - 1),
        "p": float(p), "d": float(d), "d_ci": (float(lo), float(hi)),
        "label_size": cohens_d_label(d),
        "mean_gpt": float(pivot["gpt"].mean()),
        "mean_llama": float(pivot["llama"].mean()),
    })
    log.info("  n=%d, mean_diff=%.3f, t=%.2f, p=%s, d=%.2f (%s)",
             len(diff), mean, t_stat, fmt_p(p), d, cohens_d_label(d))
    return results


# ---------------------------------------------------------------------------
# Test 7 -- Cross-model agreement (descriptive)
# ---------------------------------------------------------------------------
def test_agreement(df: pd.DataFrame) -> dict:
    log.info("[Test 7] Cross-model agreement on llm_risky_rate (descriptive)")
    pivot = df.pivot_table(
        index="post_id", columns="model", values="llm_risky_rate", aggfunc="mean"
    ).dropna()
    results = {"label": "Cross-model agreement (descriptive)", "n": int(len(pivot))}
    if not {"gpt", "llama"}.issubset(pivot.columns) or len(pivot) < 2:
        results["error"] = "need both models and N>=2"
        return results

    both_above = int(((pivot["gpt"] > 0.5) & (pivot["llama"] > 0.5)).sum())
    both_below = int(((pivot["gpt"] <= 0.5) & (pivot["llama"] <= 0.5)).sum())
    agreement = (both_above + both_below) / len(pivot)
    r, r_p = stats.pearsonr(pivot["gpt"], pivot["llama"])
    results.update({
        "both_above": both_above, "both_below": both_below,
        "agreement": float(agreement),
        "pearson_r": float(r), "pearson_p": float(r_p),
    })
    log.info("  agreement=%.1f%% (n=%d), pearson r=%.3f, p=%s",
             agreement * 100, len(pivot), r, fmt_p(r_p))
    return results


# ---------------------------------------------------------------------------
# Test 8 -- Stability decomposition (one-way ANOVA on llm_entropy by consensus)
# ---------------------------------------------------------------------------
def test_stability(df: pd.DataFrame) -> dict:
    log.info("[Test 8] One-way ANOVA: llm_entropy by consensus_level (per model)")
    results = {"label": "RQ3 -- stability: does llm_entropy track consensus_level?",
               "per_model": {}}
    for m in MODELS:
        sub = df[df["model"] == m]
        groups = [sub[sub["consensus_level"] == c]["llm_entropy"].dropna().values
                  for c in ["high_safe", "ambiguous", "high_risky"]]
        sizes = [len(g) for g in groups]
        if min(sizes) < 2:
            log.warning("[Test 8 %s] one or more groups too small: sizes=%s", m, sizes)
            results["per_model"][m] = {"group_sizes": sizes, "error": "group too small"}
            continue
        f_stat, p = stats.f_oneway(*groups)
        means = {c: float(np.mean(g)) for c, g in
                 zip(["high_safe", "ambiguous", "high_risky"], groups)}
        # Partial eta-squared for one-way ANOVA = SS_between / SS_total.
        total = np.concatenate(groups)
        grand_mean = np.mean(total)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = np.sum((total - grand_mean) ** 2)
        eta2 = ss_between / ss_total if ss_total > 0 else float("nan")
        df_between = len(groups) - 1
        df_within = sum(sizes) - len(groups)
        results["per_model"][m] = {
            "group_sizes": sizes, "means": means,
            "F": float(f_stat), "df_between": df_between, "df_within": df_within,
            "p": float(p), "partial_eta2": float(eta2),
        }
        log.info("  %s: F(%d,%d)=%.2f, p=%s, eta^2=%.3f, means=%s",
                 m, df_between, df_within, f_stat, fmt_p(p), eta2, means)
    return results


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _html_esc(s) -> str:
    return html.escape(str(s))


def _row_t1(r: dict) -> str:
    if "error" in r:
        return f"<tr><td colspan='7'>skipped: {_html_esc(r['error'])}</td></tr>"
    lo, hi = r["d_ci"]
    return (
        f"<tr><td>{r['n']}</td><td>{fmt(r['mean'])}</td>"
        f"<td>{fmt(r['sd'])}</td><td>t({r['df']})={fmt(r['t'], 2)}</td>"
        f"<td>{fmt_p(r['p'])}</td><td>{fmt(r['d'], 2)} ({r['label']})</td>"
        f"<td>[{fmt(lo, 2)}, {fmt(hi, 2)}]</td></tr>"
    )


def _row_t2(r: dict) -> str:
    if "error" in r:
        return f"<tr><td colspan='7'>skipped: {_html_esc(r['error'])}</td></tr>"
    lo, hi = r["d_ci"]
    return (
        f"<tr><td>{r['n']}</td><td>{fmt(r['mean_diff'])}</td>"
        f"<td>t({r['df']})={fmt(r['t'], 2)}</td>"
        f"<td>two={fmt_p(r['p_two_sided'])}</td>"
        f"<td>one={fmt_p(r['p_one_sided'])}</td>"
        f"<td>{fmt(r['d'], 2)} ({r['label']})</td>"
        f"<td>[{fmt(lo, 2)}, {fmt(hi, 2)}]</td></tr>"
    )


def _regression_table(summary: dict) -> str:
    rows = []
    for t in summary["terms"]:
        rows.append(
            f"<tr><td style='text-align:left'><code>{_html_esc(t['term'])}</code></td>"
            f"<td>{fmt(t['coef'])}</td><td>{fmt(t['se'])}</td>"
            f"<td>[{fmt(t['ci_lo'])}, {fmt(t['ci_hi'])}]</td>"
            f"<td>{fmt_p(t['p'])}</td></tr>"
        )
    header = ("<table class='reg'><tr><th>term</th><th>coef</th><th>SE</th>"
              "<th>95% CI</th><th>p</th></tr>")
    footer = (
        f"</table><p class='small'>Random-intercept variance (post_id): "
        f"{fmt(summary['var_post_id'])}. Residual variance: "
        f"{fmt(summary['resid_var'])}. Approx. marginal R^2 (fixed-effect "
        f"variance / total variance): {fmt(summary.get('marginal_r2_approx'))}. "
        f"Converged: {summary.get('converged')}. N = {summary['n']}.</p>"
    )
    return header + "\n".join(rows) + footer


def build_html(results: dict) -> str:
    t1 = results["h1a"]
    t2 = results["h1b"]
    t3 = results["h2"]
    t6 = results["h3"]
    t7 = results["agreement"]
    t8 = results["stability"]

    def interp_h1a() -> str:
        p = t1["pooled"]
        if not p or "error" in p:
            return ""
        return (f"Pooled mean abs_gap is {fmt(p['mean'])} (d={fmt(p['d'], 2)}, "
                f"{p['label']}). Interpretation under the interim test: LLM "
                f"and human choice rates differ from one another by a non-"
                f"trivial amount on average.")

    def interp_h1b() -> str:
        p = t2["pooled"]
        if not p or "error" in p:
            return ""
        return (f"On ambiguous posts (n={p['n']} paired observations), the "
                f"human choice distribution is more spread out than the LLM's "
                f"by {fmt(p['mean_diff'])} nats on average (d={fmt(p['d'], 2)}, "
                f"{p['label']}). Interim support for H1b if one-sided p is small.")

    def interp_h3() -> str:
        if "error" in t6:
            return ""
        return (f"GPT abs_gap mean = {fmt(t6['mean_gpt'])}, Llama abs_gap "
                f"mean = {fmt(t6['mean_llama'])} (paired difference "
                f"{fmt(t6['mean_diff'])}, d={fmt(t6['d'], 2)}, "
                f"{t6['label_size']}). A small effect size here suggests "
                f"the two models diverge from humans in broadly similar "
                f"magnitudes, leaving model as a moderate contributor to "
                f"variance.")

    def interp_agreement() -> str:
        if "error" in t7:
            return ""
        return (f"GPT and Llama agree on the majority direction for "
                f"{t7['agreement'] * 100:.1f}% of {t7['n']} shared posts, "
                f"with Pearson r = {fmt(t7['pearson_r'], 3)}. If agreement "
                f"and correlation are already high, adding Claude will not "
                f"change the qualitative picture, though it is still required "
                f"for the preregistered confirmatory tests.")

    def interp_stability() -> str:
        lines = []
        for m in MODELS:
            r = t8["per_model"].get(m, {})
            if "error" in r:
                continue
            means = r["means"]
            lines.append(
                f"{m}: F({r['df_between']},{r['df_within']})="
                f"{fmt(r['F'], 2)}, p={fmt_p(r['p'])}, "
                f"eta^2={fmt(r['partial_eta2'], 3)}. "
                f"Means: high_safe={fmt(means['high_safe'])}, "
                f"ambiguous={fmt(means['ambiguous'])}, "
                f"high_risky={fmt(means['high_risky'])}."
            )
        return "<br>".join(lines)

    parts = [f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Interim stats -- Llama + GPT</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif;
          max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1 {{ border-bottom: 2px solid #888; padding-bottom: 0.3em; }}
  h2 {{ margin-top: 2em; color: #333; }}
  table {{ border-collapse: collapse; margin: 0.8em 0; }}
  table.reg {{ font-family: Menlo, Consolas, monospace; font-size: 0.9em; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
  th {{ background: #f4f4f4; text-align: center; }}
  .small {{ font-size: 0.85em; color: #555; }}
  .interp {{ margin: 0.5em 0 0.2em; font-style: italic; color: #333; }}
  code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
</style></head><body>
<h1>Interim statistical analysis -- Llama + GPT</h1>
{STRONG_DISCLAIMER_HTML}
<p><b>Preregistration notes (user to verify filed before Claude run):</b></p>
<ul>
  <li>SES binary contrast (low vs non-low) -- used in Test 5.</li>
  <li>Exclusion of <code>trade_off_type</code> from the H2 regression
      -- near-constant in pilot (~92% risk_vs_stability).</li>
</ul>

<h2>Test 1 -- H1a: mean abs_gap &gt; 0</h2>
<p class='small'>1-sample t-test, H0: mean(abs_gap) = 0. Two-sided.</p>
<table><tr><th>stratum</th><th>n</th><th>mean</th><th>sd</th>
<th>t</th><th>p</th><th>Cohen's d</th><th>95% CI d</th></tr>
<tr><th>pooled</th>{_row_t1(t1['pooled'])[4:]}
<tr><th>gpt</th>{_row_t1(t1['per_model']['gpt'])[4:]}
<tr><th>llama</th>{_row_t1(t1['per_model']['llama'])[4:]}
</table>
<p class='interp'>{interp_h1a()}</p>

<h2>Test 2 -- H1b: entropy collapse on ambiguous stratum</h2>
<p class='small'>Paired difference (human_entropy - llm_entropy) on
posts with consensus_level = 'ambiguous'. One-sided, expecting positive
(humans more uncertain than LLM).</p>
<table><tr><th>stratum</th><th>n</th><th>mean diff</th>
<th>t</th><th>p (two)</th><th>p (one)</th>
<th>Cohen's d</th><th>95% CI d</th></tr>
<tr><th>pooled</th>{_row_t2(t2['pooled'])[4:]}
<tr><th>gpt</th>{_row_t2(t2['per_model']['gpt'])[4:]}
<tr><th>llama</th>{_row_t2(t2['per_model']['llama'])[4:]}
</table>
<p class='interp'>{interp_h1b()}</p>

<h2>Tests 3/4/5 -- H2a, H2b, H2c: moderator regression</h2>
<p class='small'>Mixed-effects linear regression on <code>abs_gap</code>
with a random intercept grouped by <code>post_id</code>.
Formula (SES as 3-level factor):<br>
<code>{_html_esc(t3['full'].get('formula', 'n/a'))}</code></p>

<h3>Full model (SES 3-level)</h3>
{_regression_table(t3['full'])}

<h3>Binary SES contrast (H2c preregistration update)</h3>
<p class='small'>Same regression with <code>ses_level</code> collapsed
to <code>ses_strain = 1[ses_level == 'low']</code>. The
<code>ses_strain</code> coefficient is the contrast of interest.</p>
{_regression_table(t3['binary'])}

<h2>Test 6 -- H3: cross-model abs_gap difference</h2>
<p class='small'>Paired t-test on abs_gap by post_id (gpt - llama).
Becomes a 3-level within-post ANOVA when Claude is added.</p>
"""]
    if "error" in t6:
        parts.append(f"<p>skipped: {_html_esc(t6['error'])}</p>")
    else:
        parts.append(
            f"<table><tr><th>n pairs</th><th>mean_gpt</th><th>mean_llama</th>"
            f"<th>mean diff</th><th>t</th><th>p</th><th>Cohen's d</th>"
            f"<th>95% CI d</th></tr>"
            f"<tr><td>{t6['n']}</td><td>{fmt(t6['mean_gpt'])}</td>"
            f"<td>{fmt(t6['mean_llama'])}</td><td>{fmt(t6['mean_diff'])}</td>"
            f"<td>t({t6['df']})={fmt(t6['t'], 2)}</td>"
            f"<td>{fmt_p(t6['p'])}</td>"
            f"<td>{fmt(t6['d'], 2)} ({t6['label_size']})</td>"
            f"<td>[{fmt(t6['d_ci'][0], 2)}, {fmt(t6['d_ci'][1], 2)}]</td></tr></table>"
        )
    parts.append(f"<p class='interp'>{interp_h3()}</p>")

    parts.append("<h2>Test 7 -- cross-model agreement (descriptive, no test)</h2>")
    if "error" in t7:
        parts.append(f"<p>skipped: {_html_esc(t7['error'])}</p>")
    else:
        parts.append(
            f"<table><tr><th>n shared posts</th><th>both risky</th>"
            f"<th>both safe</th><th>agreement</th><th>Pearson r</th>"
            f"<th>r p-value</th></tr>"
            f"<tr><td>{t7['n']}</td><td>{t7['both_above']}</td>"
            f"<td>{t7['both_below']}</td>"
            f"<td>{t7['agreement'] * 100:.1f}%</td>"
            f"<td>{fmt(t7['pearson_r'], 3)}</td>"
            f"<td>{fmt_p(t7['pearson_p'])}</td></tr></table>"
        )
    parts.append(f"<p class='interp'>{interp_agreement()}</p>")

    parts.append("<h2>Test 8 -- stability: llm_entropy by consensus_level</h2>")
    parts.append("<p class='small'>One-way ANOVA per model. A well-"
                 "calibrated model should show F significantly &gt; 1 with "
                 "ambiguous &gt; high_safe / high_risky group means.</p>")
    parts.append("<table><tr><th>model</th><th>group sizes (safe/amb/risky)</th>"
                 "<th>mean safe</th><th>mean amb</th><th>mean risky</th>"
                 "<th>F</th><th>p</th><th>partial eta^2</th></tr>")
    for m in MODELS:
        r = t8["per_model"].get(m, {})
        if "error" in r:
            parts.append(f"<tr><th>{m}</th><td colspan='7'>skipped: "
                         f"{_html_esc(r['error'])} (sizes={r.get('group_sizes')})</td></tr>")
            continue
        means = r["means"]
        parts.append(
            f"<tr><th>{m}</th><td>{r['group_sizes']}</td>"
            f"<td>{fmt(means['high_safe'])}</td>"
            f"<td>{fmt(means['ambiguous'])}</td>"
            f"<td>{fmt(means['high_risky'])}</td>"
            f"<td>F({r['df_between']},{r['df_within']})={fmt(r['F'], 2)}</td>"
            f"<td>{fmt_p(r['p'])}</td><td>{fmt(r['partial_eta2'], 3)}</td></tr>"
        )
    parts.append("</table>")
    parts.append(f"<p class='interp'>{interp_stability()}</p>")

    parts.append(STRONG_DISCLAIMER_HTML)
    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Plain-text summary
# ---------------------------------------------------------------------------
def build_txt(results: dict) -> str:
    lines = [STRONG_DISCLAIMER_TXT, "=" * 72, ""]

    def section(title):
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    t1 = results["h1a"]
    section("Test 1 -- H1a: mean abs_gap vs 0 (1-sample t-test)")
    for k, r in [("pooled", t1["pooled"])] + list(t1["per_model"].items()):
        if "error" in r:
            lines.append(f"  {k}: skipped ({r['error']})")
            continue
        lo, hi = r["d_ci"]
        lines.append(
            f"  {k}: n={r['n']}, mean={r['mean']:.3f}, t({r['df']})="
            f"{r['t']:.2f}, p={fmt_p(r['p'])}, d={r['d']:.2f} ({r['label']}), "
            f"95% CI d=[{lo:.2f}, {hi:.2f}]"
        )

    t2 = results["h1b"]
    section("Test 2 -- H1b: entropy collapse on ambiguous (paired, one-sided)")
    for k, r in [("pooled", t2["pooled"])] + list(t2["per_model"].items()):
        if "error" in r:
            lines.append(f"  {k}: skipped ({r['error']})")
            continue
        lo, hi = r["d_ci"]
        lines.append(
            f"  {k}: n={r['n']}, mean_diff={r['mean_diff']:.3f}, t({r['df']})="
            f"{r['t']:.2f}, p_one={fmt_p(r['p_one_sided'])}, "
            f"d={r['d']:.2f} ({r['label']}), 95% CI d=[{lo:.2f}, {hi:.2f}]"
        )

    t3 = results["h2"]
    section("Tests 3/4/5 -- H2 mixed-effects regression")
    lines.append(f"  Formula: {t3['full'].get('formula')}")
    lines.append(f"  N (full model) = {t3['full']['n']}, "
                 f"converged={t3['full'].get('converged')}")
    lines.append(f"  Approx marginal R^2 (full) = "
                 f"{fmt(t3['full'].get('marginal_r2_approx'))}")
    lines.append(f"  Var(post_id) = {fmt(t3['full']['var_post_id'])}, "
                 f"Var(residual) = {fmt(t3['full']['resid_var'])}")
    lines.append("")
    lines.append("  Fixed-effect coefficients (full model):")
    for t in t3["full"]["terms"]:
        lines.append(
            f"    {t['term']}: coef={t['coef']:.3f} "
            f"(SE={t['se']:.3f}, 95% CI=[{t['ci_lo']:.3f},"
            f"{t['ci_hi']:.3f}], p={fmt_p(t['p'])})"
        )
    lines.append("")
    lines.append("  Binary-SES variant (H2c preregistration update):")
    for t in t3["binary"]["terms"]:
        if "ses_strain" in t["term"] or "Intercept" in t["term"]:
            lines.append(
                f"    {t['term']}: coef={t['coef']:.3f} "
                f"(SE={t['se']:.3f}, 95% CI=[{t['ci_lo']:.3f},"
                f"{t['ci_hi']:.3f}], p={fmt_p(t['p'])})"
            )

    t6 = results["h3"]
    section("Test 6 -- H3: paired t-test on abs_gap (gpt vs llama)")
    if "error" in t6:
        lines.append(f"  skipped: {t6['error']}")
    else:
        lo, hi = t6["d_ci"]
        lines.append(
            f"  n={t6['n']}, mean_gpt={t6['mean_gpt']:.3f}, "
            f"mean_llama={t6['mean_llama']:.3f}, diff={t6['mean_diff']:.3f}, "
            f"t({t6['df']})={t6['t']:.2f}, p={fmt_p(t6['p'])}, "
            f"d={t6['d']:.2f} ({t6['label_size']}), "
            f"95% CI d=[{lo:.2f}, {hi:.2f}]"
        )

    t7 = results["agreement"]
    section("Test 7 -- cross-model agreement (descriptive)")
    if "error" in t7:
        lines.append(f"  skipped: {t7['error']}")
    else:
        lines.append(
            f"  n={t7['n']}, agreement={t7['agreement']*100:.1f}%, "
            f"Pearson r={t7['pearson_r']:.3f}, p={fmt_p(t7['pearson_p'])}"
        )

    t8 = results["stability"]
    section("Test 8 -- stability: llm_entropy by consensus_level (one-way ANOVA)")
    for m in MODELS:
        r = t8["per_model"].get(m, {})
        if "error" in r:
            lines.append(f"  {m}: skipped ({r['error']}, sizes={r.get('group_sizes')})")
            continue
        means = r["means"]
        lines.append(
            f"  {m}: F({r['df_between']},{r['df_within']})={r['F']:.2f}, "
            f"p={fmt_p(r['p'])}, partial_eta^2={r['partial_eta2']:.3f}, "
            f"means(safe/amb/risky)="
            f"({means['high_safe']:.3f}/{means['ambiguous']:.3f}/"
            f"{means['high_risky']:.3f})"
        )

    lines.append("")
    lines.append(STRONG_DISCLAIMER_TXT)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-html", action="store_true",
                   help="Only log + write the TXT summary; do not render HTML.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = load_data()

    results = {
        "h1a":        test_h1a(df),
        "h1b":        test_h1b(df),
        "h2":         test_h2_abc(df),
        "h3":         test_h3(df),
        "agreement":  test_agreement(df),
        "stability":  test_stability(df),
    }

    txt = build_txt(results)
    SUMMARY_PATH.write_text(txt, encoding="utf-8")
    log.info("Wrote %s", SUMMARY_PATH)

    if args.skip_html:
        log.info("--skip-html set: HTML not rendered.")
        return 0

    html_out = build_html(results)
    REPORT_PATH.write_text(html_out, encoding="utf-8")
    log.info("Wrote %s", REPORT_PATH)
    log.info("Interim stats complete. Open %s in a browser.", REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
