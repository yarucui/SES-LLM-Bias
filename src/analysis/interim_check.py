"""
Interim descriptive analysis -- Llama + GPT only.

Two of the three preregistered study models have been run (Llama-4 Maverick
and GPT-5). Before spending the budget on Claude Sonnet 4.6, the user wants
to inspect whether the research hypotheses look directionally promising.

This script is intentionally *descriptive*. It does NOT run any statistical
hypothesis test, does NOT compute FDR / BH adjustment, and does NOT produce
text claiming that any hypothesis is confirmed or rejected. The
preregistered confirmatory analysis happens only after all three models are
collected; running tests on partial data would constitute peeking.

What the script does:
    * Loads data/experiment/comparison_table.csv
    * Filters to qualified_for_analysis == 1 and the requested model subset
    * Prints headline descriptive numbers
    * Renders five figures into data/analysis/interim_figures/
    * Writes a single-page HTML report that embeds all five figures inline
      alongside a plain-English "what to look for" / "what this run shows"
      caption per figure
    * Writes a plain-text summary for grepping / logging

Usage:
    python src/analysis/interim_check.py
    python src/analysis/interim_check.py --models gpt llama
    python src/analysis/interim_check.py --skip-figures
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Paths + logging
# ---------------------------------------------------------------------------
ANALYSIS_DIR = config.DATA_DIR / "analysis"
FIGURES_DIR = ANALYSIS_DIR / "interim_figures"
REPORT_PATH = ANALYSIS_DIR / "interim_report.html"
SUMMARY_PATH = ANALYSIS_DIR / "interim_summary.txt"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "interim_check.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("interim_check")

COMPARISON_CSV = config.EXPERIMENT_DIR / "comparison_table.csv"

# Deterministic palette so both figures and the HTML caption text line up.
MODEL_PALETTE = {"gpt": "#1f77b4", "llama": "#d62728", "claude": "#2ca02c"}

DISCLAIMER = (
    "Interim descriptive analysis. Confirmatory tests will be run after "
    "all three preregistered models are collected."
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(models: list[str]) -> pd.DataFrame:
    if not COMPARISON_CSV.exists():
        raise FileNotFoundError(
            f"comparison_table.csv not found at {COMPARISON_CSV}. Run step5 first."
        )
    df = pd.read_csv(COMPARISON_CSV)
    log.info("Loaded %d rows from %s", len(df), COMPARISON_CSV.name)

    df = df[df["qualified_for_analysis"] == 1].copy()
    log.info("After qualified_for_analysis==1 filter: %d rows", len(df))

    df = df[df["model"].isin(models)].copy()
    log.info("After model filter (%s): %d rows", ", ".join(models), len(df))

    if df.empty:
        raise RuntimeError(
            "No rows remain after filtering. Check --models and that step5 "
            "has been run for those models."
        )

    # Seaborn/matplotlib prefer category dtypes with an explicit order for
    # consistent panel ordering.
    consensus_order = ["high_safe", "ambiguous", "high_risky"]
    df["consensus_level"] = pd.Categorical(
        df["consensus_level"], categories=consensus_order, ordered=True
    )
    return df


# ---------------------------------------------------------------------------
# Headline descriptive numbers
# ---------------------------------------------------------------------------
def compute_headline(df: pd.DataFrame) -> dict:
    """Compute the descriptive numbers used both in logs and in the HTML."""
    out: dict = {"per_model": {}, "cross_model": {}, "ambiguous": {}, "ses": {}}

    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        out["per_model"][model] = {
            "n_posts": int(sub["post_id"].nunique()),
            "n_qualified": int(len(sub)),
            "mean_abs_gap": float(sub["abs_gap"].mean()),
            "median_abs_gap": float(sub["abs_gap"].median()),
            "mean_entropy_gap": float(sub["entropy_gap"].mean()),
        }

    # Cross-model directional agreement (on posts both models covered).
    pivot = df.pivot_table(
        index="post_id", columns="model", values="llm_risky_rate", aggfunc="mean"
    ).dropna()
    if pivot.shape[1] >= 2 and {"gpt", "llama"}.issubset(pivot.columns):
        both_above = ((pivot["gpt"] > 0.5) & (pivot["llama"] > 0.5)).sum()
        both_below = ((pivot["gpt"] <= 0.5) & (pivot["llama"] <= 0.5)).sum()
        agreement = (both_above + both_below) / len(pivot) if len(pivot) else float("nan")
        out["cross_model"] = {
            "n_shared_posts": int(len(pivot)),
            "directional_agreement": float(agreement),
        }
    else:
        out["cross_model"] = {"n_shared_posts": int(len(pivot)), "directional_agreement": None}

    # Ambiguous-stratum entropy gaps (the Finding-1 signal).
    for model in sorted(df["model"].unique()):
        amb = df[(df["model"] == model) & (df["consensus_level"] == "ambiguous")]
        out["ambiguous"][model] = {
            "n": int(len(amb)),
            "mean_entropy_gap": float(amb["entropy_gap"].mean()) if len(amb) else float("nan"),
        }

    # Low vs non-low SES (collapsed binary).
    df_ses = df.copy()
    df_ses["ses_strain"] = (df_ses["ses_level"] == "low").astype(int)
    for model in sorted(df["model"].unique()):
        sub = df_ses[df_ses["model"] == model]
        low = sub[sub["ses_strain"] == 1]["abs_gap"]
        non = sub[sub["ses_strain"] == 0]["abs_gap"]
        out["ses"][model] = {
            "n_low": int(len(low)),
            "n_non_low": int(len(non)),
            "median_abs_gap_low": float(low.median()) if len(low) else float("nan"),
            "median_abs_gap_non_low": float(non.median()) if len(non) else float("nan"),
        }

    return out


def log_headline(h: dict) -> None:
    for model, s in h["per_model"].items():
        log.info(
            "%s: n_posts=%d, n_qualified=%d, mean_abs_gap=%.3f, mean_entropy_gap=%.3f",
            model, s["n_posts"], s["n_qualified"], s["mean_abs_gap"], s["mean_entropy_gap"],
        )
    cm = h["cross_model"]
    if cm.get("directional_agreement") is not None:
        log.info(
            "Cross-model directional agreement (gpt vs llama) on %d shared posts: %.1f%%",
            cm["n_shared_posts"], cm["directional_agreement"] * 100,
        )


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", path)
    return path


def _annotate_counts(ax: plt.Axes, df: pd.DataFrame, x_col: str, hue_col: str | None = None) -> None:
    """Annotate each x category (and optionally hue sub-group) with n=...."""
    if hue_col is None:
        counts = df.groupby(x_col, observed=True).size()
        for i, cat in enumerate(counts.index):
            ax.text(i, ax.get_ylim()[0], f"n={counts.loc[cat]}",
                    ha="center", va="bottom", fontsize=8, color="gray")
    else:
        counts = df.groupby([x_col, hue_col], observed=True).size()
        summary = counts.groupby(level=0, observed=True).sum()
        for i, cat in enumerate(summary.index):
            ax.text(i, ax.get_ylim()[0], f"n={summary.loc[cat]}",
                    ha="center", va="bottom", fontsize=8, color="gray")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig1_distribution_shape(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1a -- overlaid histograms
    ax = axes[0]
    humans = df.drop_duplicates("post_id")["human_risky_ratio"]
    ax.hist(humans, bins=25, alpha=0.5, label="human", color="#444444", density=True)
    for model in sorted(df["model"].unique()):
        ax.hist(
            df[df["model"] == model]["llm_risky_rate"],
            bins=25, alpha=0.5, label=model,
            color=MODEL_PALETTE.get(model, None), density=True,
        )
    ax.set_xlabel("risky ratio")
    ax.set_ylabel("density")
    ax.set_title("Choice rate distributions")
    ax.set_xlim(-0.02, 1.02)
    ax.legend()

    # 1b -- entropy scatter (human vs llm)
    ax = axes[1]
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        ax.scatter(sub["human_entropy"], sub["llm_entropy"],
                   s=12, alpha=0.45, label=model,
                   color=MODEL_PALETTE.get(model, None))
    lim_max = max(df["human_entropy"].max(), df["llm_entropy"].max(), 0.7)
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1, alpha=0.6)
    ax.set_xlabel("human_entropy")
    ax.set_ylabel("llm_entropy")
    ax.set_title("Per-scenario entropy: humans vs LLM")
    ax.text(0.02, 0.97,
            "Points below diagonal =\nLLM more confident (entropy collapsed)",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    ax.legend()

    # 1c -- entropy gap by consensus level
    ax = axes[2]
    palette = {m: MODEL_PALETTE.get(m, "gray") for m in df["model"].unique()}
    sns.boxplot(data=df, x="consensus_level", y="entropy_gap",
                hue="model", ax=ax, palette=palette)
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Entropy gap (positive = LLM collapsed)")
    ax.set_xlabel("consensus_level")
    ax.set_ylabel("entropy_gap")

    fig.suptitle("Figure 1 -- distribution shape (Finding 1)", fontsize=13)
    fig.tight_layout()
    return _save(fig, "fig1_distribution_shape.png")


def fig2_decision_features(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    palette = {m: MODEL_PALETTE.get(m, "gray") for m in df["model"].unique()}

    def panel(ax, x_col, order, title):
        sub = df.dropna(subset=[x_col]).copy()
        sub[x_col] = pd.Categorical(sub[x_col], categories=order, ordered=True)
        sns.boxplot(data=sub, x=x_col, y="abs_gap", hue="model",
                    order=order, ax=ax, palette=palette)
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel("abs_gap")
        _annotate_counts(ax, sub, x_col, "model")

    panel(axes[0], "reversibility",
          ["reversible", "partially_reversible", "irreversible"],
          "abs_gap by reversibility")
    panel(axes[1], "time_horizon", ["short", "long"],
          "abs_gap by time_horizon")
    panel(axes[2], "resource_constraint",
          ["financial", "time", "social", "health", "geographic"],
          "abs_gap by resource_constraint")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle("Figure 2 -- decision feature heterogeneity (Finding 2a)", fontsize=13)
    fig.tight_layout()
    return _save(fig, "fig2_decision_features.png")


def fig3_ses_strain(df: pd.DataFrame) -> Path:
    # Collapse SES to binary (low vs non-low) per preregistration update.
    d = df.copy()
    d["ses_strain"] = (d["ses_level"] == "low").astype(int)

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {m: MODEL_PALETTE.get(m, "gray") for m in d["model"].unique()}
    sns.boxplot(data=d, x="ses_strain", y="abs_gap",
                hue="model", ax=ax, palette=palette)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["non-low (0)", "low (1)"])
    ax.set_xlabel("ses_strain")
    ax.set_ylabel("abs_gap")

    n_low = int((d["ses_strain"] == 1).sum())
    n_non = int((d["ses_strain"] == 0).sum())
    ax.text(0.02, 0.97,
            f"Low-SES rows: n={n_low}. Non-low rows: n={n_non}.",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    fig.suptitle("Figure 3 -- SES strain (Finding 2b, binary contrast)", fontsize=13)
    fig.tight_layout()
    return _save(fig, "fig3_ses_strain.png")


def fig4_cross_model(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 4a -- abs_gap by model
    ax = axes[0]
    palette = {m: MODEL_PALETTE.get(m, "gray") for m in df["model"].unique()}
    sns.violinplot(data=df, x="model", y="abs_gap",
                   hue="model", ax=ax, palette=palette,
                   inner="quartile", cut=0, legend=False)
    ax.set_title("abs_gap by model (violin)")
    for i, model in enumerate(sorted(df["model"].unique())):
        med = df[df["model"] == model]["abs_gap"].median()
        q1 = df[df["model"] == model]["abs_gap"].quantile(0.25)
        q3 = df[df["model"] == model]["abs_gap"].quantile(0.75)
        ax.text(i, ax.get_ylim()[1] * 0.95,
                f"median={med:.3f}\nIQR=[{q1:.3f}, {q3:.3f}]",
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # 4b -- per-post cross-model scatter coloured by human_risky_ratio
    ax = axes[1]
    pivot = df.pivot_table(
        index="post_id", columns="model", values="llm_risky_rate", aggfunc="mean"
    ).dropna()
    human_by_post = df.drop_duplicates("post_id").set_index("post_id")["human_risky_ratio"]
    if {"gpt", "llama"}.issubset(pivot.columns) and len(pivot) > 0:
        colours = human_by_post.reindex(pivot.index)
        sc = ax.scatter(pivot["gpt"], pivot["llama"],
                        c=colours, cmap="viridis", s=20, alpha=0.7,
                        edgecolor="none")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
        ax.set_xlabel("gpt llm_risky_rate")
        ax.set_ylabel("llama llm_risky_rate")
        ax.set_title(f"Where do GPT and Llama disagree? (n={len(pivot)} shared)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("human_risky_ratio")
    else:
        ax.text(0.5, 0.5,
                "Need both 'gpt' and 'llama' in --models for this panel",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()

    fig.suptitle("Figure 4 -- cross-model comparison", fontsize=13)
    fig.tight_layout()
    return _save(fig, "fig4_cross_model.png")


def fig5_stability(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = {m: MODEL_PALETTE.get(m, "gray") for m in df["model"].unique()}
    sns.boxplot(data=df, x="consensus_level", y="llm_entropy",
                hue="model", ax=ax, palette=palette)
    ax.axhline(np.log(2), color="black", linestyle="--", linewidth=1, alpha=0.7,
               label="log(2) max binary entropy")
    ax.set_title("Within-model entropy by consensus level")
    ax.set_xlabel("consensus_level")
    ax.set_ylabel("llm_entropy")
    ax.legend(title=None, loc="upper right")
    fig.suptitle("Figure 5 -- stability / calibration", fontsize=13)
    fig.tight_layout()
    return _save(fig, "fig5_stability.png")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
def _img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def build_html(df: pd.DataFrame, headline: dict, fig_paths: dict[str, Path]) -> str:
    models_sorted = sorted(df["model"].unique())

    # Per-figure "what this run shows" snippets.
    amb = headline["ambiguous"]
    amb_line = ", ".join(
        f"{m}={_fmt(amb[m]['mean_entropy_gap'])}" for m in models_sorted
    )

    per_model = headline["per_model"]
    per_model_line = ", ".join(
        f"{m} median={_fmt(per_model[m]['median_abs_gap'])}" for m in models_sorted
    )

    ses = headline["ses"]
    ses_lines = " | ".join(
        f"{m}: low median={_fmt(ses[m]['median_abs_gap_low'])}, "
        f"non-low median={_fmt(ses[m]['median_abs_gap_non_low'])}"
        for m in models_sorted
    )

    cm = headline["cross_model"]
    if cm.get("directional_agreement") is not None:
        cm_line = (f"Directional agreement between GPT and Llama on "
                   f"{cm['n_shared_posts']} shared posts: "
                   f"{cm['directional_agreement'] * 100:.1f}%.")
    else:
        cm_line = "Cross-model agreement not computed (both models not present)."

    # Decision-support auto-fills.
    gpt_amb = amb.get("gpt", {}).get("mean_entropy_gap", float("nan"))
    llama_amb = amb.get("llama", {}).get("mean_entropy_gap", float("nan"))

    figure_blocks = [
        {
            "title": "Figure 1 -- distribution shape (Finding 1)",
            "look_for": (
                "If the third panel shows the entropy gap is largest in the "
                "<b>ambiguous</b> stratum, this supports H1b (LLMs collapse "
                "to majority answers when humans are genuinely split)."
            ),
            "this_run": f"Mean entropy gap on ambiguous posts: {amb_line}. "
                        f"Values above zero indicate the LLM's per-scenario "
                        f"distribution is tighter than the human distribution.",
            "path": fig_paths["fig1"],
        },
        {
            "title": "Figure 2 -- decision feature heterogeneity (Finding 2a)",
            "look_for": (
                "Look for visibly larger abs_gap in <b>irreversible</b> "
                "decisions and in <b>long</b>-horizon decisions. Resource "
                "constraint is exploratory; sample sizes per cell vary."
            ),
            "this_run": f"Overall median abs_gap per model: {per_model_line}.",
            "path": fig_paths["fig2"],
        },
        {
            "title": "Figure 3 -- SES strain (Finding 2b, binary)",
            "look_for": (
                "Given the heavy skew in the SES distribution, the "
                "preregistration update collapses to low vs non-low. If the "
                "low bar is meaningfully higher than non-low, that is an "
                "early signal for H2c."
            ),
            "this_run": ses_lines,
            "path": fig_paths["fig3"],
        },
        {
            "title": "Figure 4 -- cross-model comparison",
            "look_for": (
                "Panel 4b tells you how much new information a third model "
                "would add. If GPT and Llama already cluster tightly along "
                "the diagonal, Claude is unlikely to change the qualitative "
                "picture -- though it is still required for the "
                "preregistered confirmatory tests."
            ),
            "this_run": cm_line,
            "path": fig_paths["fig4"],
        },
        {
            "title": "Figure 5 -- stability / calibration",
            "look_for": (
                "A well-calibrated model should show entropy that roughly "
                "tracks the human consensus level (low entropy where humans "
                "agree, higher where humans are split). A flat, consistently "
                "low line indicates collapse regardless of human uncertainty."
            ),
            "this_run": (
                "See per-model median entropy in each consensus stratum. "
                "The dashed line at log(2) = 0.693 marks the theoretical "
                "maximum for a binary choice."
            ),
            "path": fig_paths["fig5"],
        },
    ]

    # Decision-support heuristic block.
    decision_points = []
    if not np.isnan(gpt_amb) and not np.isnan(llama_amb):
        avg_amb = (gpt_amb + llama_amb) / 2
        decision_points.append(
            f"<li>Ambiguous-stratum mean entropy gap: gpt={_fmt(gpt_amb)}, "
            f"llama={_fmt(llama_amb)} (avg={_fmt(avg_amb)}). "
            f"Heuristic &gt;0.15 &rarr; likely worth running Claude.</li>"
        )
    if cm.get("directional_agreement") is not None:
        decision_points.append(
            f"<li>GPT-vs-Llama directional agreement: "
            f"{cm['directional_agreement'] * 100:.1f}% across "
            f"{cm['n_shared_posts']} shared posts. "
            f"&gt;90% means Claude is unlikely to move the qualitative "
            f"picture, though still required for preregistered tests.</li>"
        )
    if decision_points:
        decision_html = "<ul>" + "\n".join(decision_points) + "</ul>"
    else:
        decision_html = "<p>Insufficient data to auto-fill decision points.</p>"

    # Assemble HTML.
    body_parts = [f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Interim check -- Llama + GPT</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1100px;
        margin: 2em auto; padding: 0 1em; color: #222; }}
 h1 {{ border-bottom: 2px solid #888; padding-bottom: 0.3em; }}
 h2 {{ margin-top: 2em; color: #333; }}
 .disclaimer {{ background: #fff6d6; border: 1px solid #d4b73e;
                padding: 0.8em 1em; border-radius: 6px; margin: 1em 0; }}
 .summary-table {{ border-collapse: collapse; margin: 1em 0; }}
 .summary-table th, .summary-table td {{ border: 1px solid #ccc;
                                          padding: 6px 10px; text-align: right; }}
 .summary-table th {{ background: #f4f4f4; text-align: center; }}
 img {{ max-width: 100%; border: 1px solid #e0e0e0;
        border-radius: 4px; margin-top: 0.5em; }}
 .caption {{ font-size: 0.95em; }}
 .caption b {{ color: #333; }}
</style></head><body>
<h1>Interim check -- Llama + GPT</h1>
<div class='disclaimer'><b>{DISCLAIMER}</b></div>
<h2>Headline descriptive numbers</h2>
<table class='summary-table'>
<tr><th>model</th><th>n_posts</th><th>n_qualified</th>
    <th>mean abs_gap</th><th>median abs_gap</th><th>mean entropy_gap</th></tr>"""]

    for m in models_sorted:
        s = per_model[m]
        body_parts.append(
            f"<tr><th>{m}</th>"
            f"<td>{s['n_posts']}</td><td>{s['n_qualified']}</td>"
            f"<td>{_fmt(s['mean_abs_gap'])}</td>"
            f"<td>{_fmt(s['median_abs_gap'])}</td>"
            f"<td>{_fmt(s['mean_entropy_gap'])}</td></tr>"
        )
    body_parts.append("</table>")
    body_parts.append(f"<p>{cm_line}</p>")

    for block in figure_blocks:
        b64 = _img_b64(block["path"])
        body_parts.append(f"""<h2>{block['title']}</h2>
<p class='caption'><b>What to look for:</b> {block['look_for']}</p>
<p class='caption'><b>What this run shows:</b> {block['this_run']}</p>
<img src='data:image/png;base64,{b64}' />""")

    body_parts.append(f"""<h2>Should we run Claude Sonnet 4.6?</h2>
<p>Heuristics (descriptive, not a stopping rule):</p>
<ul>
  <li>If the ambiguous-stratum entropy gap is roughly &gt;0.15 for both
      models, running Claude is likely worth it.</li>
  <li>If the reversibility or time_horizon panels show a difference of
      &gt;0.05 in median abs_gap with the expected sign, running Claude is
      likely worth it.</li>
  <li>If GPT and Llama already agree on 90%+ of post-level directions,
      Claude probably will not move the qualitative picture much, but is
      still needed for the preregistered confirmatory tests.</li>
</ul>
<p><b>Observed values from this run:</b></p>
{decision_html}
<div class='disclaimer'>{DISCLAIMER}</div>
</body></html>""")

    return "\n".join(body_parts)


# ---------------------------------------------------------------------------
# Plain text summary
# ---------------------------------------------------------------------------
def write_summary_txt(headline: dict) -> None:
    lines = ["Interim descriptive summary (Llama + GPT only)", "=" * 48, ""]
    for model, s in headline["per_model"].items():
        lines.append(
            f"{model}: n_posts={s['n_posts']}, n_qualified={s['n_qualified']}, "
            f"mean_abs_gap={s['mean_abs_gap']:.3f}, "
            f"median_abs_gap={s['median_abs_gap']:.3f}, "
            f"mean_entropy_gap={s['mean_entropy_gap']:.3f}"
        )

    cm = headline["cross_model"]
    if cm.get("directional_agreement") is not None:
        lines.append("")
        lines.append(
            f"Cross-model directional agreement "
            f"(n={cm['n_shared_posts']}): "
            f"{cm['directional_agreement'] * 100:.1f}%"
        )

    lines.append("")
    lines.append("Ambiguous-stratum mean entropy gap:")
    for m, s in headline["ambiguous"].items():
        lines.append(f"  {m}: n={s['n']}, mean_entropy_gap={s['mean_entropy_gap']:.3f}")

    lines.append("")
    lines.append("SES strain (binary):")
    for m, s in headline["ses"].items():
        lines.append(
            f"  {m}: low n={s['n_low']} (median abs_gap={s['median_abs_gap_low']:.3f}), "
            f"non-low n={s['n_non_low']} (median abs_gap={s['median_abs_gap_non_low']:.3f})"
        )

    lines.append("")
    lines.append(DISCLAIMER)

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", SUMMARY_PATH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["gpt", "llama"],
                   help="Models to include (default: gpt llama).")
    p.add_argument("--skip-figures", action="store_true",
                   help="Only compute and log headline numbers; do not render figures or HTML.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sns.set_theme(style="whitegrid")

    df = load_data(args.models)
    headline = compute_headline(df)
    log_headline(headline)

    if args.skip_figures:
        write_summary_txt(headline)
        log.info("--skip-figures set: figures and HTML not rendered.")
        return 0

    fig_paths = {
        "fig1": fig1_distribution_shape(df),
        "fig2": fig2_decision_features(df),
        "fig3": fig3_ses_strain(df),
        "fig4": fig4_cross_model(df),
        "fig5": fig5_stability(df),
    }

    html = build_html(df, headline, fig_paths)
    REPORT_PATH.write_text(html, encoding="utf-8")
    log.info("Wrote %s", REPORT_PATH)

    write_summary_txt(headline)
    log.info("Interim check complete. Open %s in a browser.", REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
