"""
Step 2b — SES annotation validation and human reliability check.

DESIGN (v5.2):

What this step does — three things only, no LLM calls:

  1. VALIDATE — reads all_scored_ok.jsonl from step 2, checks that
     ses_cue_intensity is in {A0, A1, A2} and ses_sensitivity is in
     {B0, B1, B2}. Posts with invalid/missing SES fields are flagged
     and excluded. To fix them, re-run step 2 with --rerun-errors.

  2. OUTPUT — writes all_scored_valid.jsonl containing every post with
     a valid SES annotation, regardless of ses_sensitivity level.
     ALL of B0/B1/B2 and A0/A1/A2 are included. Step 3 samples from
     this and carries ses_sensitivity as an analysis-time stratification
     variable. Excluding B0 posts would remove the cleanest bias signal:
     any SES differential on a B0 scenario cannot be explained as
     rational inference — it is unambiguous bias.

  3. HUMAN VALIDATION — exports a stratified 20% sample as CSV for
     manual coding. If human labels already exist, computes
     Krippendorff's alpha between Gemini and human codings.

Why no LLM calls:
  Step 2 Call B handles SES annotation with internal retries.
  Posts where Call B failed are marked ses_call_b_error and can be
  retried via: python step2_score.py --rerun-errors

Why no B2-only filter:
  B0 scenarios (SES should not affect advice) are the strongest bias
  test — LLM differentials there cannot be rationalised as legitimate
  SES-sensitive advice. ses_sensitivity is preserved for analysis-time
  stratification (B0 vs B1 vs B2 as a moderator variable).

Usage:
    python src/pipeline/step2b_ses_filter.py
    python src/pipeline/step2b_ses_filter.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "step2b_ses_filter.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("step2b_ses_filter")

VALID_CUE_LEVELS       = {"A0", "A1", "A2"}
VALID_SENS_LEVELS      = {"B0", "B1", "B2"}
HUMAN_SAMPLE_FRACTION  = 0.20
KRIPPENDORFF_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def ses_annotation_is_valid(rec: dict) -> bool:
    return (
        rec.get("ses_cue_intensity") in VALID_CUE_LEVELS
        and rec.get("ses_sensitivity") in VALID_SENS_LEVELS
    )


# ---------------------------------------------------------------------------
# Krippendorff's alpha (nominal, two coders)
# ---------------------------------------------------------------------------

def krippendorff_alpha_nominal(pairs: list[tuple[str, str]]) -> float | None:
    clean = [(a, b) for a, b in pairs if a and b]
    n = len(clean)
    if n < 2:
        return None
    D_o = sum(1 for a, b in clean if a != b) / n
    counts: Counter = Counter()
    for a, b in clean:
        counts[a] += 1
        counts[b] += 1
    total = sum(counts.values())
    D_e = 1.0 - sum(c / total * (c - 1) / (total - 1) for c in counts.values())
    if D_e == 0:
        return 1.0 if D_o == 0 else None
    alpha = 1.0 - D_o / D_e
    return None if (math.isnan(alpha) or math.isinf(alpha)) else alpha


# ---------------------------------------------------------------------------
# Human validation CSV
# ---------------------------------------------------------------------------

HUMAN_COLS = [
    "post_id", "domain", "consensus_level", "ses_sensitivity", "ses_cue_intensity",
    "title", "selftext_excerpt", "option_risky", "option_safe",
    "gemini_ses_cue_intensity", "gemini_ses_sensitivity",
    "gemini_ses_flip_reasoning", "gemini_ses_natural_cues",
    "human_ses_cue_intensity", "human_ses_sensitivity", "human_notes",
]

HUMAN_INSTRUCTIONS = (
    "# Fill in human_ses_cue_intensity (A0/A1/A2) and "
    "human_ses_sensitivity (B0/B1/B2) for each row.\n"
    "# ses_sensitivity: would substantially more resources flip the recommendation?\n"
    "# B0=no  B1=weakly  B2=strongly (could reverse advice)\n"
    "# All sensitivity levels included — do not skip B0 rows.\n"
)


def load_existing_human_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            lines = [l for l in f if not l.startswith("#")]
        for row in csv.DictReader(lines):
            pid = (row.get("post_id") or "").strip()
            hc  = (row.get("human_ses_cue_intensity") or "").strip()
            hs  = (row.get("human_ses_sensitivity") or "").strip()
            if pid and (hc or hs):
                out[pid] = {
                    "human_ses_cue_intensity": hc,
                    "human_ses_sensitivity":   hs,
                }
    except Exception as e:
        log.warning("Could not parse human validation CSV: %s", e)
    return out


def write_human_validation_sample(
    sample: list[dict], path: Path, existing_human: dict[str, dict]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(HUMAN_INSTRUCTIONS)
        w = csv.DictWriter(f, fieldnames=HUMAN_COLS)
        w.writeheader()
        for r in sample:
            pid   = r["post_id"]
            prior = existing_human.get(pid, {})
            w.writerow({
                "post_id":           pid,
                "domain":            r.get("domain", ""),
                "consensus_level":   r.get("consensus_level", ""),
                "ses_sensitivity":   r.get("ses_sensitivity", ""),
                "ses_cue_intensity": r.get("ses_cue_intensity", ""),
                "title":             (r.get("title") or "")[:200],
                "selftext_excerpt":  (r.get("selftext") or "")[:300].replace("\n", " "),
                "option_risky":      (r.get("option_risky") or "")[:200],
                "option_safe":       (r.get("option_safe") or "")[:200],
                "gemini_ses_cue_intensity":  r.get("ses_cue_intensity", ""),
                "gemini_ses_sensitivity":    r.get("ses_sensitivity", ""),
                "gemini_ses_flip_reasoning": (r.get("ses_flip_reasoning") or "")[:400],
                "gemini_ses_natural_cues":   json.dumps(r.get("ses_natural_cues", [])),
                "human_ses_cue_intensity":   prior.get("human_ses_cue_intensity", ""),
                "human_ses_sensitivity":     prior.get("human_ses_sensitivity", ""),
                "human_notes":               "",
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2b: validate SES annotations, export human validation sample."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write output files.")
    args = parser.parse_args()

    config.POSTS_SCORED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load all OK posts from step 2 ─────────────────────────────────────
    src = config.POSTS_SCORED_DIR / "all_scored_ok.jsonl"
    if not src.exists():
        log.error("Missing %s — run step2_score.py first", src)
        sys.exit(1)

    all_posts: list[dict] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_posts.append(json.loads(line))
    log.info("Loaded %d scored posts from step 2", len(all_posts))

    # ── Validate SES fields ────────────────────────────────────────────────
    valid_posts   = [p for p in all_posts if ses_annotation_is_valid(p)]
    invalid_posts = [p for p in all_posts if not ses_annotation_is_valid(p)]

    log.info("Valid SES annotations : %d / %d", len(valid_posts), len(all_posts))
    if invalid_posts:
        log.warning(
            "%d posts have invalid/missing SES fields. "
            "Fix with: python step2_score.py --rerun-errors",
            len(invalid_posts),
        )

    # ── Write all valid posts → step 3 ────────────────────────────────────
    if not args.dry_run:
        out_path = config.POSTS_SCORED_DIR / "all_scored_valid.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in valid_posts:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("Wrote %d posts → all_scored_valid.jsonl", len(valid_posts))

    # ── Breakdown stats ────────────────────────────────────────────────────
    sens_counts = Counter(p.get("ses_sensitivity") for p in valid_posts)
    cue_counts  = Counter(p.get("ses_cue_intensity") for p in valid_posts)
    dom_counts  = Counter(p.get("domain") for p in valid_posts)
    cons_counts = Counter(p.get("consensus_level") for p in valid_posts)
    chan_counts: Counter = Counter()
    for p in valid_posts:
        for ch in (p.get("ses_channels") or []):
            chan_counts[ch] += 1

    log.info("")
    log.info("SES sensitivity: %s", dict(sens_counts))
    log.info("SES cue level:   %s", dict(cue_counts))
    log.info("By domain:       %s", dict(dom_counts))
    log.info("By consensus:    %s", dict(cons_counts))
    log.info("Top SES channels: %s",
             ", ".join(f"{k}={v}" for k, v in chan_counts.most_common(5)))

    # ── Human validation sample ────────────────────────────────────────────
    # Stratified across domain × consensus_level × ses_sensitivity
    rng = random.Random(42)
    sample: list[dict] = []
    by_stratum: dict[tuple, list] = defaultdict(list)
    for p in valid_posts:
        key = (
            p.get("domain", "?"),
            p.get("consensus_level", "?"),
            p.get("ses_sensitivity", "?"),
        )
        by_stratum[key].append(p)

    for items in by_stratum.values():
        k = max(1, round(len(items) * HUMAN_SAMPLE_FRACTION))
        sample.extend(rng.sample(items, min(k, len(items))))

    human_csv      = config.POSTS_SCORED_DIR / "ses_human_validation_sample.csv"
    existing_human = load_existing_human_labels(human_csv)

    if not args.dry_run:
        write_human_validation_sample(sample, human_csv, existing_human)

    # ── Krippendorff's alpha ───────────────────────────────────────────────
    alpha_cue = alpha_sen = None
    if existing_human:
        by_pid = {p["post_id"]: p for p in valid_posts}
        pairs_cue, pairs_sen = [], []
        for pid, h in existing_human.items():
            g = by_pid.get(pid)
            if not g:
                continue
            if g.get("ses_cue_intensity") and h.get("human_ses_cue_intensity"):
                pairs_cue.append((g["ses_cue_intensity"], h["human_ses_cue_intensity"]))
            if g.get("ses_sensitivity") and h.get("human_ses_sensitivity"):
                pairs_sen.append((g["ses_sensitivity"], h["human_ses_sensitivity"]))

        alpha_cue = krippendorff_alpha_nominal(pairs_cue)
        alpha_sen = krippendorff_alpha_nominal(pairs_sen)
        for name, a in [("cue_intensity", alpha_cue), ("sensitivity", alpha_sen)]:
            status = "OK" if (a is not None and a >= KRIPPENDORFF_THRESHOLD) else "BELOW THRESHOLD"
            log.info("Krippendorff alpha (%s): %s  [%s]",
                     name, f"{a:.3f}" if a is not None else "n/a", status)

    # ── Write report ───────────────────────────────────────────────────────
    report = {
        "total_input":        len(all_posts),
        "valid_annotation":   len(valid_posts),
        "invalid_annotation": len(invalid_posts),
        "sensitivity_counts": dict(sens_counts),
        "cue_counts":         dict(cue_counts),
        "domain_counts":      dict(dom_counts),
        "consensus_counts":   dict(cons_counts),
        "top_ses_channels":   dict(chan_counts.most_common(6)),
        "human_sample_size":  len(sample),
        "krippendorff_alpha_cue_intensity": alpha_cue,
        "krippendorff_alpha_sensitivity":   alpha_sen,
    }
    if not args.dry_run:
        with (config.POSTS_SCORED_DIR / "ses_annotation_report.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(report, f, indent=2)

    log.info("")
    log.info("=" * 60)
    log.info("STEP 2b COMPLETE")
    log.info("Valid posts → step 3   : %d  (B0=%d B1=%d B2=%d)",
             len(valid_posts),
             sens_counts.get("B0", 0),
             sens_counts.get("B1", 0),
             sens_counts.get("B2", 0))
    log.info("Invalid (excluded)     : %d", len(invalid_posts))
    log.info("Human validation sample: %d posts", len(sample))
    log.info("")
    log.info("Output files:")
    log.info("  all_scored_valid.jsonl          → step 3 (all SES-valid posts)")
    log.info("  ses_annotation_report.json      → counts by sensitivity/cue/domain")
    log.info("  ses_human_validation_sample.csv → %d posts for coding", len(sample))
    log.info("")
    log.info("Next: python step3_extract.py")


if __name__ == "__main__":
    main()