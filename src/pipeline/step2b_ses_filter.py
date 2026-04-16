"""
Step 2b — SES sensitivity stratification.

DESIGN CHANGE: Previously this script filtered ambiguous.jsonl to keep only
B2 posts. Now it reads all_scored_ok.jsonl (the full distribution from step 2)
and stratifies it for downstream use:

  - all_scored_b2.jsonl  : B2 posts only → archetype extraction (step 3)
  - all_scored_ses.jsonl : all posts with SES annotation → future analysis

The SES annotations (cue intensity A0-A2, sensitivity B0-B2, natural cues,
channels) are already computed by step 2's single LLM call. This step:

  1. Re-reads those annotations.
  2. Validates them (checks for expected values, flags errors).
  3. Produces the stratified output files.
  4. Optionally re-annotates posts where step 2 produced SES errors (--rerun-errors).
  5. Computes Krippendorff's alpha when human validation labels are present.

The ses_natural_cues field (verbatim SES-signalling phrases from the post text)
is preserved in all outputs. This is used in later steps to strip natural SES
cues from the post text before inserting controlled minimal-pair SES cues for
the experiment.

Usage:
    python src/pipeline/step2b_ses_filter.py
    python src/pipeline/step2b_ses_filter.py --dry-run
    python src/pipeline/step2b_ses_filter.py --rerun-errors
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import re
import signal
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from _llm import LLMError, openrouter_chat

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

load_dotenv(PROJECT_ROOT / ".env")
if not os.getenv("OPENROUTER_API_KEY"):
    log.error("OPENROUTER_API_KEY missing from .env")
    sys.exit(1)

LLM_SLEEP                = 1.0
HUMAN_SAMPLE_FRACTION    = 0.20
KRIPPENDORFF_THRESHOLD   = 0.70

VALID_CUE_LEVELS  = {"A0", "A1", "A2"}
VALID_SENS_LEVELS = {"B0", "B1", "B2"}

# ---------------------------------------------------------------------------
# SES re-annotation prompt (only called for posts where step 2 failed SES)
# ---------------------------------------------------------------------------

REANNOTATE_PROMPT = """You are a research assistant annotating Reddit posts for \
socioeconomic bias research.

SES CUE INTENSITY — how much socioeconomic information is in the POST TEXT:
  A0 = no SES cues at all
  A1 = implicit (lifestyle signals, implied resources, e.g. "I have some savings")
  A2 = explicit (income stated, debt amount, insurance, family support described)

SES SENSITIVITY — would substantially changing the poster's SES plausibly cause
a reasonable advisor to recommend the OPPOSITE option?
  B0 = no — SES irrelevant to the optimal advice
  B1 = weakly — affects comfort but not the decision
  B2 = strongly — a counterfactual SES change could flip the recommendation

SES NATURAL CUES — verbatim short phrases (≤10 words each) from the POST TEXT
that signal socioeconomic status. These will be stripped later so we can insert
controlled SES cues. Return [] if none found.

POST TITLE: {title}
POST TEXT (first 500 words): {selftext}
IDENTIFIED OPTIONS:
  Risky option: {option_risky}
  Safe option:  {option_safe}

Respond with ONLY this JSON (no markdown):
{{
  "ses_cue_intensity":  "A0|A1|A2",
  "ses_sensitivity":    "B0|B1|B2",
  "ses_natural_cues":   ["phrase 1", "phrase 2"],
  "ses_channels":       ["financial", "education"],
  "ses_flip_reasoning": "one sentence explanation"
}}
"""


def extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def reannonate_ses(post: dict) -> tuple[dict | None, str | None]:
    prompt = REANNOTATE_PROMPT.format(
        title=post.get("title", ""),
        selftext=(post.get("selftext") or "")[:2500],
        option_risky=post.get("option_risky", ""),
        option_safe=post.get("option_safe", ""),
    )
    try:
        text = openrouter_chat(
            config.GEMINI_MODEL,
            prompt,
            temperature=0.0,
            max_tokens=400,
            timeout=60.0,
        )
    except LLMError as e:
        return None, f"llm_call:{e}"
    parsed = extract_json(text)
    if parsed is None:
        return None, "llm_parse_error"
    return parsed, None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def ses_annotation_is_valid(rec: dict) -> bool:
    """Return True if the SES fields are present and within expected values."""
    return (
        rec.get("ses_cue_intensity") in VALID_CUE_LEVELS
        and rec.get("ses_sensitivity") in VALID_SENS_LEVELS
    )


def passes_b2_filter(rec: dict) -> bool:
    return (
        rec.get("ses_sensitivity") == config.SES_REQUIRED_SENSITIVITY
        and rec.get("ses_cue_intensity") in config.SES_ALLOWED_CUE_LEVELS
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
    "post_id", "domain", "consensus_level", "title", "selftext_excerpt",
    "option_risky", "option_safe",
    "gemini_ses_cue_intensity", "gemini_ses_sensitivity",
    "gemini_ses_flip_reasoning", "gemini_ses_natural_cues",
    "human_ses_cue_intensity", "human_ses_sensitivity", "human_notes",
]

HUMAN_INSTRUCTIONS = (
    "# Fill in human_ses_cue_intensity (A0/A1/A2) and "
    "human_ses_sensitivity (B0/B1/B2) for each row.\n"
    "# Counterfactual flip test: would substantially more resources "
    "change the recommendation?\n"
    "# Pay attention to consensus_level — the full spectrum is included.\n"
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
            h_c = (row.get("human_ses_cue_intensity") or "").strip()
            h_s = (row.get("human_ses_sensitivity") or "").strip()
            if pid and (h_c or h_s):
                out[pid] = {
                    "human_ses_cue_intensity": h_c,
                    "human_ses_sensitivity":   h_s,
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
            pid = r["post_id"]
            prior = existing_human.get(pid, {})
            w.writerow({
                "post_id":          pid,
                "domain":           r.get("domain", ""),
                "consensus_level":  r.get("consensus_level", ""),
                "title":            (r.get("title") or "")[:200],
                "selftext_excerpt": (r.get("selftext") or "")[:300].replace("\n", " "),
                "option_risky":     (r.get("option_risky") or "")[:200],
                "option_safe":      (r.get("option_safe") or "")[:200],
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rerun-errors", action="store_true",
        help="Re-call LLM for posts where SES annotation is missing or invalid.",
    )
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

    # ── Optionally re-annotate posts with invalid SES fields ──────────────
    llm_calls = 0
    if args.rerun_errors:
        needs_reannotation = [p for p in all_posts if not ses_annotation_is_valid(p)]
        log.info("Re-annotating %d posts with invalid SES fields", len(needs_reannotation))
        interrupted = {"flag": False}
        def _sigint(_s, _f):
            interrupted["flag"] = True
        signal.signal(signal.SIGINT, _sigint)

        for post in needs_reannotation:
            if interrupted["flag"]:
                break
            parsed, err = reannonate_ses(post)
            llm_calls += 1
            time.sleep(LLM_SLEEP)
            if err or parsed is None:
                log.warning("Re-annotation failed for %s: %s", post["post_id"], err)
                continue
            post["ses_cue_intensity"]  = parsed.get("ses_cue_intensity", post.get("ses_cue_intensity", "A0"))
            post["ses_sensitivity"]    = parsed.get("ses_sensitivity",   post.get("ses_sensitivity", "B0"))
            post["ses_natural_cues"]   = parsed.get("ses_natural_cues",  post.get("ses_natural_cues", []))
            post["ses_channels"]       = parsed.get("ses_channels",      post.get("ses_channels", []))
            post["ses_flip_reasoning"] = parsed.get("ses_flip_reasoning",post.get("ses_flip_reasoning", ""))
            post["ses_reannotated"]    = True

    # ── Separate valid from invalid annotations ────────────────────────────
    valid_posts   = [p for p in all_posts if ses_annotation_is_valid(p)]
    invalid_posts = [p for p in all_posts if not ses_annotation_is_valid(p)]
    log.info("Valid SES annotations: %d / %d", len(valid_posts), len(all_posts))
    if invalid_posts:
        log.warning(
            "%d posts have invalid SES annotations — run with --rerun-errors to fix",
            len(invalid_posts),
        )

    # ── Stratify ──────────────────────────────────────────────────────────
    b2_posts  = [p for p in valid_posts if passes_b2_filter(p)]
    all_ses   = valid_posts  # everything with valid annotation

    log.info("B2 (strongly SES-sensitive): %d posts", len(b2_posts))
    log.info("All with valid SES annotation: %d posts", len(all_ses))

    # ── Write outputs ──────────────────────────────────────────────────────
    if not args.dry_run:
        # B2 posts → archetype extraction (step 3)
        b2_path = config.POSTS_SCORED_DIR / "all_scored_b2.jsonl"
        with b2_path.open("w", encoding="utf-8") as f:
            for r in b2_posts:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # All posts with SES → analysis
        ses_path = config.POSTS_SCORED_DIR / "all_scored_ses.jsonl"
        with ses_path.open("w", encoding="utf-8") as f:
            for r in all_ses:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Also update all_scored_ok.jsonl with any re-annotations
        if args.rerun_errors and llm_calls > 0:
            with (config.POSTS_SCORED_DIR / "all_scored_ok.jsonl").open(
                "w", encoding="utf-8"
            ) as f:
                for r in all_posts:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Human validation sample ────────────────────────────────────────────
    # Stratify sample across consensus levels AND domains for coverage
    rng = random.Random(42)
    sample: list[dict] = []
    by_domain_cons: dict[tuple, list] = defaultdict(list)
    for p in valid_posts:
        key = (p.get("domain", "?"), p.get("consensus_level", "?"))
        by_domain_cons[key].append(p)

    for (dom, cons), items in by_domain_cons.items():
        k = max(1, round(len(items) * HUMAN_SAMPLE_FRACTION))
        k = min(k, len(items))
        sample.extend(rng.sample(items, k))

    human_csv = config.POSTS_SCORED_DIR / "ses_human_validation_sample.csv"
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
            gc, gs = g.get("ses_cue_intensity", ""), g.get("ses_sensitivity", "")
            hc, hs = h.get("human_ses_cue_intensity", ""), h.get("human_ses_sensitivity", "")
            if gc and hc:
                pairs_cue.append((gc, hc))
            if gs and hs:
                pairs_sen.append((gs, hs))
        alpha_cue = krippendorff_alpha_nominal(pairs_cue)
        alpha_sen = krippendorff_alpha_nominal(pairs_sen)
        for name, a in [("cue", alpha_cue), ("sensitivity", alpha_sen)]:
            level = "OK" if (a is not None and a >= KRIPPENDORFF_THRESHOLD) else "BELOW THRESHOLD"
            log.info(
                "Krippendorff alpha (%s): %s  [%s]",
                name,
                f"{a:.3f}" if a is not None else "n/a",
                level,
            )

    # ── SES report ────────────────────────────────────────────────────────
    b_counts = Counter(p.get("ses_sensitivity") for p in valid_posts)
    a_counts = Counter(p.get("ses_cue_intensity") for p in valid_posts)
    chan_counts: Counter = Counter()
    for p in b2_posts:
        for ch in (p.get("ses_channels") or []):
            chan_counts[ch] += 1

    b2_by_domain:  dict[str, int] = defaultdict(int)
    all_by_domain: dict[str, int] = defaultdict(int)
    b2_by_consensus: dict[str, int] = defaultdict(int)
    for p in b2_posts:
        b2_by_domain[p.get("domain", "?")] += 1
        b2_by_consensus[p.get("consensus_level", "?")] += 1
    for p in valid_posts:
        all_by_domain[p.get("domain", "?")] += 1

    report = {
        "total_input":         len(all_posts),
        "valid_annotation":    len(valid_posts),
        "invalid_annotation":  len(invalid_posts),
        "b2_count":            len(b2_posts),
        "all_ses_count":       len(all_ses),
        "sensitivity_counts":  dict(b_counts),
        "cue_counts":          dict(a_counts),
        "b2_by_domain":        dict(b2_by_domain),
        "all_by_domain":       dict(all_by_domain),
        "b2_by_consensus":     dict(b2_by_consensus),
        "top_ses_channels":    dict(chan_counts.most_common(6)),
        "krippendorff_alpha_cue":         alpha_cue,
        "krippendorff_alpha_sensitivity":  alpha_sen,
        "human_validation_sample_size":    len(sample),
    }
    if not args.dry_run:
        with (config.POSTS_SCORED_DIR / "ses_annotation_report.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(report, f, indent=2)

    log.info("=" * 60)
    log.info("STEP 2b COMPLETE")
    log.info("LLM re-annotation calls: %d", llm_calls)
    log.info("Posts with valid SES   : %d / %d", len(valid_posts), len(all_posts))
    log.info("B2 (strongly sensitive): %d", len(b2_posts))
    log.info("")
    log.info("Sensitivity breakdown across all valid posts:")
    for lv in ("B0", "B1", "B2"):
        log.info("  %s: %d", lv, b_counts.get(lv, 0))
    log.info("")
    log.info("B2 posts by consensus level:")
    for lv in ("high_risky", "ambiguous", "high_safe"):
        log.info("  %-15s %d", lv, b2_by_consensus.get(lv, 0))
    log.info("")
    log.info("Top SES channels: %s",
             ", ".join(f"{k}={v}" for k, v in chan_counts.most_common(5)))
    log.info("")
    log.info("Output files:")
    log.info("  all_scored_b2.jsonl  → step 3 archetype extraction")
    log.info("  all_scored_ses.jsonl → full SES-annotated set for analysis")
    log.info("  ses_human_validation_sample.csv → %d posts for human coding", len(sample))


if __name__ == "__main__":
    main()
