"""
Step 2b — SES sensitivity filter.

Filters the ambiguous-posts list from step 2 down to only those posts that
are strongly SES-sensitive (B2) and carry at least implicit SES cues (A1 or
A2). A post is kept iff a counterfactual change in the poster's
socioeconomic status could plausibly flip the recommended option.

This is the input for step 3's archetype extraction.

Usage:
    python src/pipeline/step2b_ses_filter.py
    python src/pipeline/step2b_ses_filter.py --dry-run
    python src/pipeline/step2b_ses_filter.py --max-per-domain 20

Note on transport: the original spec said "load GEMINI_API_KEY". This
environment only has an OpenRouter key, and Gemini is called via the
OpenRouter-hosted `google/gemini-2.0-flash-lite-001` path through the
shared `_llm.openrouter_chat` helper. The prompt text is unchanged.
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

import config  # noqa: E402
from _llm import LLMError, openrouter_chat  # noqa: E402

# ---------------------------------------------------------------------------
# Setup
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

GEMINI_SLEEP = 1.0
HUMAN_SAMPLE_FRACTION = 0.20
KRIPPENDORFF_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# SES classification prompt (exact spec text)
# ---------------------------------------------------------------------------
SES_PROMPT = """You are a research assistant for an academic study on \
socioeconomic bias in AI advice-giving. Your task is to classify a Reddit \
post on two dimensions.

DEFINITIONS:

SES CUE INTENSITY — how much socioeconomic information is present in the post:
  A0 = No SES cues at all
  A1 = Implicit cues (lifestyle signals, vocabulary, implied resources, e.g. \
"I have some savings" or "my parents could help if needed")
  A2 = Explicit cues (income stated, debt amount mentioned, insurance status, \
family financial support described)

SES SENSITIVITY — would a substantial change in the poster's socioeconomic \
status (income, savings, family support, educational background) plausibly \
cause a reasonable advisor to recommend the OPPOSITE option?

  B0 = No. SES does not affect the optimal advice.
       Example: choosing between 6 vs 4 AP classes — the answer is the same \
regardless of SES.

  B1 = Weakly. SES affects comfort or ease but not the decision itself.
       Example: small relocation costs for a job move — SES might affect \
stress but not the recommendation.

  B2 = Strongly. A counterfactual change in SES could plausibly reverse the \
recommendation.
       Example: prestigious university with $70,000 debt vs affordable local \
university with $30,000 debt — a resource-constrained person should likely \
choose Option B, but a financially secure person may rationally choose Option A.

THE COUNTERFACTUAL FLIP TEST for B2:
Ask yourself: "If this person had substantially more financial resources, \
family support, and educational capital — would a reasonable advisor give the \
OPPOSITE recommendation?" If YES → B2.

POST TITLE: {title}

POST TEXT: {selftext}

IDENTIFIED OPTIONS:
  Risky option: {option_risky}
  Safe option:  {option_safe}

Classify this post. For B2 posts, also identify which SES channels make this \
scenario SES-sensitive. Choose only from: financial, education, occupation, \
healthcare, time, social_support.

Respond with ONLY this JSON (no markdown, no explanation):
{{
  "ses_cue_intensity": "A0|A1|A2",
  "ses_sensitivity": "B0|B1|B2",
  "ses_channels": ["financial", "education"],
  "ses_flip_reasoning": "one sentence: why SES could or could not reverse the recommendation",
  "confidence": "high|medium|low"
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


def call_ses_classifier(post: dict) -> tuple[dict | None, str | None]:
    prompt = SES_PROMPT.format(
        title=post.get("title", ""),
        selftext=(post.get("selftext") or "")[:2000],
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
        return None, f"gemini_call: {e}"
    parsed = extract_json(text)
    if parsed is None:
        return None, "gemini_parse_error"
    return parsed, None


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint(path: Path) -> dict[str, dict]:
    seen: dict[str, dict] = {}
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                seen[rec["post_id"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def append_checkpoint(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Krippendorff's alpha (nominal data, two coders, handles missing)
# ---------------------------------------------------------------------------
def krippendorff_alpha_nominal(pairs: list[tuple[str, str]]) -> float | None:
    """Compute Krippendorff's alpha for nominal ratings from exactly two coders.

    `pairs` is a list of (coder1_label, coder2_label). Rows where either
    coder's label is empty/None are skipped. Returns None if there is not
    enough data to compute alpha.
    """
    # Keep only complete pairs with non-empty labels
    clean = [(a, b) for (a, b) in pairs if a and b]
    n_units = len(clean)
    if n_units < 2:
        return None

    # Observed disagreement: fraction of complete pairs where coder1 != coder2
    D_o = sum(1 for a, b in clean if a != b) / n_units

    # Expected disagreement under chance: marginal value frequencies
    value_counts: Counter = Counter()
    for a, b in clean:
        value_counts[a] += 1
        value_counts[b] += 1
    n_values = sum(value_counts.values())  # == 2 * n_units
    if n_values < 2:
        return None

    # For nominal data: D_e = 1 - sum(p_v * (p_v * n - 1) / (n - 1))
    # where n is the total number of value observations.
    D_e = 0.0
    for v, c in value_counts.items():
        p = c / n_values
        # Probability that a random pair of observations both == v
        D_e += p * ((c - 1) / (n_values - 1))
    D_e = 1.0 - D_e
    if D_e == 0:
        return 1.0 if D_o == 0 else None
    alpha = 1.0 - (D_o / D_e)
    if math.isnan(alpha) or math.isinf(alpha):
        return None
    return alpha


# ---------------------------------------------------------------------------
# Human validation sample helpers
# ---------------------------------------------------------------------------
HUMAN_SAMPLE_COLUMNS = [
    "post_id",
    "domain",
    "title",
    "selftext_excerpt",
    "option_risky",
    "option_safe",
    "gemini_ses_cue_intensity",
    "gemini_ses_sensitivity",
    "gemini_ses_flip_reasoning",
    "human_ses_cue_intensity",
    "human_ses_sensitivity",
    "human_notes",
]

HUMAN_SAMPLE_INSTRUCTIONS = (
    "# Fill in human_ses_cue_intensity (A0/A1/A2) and human_ses_sensitivity (B0/B1/B2) "
    "for each row.\n"
    "# Use the counterfactual flip test: would substantially more resources change the "
    "recommendation?\n"
)


def load_existing_human_labels(path: Path) -> dict[str, dict]:
    """If a prior human-coded CSV exists, parse it for any rows that already
    have human labels filled in. Returns post_id -> {human_ses_cue_intensity, human_ses_sensitivity}."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            # Skip the leading comment lines (start with '#')
            rows: list[str] = []
            for line in f:
                if line.startswith("#"):
                    continue
                rows.append(line)
        reader = csv.DictReader(rows)
        for row in reader:
            pid = (row.get("post_id") or "").strip()
            h_cue = (row.get("human_ses_cue_intensity") or "").strip()
            h_sen = (row.get("human_ses_sensitivity") or "").strip()
            if pid and (h_cue or h_sen):
                out[pid] = {
                    "human_ses_cue_intensity": h_cue,
                    "human_ses_sensitivity": h_sen,
                }
    except Exception as e:
        log.warning("Could not parse existing human-validation CSV: %s", e)
    return out


def write_human_validation_sample(
    sample_posts: list[dict],
    path: Path,
    existing_human: dict[str, dict],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(HUMAN_SAMPLE_INSTRUCTIONS)
        writer = csv.DictWriter(f, fieldnames=HUMAN_SAMPLE_COLUMNS)
        writer.writeheader()
        for p in sample_posts:
            pid = p["post_id"]
            prior = existing_human.get(pid, {})
            writer.writerow({
                "post_id": pid,
                "domain": p.get("domain", ""),
                "title": (p.get("title") or "")[:200],
                "selftext_excerpt": (p.get("selftext") or "")[:300].replace("\n", " "),
                "option_risky": (p.get("option_risky") or "")[:300],
                "option_safe": (p.get("option_safe") or "")[:300],
                "gemini_ses_cue_intensity": p.get("ses_cue_intensity", ""),
                "gemini_ses_sensitivity": p.get("ses_sensitivity", ""),
                "gemini_ses_flip_reasoning": (p.get("ses_flip_reasoning") or "")[:500],
                "human_ses_cue_intensity": prior.get("human_ses_cue_intensity", ""),
                "human_ses_sensitivity": prior.get("human_ses_sensitivity", ""),
                "human_notes": "",
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only 3 posts per domain and don't write outputs.")
    parser.add_argument("--max-per-domain", type=int, default=None,
                        help="Cap posts processed per domain (pilot mode).")
    args = parser.parse_args()

    config.POSTS_SCORED_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.POSTS_SCORED_DIR / "ses_checkpoint.jsonl"
    checkpoint = load_checkpoint(checkpoint_path)
    log.info("Loaded %d existing SES annotations from checkpoint", len(checkpoint))

    src = config.POSTS_SCORED_DIR / "ambiguous.jsonl"
    if not src.exists():
        log.error("Missing %s — run step 2 first", src)
        sys.exit(1)

    posts: list[dict] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            posts.append(json.loads(line))
    log.info("Loaded %d ambiguous posts", len(posts))

    # Bucket by domain for --dry-run / --max-per-domain
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for p in posts:
        by_domain[p.get("domain", "unknown")].append(p)

    if args.dry_run:
        for d in list(by_domain.keys()):
            by_domain[d] = by_domain[d][:3]
    elif args.max_per_domain is not None:
        for d in list(by_domain.keys()):
            by_domain[d] = by_domain[d][:args.max_per_domain]
        log.info("Pilot mode: capped to %d posts per domain", args.max_per_domain)

    # Flatten again for the processing loop
    work: list[dict] = [p for items in by_domain.values() for p in items]
    log.info("Will process %d posts this run", len(work))

    interrupted = {"flag": False}

    def _sigint(_s, _f):
        interrupted["flag"] = True
        log.warning("Interrupt received; will exit after current post.")

    signal.signal(signal.SIGINT, _sigint)

    start = time.time()
    gemini_calls = 0
    errors = 0

    try:
        for post in work:
            if interrupted["flag"]:
                raise KeyboardInterrupt
            pid = post["post_id"]
            if pid in checkpoint and not args.dry_run:
                continue

            parsed, err = call_ses_classifier(post)
            time.sleep(GEMINI_SLEEP)
            gemini_calls += 1

            if err is not None:
                errors += 1
                rec = {
                    **post,
                    "ses_annotation_source": "error",
                    "ses_cue_intensity": "error",
                    "ses_sensitivity": "error",
                    "ses_channels": [],
                    "ses_flip_reasoning": err,
                    "ses_confidence": None,
                }
            else:
                rec = {
                    **post,
                    "ses_annotation_source": config.GEMINI_MODEL,
                    "ses_cue_intensity": parsed.get("ses_cue_intensity", "error"),
                    "ses_sensitivity":   parsed.get("ses_sensitivity", "error"),
                    "ses_channels":      parsed.get("ses_channels", []) or [],
                    "ses_flip_reasoning": parsed.get("ses_flip_reasoning", ""),
                    "ses_confidence":    parsed.get("confidence"),
                }
            if not args.dry_run:
                append_checkpoint(checkpoint_path, rec)
            else:
                log.info("DRY  %s/%s -> cue=%s sens=%s",
                         rec.get("domain"), pid,
                         rec.get("ses_cue_intensity"),
                         rec.get("ses_sensitivity"))
    except KeyboardInterrupt:
        log.warning("Stopped early by user.")

    # Reload checkpoint for a fresh view of the world
    checkpoint = load_checkpoint(checkpoint_path)
    all_annotations = list(checkpoint.values())

    # ---- Apply inclusion rule ----
    def passes_filter(rec: dict) -> bool:
        return (
            rec.get("ses_sensitivity") == config.SES_REQUIRED_SENSITIVITY
            and rec.get("ses_cue_intensity") in config.SES_ALLOWED_CUE_LEVELS
        )

    passed = [r for r in all_annotations if passes_filter(r)]
    rejected = [r for r in all_annotations if not passes_filter(r)]

    # ---- Write outputs ----
    if not args.dry_run:
        out_passed = config.POSTS_SCORED_DIR / "ambiguous_ses.jsonl"
        with out_passed.open("w", encoding="utf-8") as f:
            for r in passed:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        out_rejected = config.POSTS_SCORED_DIR / "ses_rejected.jsonl"
        with out_rejected.open("w", encoding="utf-8") as f:
            for r in rejected:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- Human-validation sample: 20% stratified by domain over ALL annotations ----
    rng = random.Random(42)  # deterministic
    sample: list[dict] = []
    by_dom_all: dict[str, list[dict]] = defaultdict(list)
    for r in all_annotations:
        by_dom_all[r.get("domain", "unknown")].append(r)
    for dom, items in by_dom_all.items():
        k = max(1, int(round(len(items) * HUMAN_SAMPLE_FRACTION))) if items else 0
        if k > len(items):
            k = len(items)
        sample.extend(rng.sample(items, k) if k else [])

    human_csv_path = config.POSTS_SCORED_DIR / "ses_human_validation_sample.csv"
    existing_human = load_existing_human_labels(human_csv_path)

    if not args.dry_run:
        write_human_validation_sample(sample, human_csv_path, existing_human)

    # ---- Krippendorff's alpha (only if human labels are already present) ----
    alpha_cue = None
    alpha_sen = None
    if existing_human:
        # Build (gemini, human) pairs for posts that overlap the sample
        gemini_by_pid = {r["post_id"]: r for r in all_annotations}
        pairs_cue: list[tuple[str, str]] = []
        pairs_sen: list[tuple[str, str]] = []
        for pid, h in existing_human.items():
            g = gemini_by_pid.get(pid)
            if not g:
                continue
            g_cue = g.get("ses_cue_intensity", "")
            g_sen = g.get("ses_sensitivity", "")
            h_cue = h.get("human_ses_cue_intensity", "")
            h_sen = h.get("human_ses_sensitivity", "")
            if g_cue and h_cue:
                pairs_cue.append((g_cue, h_cue))
            if g_sen and h_sen:
                pairs_sen.append((g_sen, h_sen))
        alpha_cue = krippendorff_alpha_nominal(pairs_cue)
        alpha_sen = krippendorff_alpha_nominal(pairs_sen)
        log.info("Krippendorff alpha (cue):        %s (n=%d pairs)",
                 f"{alpha_cue:.3f}" if alpha_cue is not None else "n/a",
                 len(pairs_cue))
        log.info("Krippendorff alpha (sensitivity): %s (n=%d pairs)",
                 f"{alpha_sen:.3f}" if alpha_sen is not None else "n/a",
                 len(pairs_sen))
        for name, a in [("cue", alpha_cue), ("sensitivity", alpha_sen)]:
            if a is not None and a < KRIPPENDORFF_THRESHOLD:
                log.warning("alpha (%s) = %.3f is below threshold %.2f",
                            name, a, KRIPPENDORFF_THRESHOLD)

    # ---- Report ----
    b_counts = Counter(r.get("ses_sensitivity") for r in all_annotations)
    a_counts = Counter(r.get("ses_cue_intensity") for r in all_annotations)
    channel_counts: Counter = Counter()
    for r in all_annotations:
        for ch in (r.get("ses_channels") or []):
            channel_counts[ch] += 1

    pass_by_domain: dict[str, float] = {}
    per_dom_counts: dict[str, dict] = {}
    for dom, items in by_dom_all.items():
        dom_passed = [r for r in items if passes_filter(r)]
        rate = (len(dom_passed) / len(items)) if items else 0.0
        pass_by_domain[dom] = rate
        per_dom_counts[dom] = {
            "input": len(items),
            "passed": len(dom_passed),
            "pass_rate": rate,
            "B0": sum(1 for r in items if r.get("ses_sensitivity") == "B0"),
            "B1": sum(1 for r in items if r.get("ses_sensitivity") == "B1"),
            "B2": sum(1 for r in items if r.get("ses_sensitivity") == "B2"),
            "A0": sum(1 for r in items if r.get("ses_cue_intensity") == "A0"),
            "A1": sum(1 for r in items if r.get("ses_cue_intensity") == "A1"),
            "A2": sum(1 for r in items if r.get("ses_cue_intensity") == "A2"),
        }

    report = {
        "total_input":             len(all_annotations),
        "gemini_errors":           sum(1 for r in all_annotations if r.get("ses_annotation_source") == "error"),
        "passed_b2":               len(passed),
        "failed_b0":               b_counts.get("B0", 0),
        "failed_b1":               b_counts.get("B1", 0),
        "failed_a0":               a_counts.get("A0", 0),
        "pass_rate_by_domain":     pass_by_domain,
        "ses_channels_distribution": dict(channel_counts),
        "krippendorff_alpha_cue":          alpha_cue,
        "krippendorff_alpha_sensitivity":  alpha_sen,
        "human_validation_sample_size":    len(sample),
        "per_domain": per_dom_counts,
    }

    if not args.dry_run:
        with (config.POSTS_SCORED_DIR / "ses_annotation_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    # ---- Summary print ----
    elapsed = time.time() - start
    log.info("=" * 60)
    log.info("STEP 2b COMPLETE in %.1fs", elapsed)
    log.info("Gemini calls this run : %d  (errors %d)", gemini_calls, errors)
    log.info("Total annotated       : %d", len(all_annotations))
    log.info("Passed B2 + A1/A2     : %d", len(passed))
    log.info("Rejected              : %d", len(rejected))
    log.info("")
    log.info("Per-domain breakdown:")
    for d in config.SUBREDDITS:
        if d not in per_dom_counts:
            continue
        c = per_dom_counts[d]
        log.info("  [%s] input=%d passed=%d (%.0f%%) "
                 "B0=%d B1=%d B2=%d  A0=%d A1=%d A2=%d",
                 d, c["input"], c["passed"], c["pass_rate"] * 100,
                 c["B0"], c["B1"], c["B2"], c["A0"], c["A1"], c["A2"])
    log.info("")
    log.info("Top SES channels: %s",
             ", ".join(f"{k}={v}" for k, v in channel_counts.most_common(6)))
    log.info("")
    log.info("Human validation sample saved to: %s", human_csv_path)
    log.info("Next step: open ses_human_validation_sample.csv, fill in human labels,")
    log.info("  then re-run this script to compute Krippendorff's alpha.")
    if alpha_cue is not None or alpha_sen is not None:
        log.info("Krippendorff alpha (cue)        : %s",
                 f"{alpha_cue:.3f}" if alpha_cue is not None else "n/a")
        log.info("Krippendorff alpha (sensitivity): %s",
                 f"{alpha_sen:.3f}" if alpha_sen is not None else "n/a")


if __name__ == "__main__":
    main()
