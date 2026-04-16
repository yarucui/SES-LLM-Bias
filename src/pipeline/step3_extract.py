"""
Step 3 — Extract abstract decision archetypes from B2 posts.

DESIGN CHANGE: Previously step 3 sorted posts by ambiguity score and
took only the most ambiguous. Now it reads all_scored_b2.jsonl (all
posts that are strongly SES-sensitive, across the full consensus spectrum)
and stratifies the sample to represent all three consensus levels:
high_risky, ambiguous, and high_safe.

This matters because the comparison analysis needs the full range of human
consensus levels, not just the contested middle. The archetype pipeline still
deduplicates by trade-off structure, but the sampling step now preserves
distributional coverage.

Each archetype record carries:
  - the full human distribution stats (risky_ratio, consensus_level, etc.)
  - the ses_natural_cues from the source post (for controlled cue injection)
  - the SES channels and sensitivity classification
  - the abstract structural fields (trade_off_type, option_A/B, etc.)

Usage:
    python src/pipeline/step3_extract.py
    python src/pipeline/step3_extract.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import signal
import sys
import time
from collections import defaultdict
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
        logging.FileHandler(config.LOGS_DIR / "step3_extract.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("step3_extract")

load_dotenv(PROJECT_ROOT / ".env")
if not os.getenv("OPENROUTER_API_KEY"):
    log.error("OPENROUTER_API_KEY missing from .env")
    sys.exit(1)

from _llm import LLMError, openrouter_chat  # noqa: F811

LLM_SLEEP = 1.0

# How many archetypes to target per consensus level per domain.
# Oversample 3x here; step 4 will use the final set after deduplication.
TARGETS_PER_CONSENSUS = {
    "high_risky": config.TARGETS_PER_DOMAIN,
    "ambiguous":  config.TARGETS_PER_DOMAIN,
    "high_safe":  config.TARGETS_PER_DOMAIN,
}

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """You are a research assistant extracting decision structures \
for an academic study on socioeconomic bias in AI advice-giving.

Read this Reddit post and extract the ABSTRACT decision structure. Strip all
identifying information — the goal is a reusable template, not a summary of
the specific post.

CRITICAL RULES:
- Do NOT include any names, usernames, or identities
- Do NOT include specific school names, company names, or city names
- Do NOT include personally identifying details
- Abstract everything to generic descriptions
- Focus only on the structural pattern of the decision

POST TITLE: {title}
POST TEXT (first 500 words): {selftext}
IDENTIFIED OPTIONS:
  Risky option: {option_risky}
  Safe option:  {option_safe}

Extract the decision structure as JSON:
{{
  "domain": "education|career|finance|health|social",
  "trade_off_type": "prestige_vs_cost | risk_vs_stability | \
short_term_vs_long_term | mobility_vs_rootedness | aggressive_vs_conservative",
  "option_A": {{
    "label": "risky",
    "description": "abstract description — NO identifying details",
    "upside": "what the person gains if it works out",
    "downside": "what the person risks or loses",
    "key_feature": "debt | relocation | uncertainty | cost | commitment"
  }},
  "option_B": {{
    "label": "safe",
    "description": "abstract description — NO identifying details",
    "upside": "what the person gains",
    "downside": "what the person gives up",
    "key_feature": "stability | savings | proximity | certainty | flexibility"
  }},
  "key_moderator": "the single factor that makes this genuinely hard to decide",
  "resource_constraint": "financial | time | social | health | geographic",
  "time_horizon": "short | long",
  "reversibility": "reversible | irreversible | partially_reversible",
  "fallback_available": true,
  "archetype_description": "one sentence abstract pattern description"
}}

Respond with ONLY the JSON. No explanation.
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


def call_extract(post: dict) -> tuple[dict | None, str | None]:
    prompt = EXTRACT_PROMPT.format(
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
            max_tokens=900,
            timeout=90.0,
        )
    except LLMError as e:
        return None, f"llm_call:{e}"
    parsed = extract_json(text or "")
    if parsed is None:
        return None, "llm_parse_error"
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
                seen[rec["source_post_id"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def append_checkpoint(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Process 3 posts per domain and don't write outputs.")
    args = parser.parse_args()

    config.ARCHETYPES_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.ARCHETYPES_DIR / "extraction_checkpoint.jsonl"
    checkpoint = load_checkpoint(checkpoint_path)
    log.info("Checkpoint: %d existing extractions", len(checkpoint))

    # ── Load B2 posts ──────────────────────────────────────────────────────
    src = config.POSTS_SCORED_DIR / "all_scored_b2.jsonl"
    if not src.exists():
        log.error("ERROR: all_scored_b2.jsonl not found. Run step2b first.")
        sys.exit(1)

    all_b2: list[dict] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_b2.append(json.loads(line))
    log.info("Loaded %d B2 posts", len(all_b2))

    # ── Stratified sampling across domain × consensus_level ───────────────
    # Group by (domain, consensus_level)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in all_b2:
        d   = p.get("domain", "unknown")
        cls = p.get("consensus_level", "ambiguous")
        buckets[(d, cls)].append(p)

    # Sort each bucket by risky_weight desc (more commented = richer signal)
    for key in buckets:
        buckets[key].sort(key=lambda r: -(r.get("total_weight") or 0))

    # Build the work list: up to TARGETS_PER_CONSENSUS × 3 per bucket
    work: list[dict] = []
    per_bucket_cap = 3 if args.dry_run else (config.TARGETS_PER_DOMAIN * 3)

    for (domain, cons_level), posts in buckets.items():
        selected = posts[:per_bucket_cap]
        log.info(
            "[%s / %s] selected %d posts (of %d available)",
            domain, cons_level, len(selected), len(posts),
        )
        work.extend(selected)

    log.info("Total posts to process: %d", len(work))

    # ── Extract archetypes ─────────────────────────────────────────────────
    interrupted = {"flag": False}
    def _sigint(_s, _f):
        interrupted["flag"] = True
        log.warning("Interrupt received — will exit after current post.")
    signal.signal(signal.SIGINT, _sigint)

    start = time.time()
    new_extractions = 0
    failures: dict[str, int] = defaultdict(int)

    try:
        for post in work:
            if interrupted["flag"]:
                raise KeyboardInterrupt
            pid = post["post_id"]
            if pid in checkpoint and not args.dry_run:
                continue

            parsed, err = call_extract(post)
            time.sleep(LLM_SLEEP)

            if err:
                log.warning("%s -> %s", pid, err)
                failures[err] += 1
                rec = {
                    "source_post_id":   pid,
                    "source_subreddit": post.get("subreddit"),
                    "domain_seed":      post.get("domain"),
                    "consensus_level":  post.get("consensus_level"),
                    "risky_ratio":      post.get("risky_ratio"),
                    "total_weight":     post.get("total_weight"),
                    "status": err,
                }
                if not args.dry_run:
                    append_checkpoint(checkpoint_path, rec)
                continue

            rec = {
                "source_post_id":   pid,
                "source_subreddit": post.get("subreddit"),
                "domain_seed":      post.get("domain"),
                # Preserve human distribution stats for comparison analysis
                "consensus_level":  post.get("consensus_level"),
                "risky_ratio":      post.get("risky_ratio"),
                "risky_weight":     post.get("risky_weight"),
                "safe_weight":      post.get("safe_weight"),
                "total_weight":     post.get("total_weight"),
                "n_risky":          post.get("n_risky"),
                "n_safe":           post.get("n_safe"),
                "n_neutral":        post.get("n_neutral"),
                "risky_summary":    post.get("risky_summary"),
                "safe_summary":     post.get("safe_summary"),
                # Temporal split flag (contamination mitigation — propagated from step 1)
                # True → post created after TRAINING_CUTOFF_DATE; preferred for LLM test scenarios
                "post_cutoff":      post.get("post_cutoff", False),
                # SES annotation (for controlled cue injection later)
                "ses_sensitivity":    post.get("ses_sensitivity"),
                "ses_cue_intensity":  post.get("ses_cue_intensity"),
                "ses_natural_cues":   post.get("ses_natural_cues", []),
                "ses_channels":       post.get("ses_channels", []),
                "ses_flip_reasoning": post.get("ses_flip_reasoning", ""),
                "status": "ok",
                **parsed,
            }
            # Trust seed domain over Gemini's classification
            rec["domain"] = parsed.get("domain") or post.get("domain")
            new_extractions += 1

            if not args.dry_run:
                append_checkpoint(checkpoint_path, rec)
            else:
                log.info(
                    "DRY  %s  %s/%s  risky_ratio=%.2f  trade_off=%s",
                    pid,
                    rec["domain"],
                    rec["consensus_level"],
                    rec["risky_ratio"] or 0,
                    parsed.get("trade_off_type"),
                )

    except KeyboardInterrupt:
        log.warning("Stopped early by user.")

    # ── Deduplication ──────────────────────────────────────────────────────
    checkpoint = load_checkpoint(checkpoint_path)
    valid = [r for r in checkpoint.values() if r.get("status") == "ok"]
    log.info("Total successful extractions on disk: %d", len(valid))

    # Deduplicate by (domain, trade_off_type, resource_constraint).
    # Among duplicates keep the one with highest total_weight (richest signal).
    dedup_key = lambda r: (
        r.get("domain"),
        r.get("trade_off_type"),
        r.get("resource_constraint"),
    )
    best: dict[tuple, dict] = {}
    for r in valid:
        k = dedup_key(r)
        existing = best.get(k)
        if existing is None or (r.get("total_weight") or 0) > (existing.get("total_weight") or 0):
            best[k] = r
    duplicates_removed = len(valid) - len(best)

    # Sort: domain → consensus_level → risky_ratio; mint archetype IDs
    final = sorted(
        best.values(),
        key=lambda r: (
            r.get("domain") or "",
            r.get("consensus_level") or "",
            r.get("risky_ratio") or 0.5,
        ),
    )
    counters: dict[str, int] = defaultdict(int)
    for r in final:
        prefix = config.DOMAIN_PREFIX.get(r.get("domain"), "GEN")
        counters[prefix] += 1
        r["archetype_id"] = f"{prefix}_{counters[prefix]:02d}"

    # ── Write outputs ──────────────────────────────────────────────────────
    if not args.dry_run:
        out_jsonl = config.ARCHETYPES_DIR / "archetypes.jsonl"
        with out_jsonl.open("w", encoding="utf-8") as f:
            for r in final:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if final:
            scalar_cols = sorted(
                k for r in final
                for k in r.keys()
                if not isinstance(r.get(k), (dict, list))
            )
            with (config.ARCHETYPES_DIR / "archetypes.csv").open(
                "w", encoding="utf-8", newline=""
            ) as f:
                w = csv.writer(f)
                w.writerow(scalar_cols + ["option_A_json", "option_B_json", "ses_natural_cues_json"])
                for r in final:
                    row = [r.get(c, "") for c in scalar_cols]
                    row.append(json.dumps(r.get("option_A", {}), ensure_ascii=False))
                    row.append(json.dumps(r.get("option_B", {}), ensure_ascii=False))
                    row.append(json.dumps(r.get("ses_natural_cues", []), ensure_ascii=False))
                    w.writerow(row)

    elapsed = time.time() - start
    log.info("=" * 60)
    log.info("STEP 3 COMPLETE in %.1fs", elapsed)
    log.info("New extractions this run : %d", new_extractions)
    log.info("Duplicates removed       : %d", duplicates_removed)
    log.info("Final archetype count    : %d", len(final))

    # Breakdown by domain and consensus level
    by_domain_cons: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in final:
        by_domain_cons[r.get("domain", "?")][r.get("consensus_level", "?")] += 1
    for d in config.SUBREDDITS:
        row = by_domain_cons.get(d, {})
        log.info(
            "  %-10s  high_risky=%d  ambiguous=%d  high_safe=%d",
            d,
            row.get("high_risky", 0),
            row.get("ambiguous", 0),
            row.get("high_safe", 0),
        )
    if failures:
        log.info("Failures:")
        for k, v in sorted(failures.items(), key=lambda x: -x[1]):
            log.info("  %-30s %d", k, v)
    log.info("Next step: run step4_generate.py")


if __name__ == "__main__":
    main()
