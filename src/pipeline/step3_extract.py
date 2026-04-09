"""
Step 3 — Extract abstract decision *archetypes* from the ambiguous posts.

Uses Gemini once per source post. The response is intentionally stripped of
all personal/identifying detail and only the abstract structural pattern is
carried forward into the dataset. Near-duplicates (same domain + trade-off
type + resource constraint) are collapsed, keeping the higher-ambiguity one.

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

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Setup
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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    log.error("OPENROUTER_API_KEY missing from .env")
    sys.exit(1)

from _llm import LLMError, openrouter_chat  # noqa: E402

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
EXTRACT_PROMPT = """You are a research assistant extracting decision structures \
for an academic study. Read this Reddit post and extract the abstract decision \
pattern.

CRITICAL RULES:
- Do NOT include any names, usernames, or real identities
- Do NOT include specific school names, company names, or city names
- Do NOT include any personally identifying details
- Abstract everything to generic descriptions
- Focus only on the structural pattern of the decision

POST TITLE: {title}
POST TEXT: {selftext}
IDENTIFIED OPTIONS:
  Risky option: {option_risky}
  Safe option: {option_safe}

Extract the decision structure as JSON:
{{
  "domain": "education|career|finance|health|social",
  "trade_off_type": "one of: prestige_vs_cost | risk_vs_stability | short_term_vs_long_term | mobility_vs_rootedness | aggressive_vs_conservative",
  "option_A": {{
    "label": "risky",
    "description": "abstract description with NO identifying details",
    "upside": "what the person gains if it works out",
    "downside": "what the person risks or loses",
    "key_feature": "one word: debt|relocation|uncertainty|cost|commitment"
  }},
  "option_B": {{
    "label": "safe",
    "description": "abstract description with NO identifying details",
    "upside": "what the person gains",
    "downside": "what the person gives up",
    "key_feature": "one word: stability|savings|proximity|certainty|flexibility"
  }},
  "key_moderator": "the single factor that makes this genuinely hard to decide (e.g. career field prestige sensitivity, time to break-even, family proximity)",
  "resource_constraint": "financial|time|social|health|geographic",
  "time_horizon": "short|long",
  "reversibility": "reversible|irreversible|partially_reversible",
  "fallback_available": true,
  "archetype_description": "one sentence describing the abstract pattern, e.g.: A person chooses between a high-prestige opportunity requiring significant debt versus a lower-prestige option with financial security"
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


def call_gemini(post: dict) -> tuple[dict | None, str | None]:
    prompt = EXTRACT_PROMPT.format(
        title=post.get("title", ""),
        selftext=post.get("selftext", "")[:2000],
        option_risky=post.get("option_risky", ""),
        option_safe=post.get("option_safe", ""),
    )
    try:
        text = openrouter_chat(
            config.GEMINI_MODEL,
            prompt,
            temperature=0.0,
            max_tokens=900,
            timeout=60.0,
        )
    except LLMError as e:
        return None, f"gemini_call: {e}"
    parsed = extract_json(text or "")
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
                seen[rec["source_post_id"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def append_checkpoint(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only 3 posts per domain and don't write outputs.")
    args = parser.parse_args()

    config.ARCHETYPES_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.ARCHETYPES_DIR / "extraction_checkpoint.jsonl"
    checkpoint = load_checkpoint(checkpoint_path)
    log.info("Loaded %d existing extractions from checkpoint", len(checkpoint))

    # Read the SES-filtered ambiguous posts and bucket by domain
    src = config.POSTS_SCORED_DIR / "ambiguous_ses.jsonl"
    if not src.exists():
        log.error("ERROR: ambiguous_ses.jsonl not found.")
        log.error("Run step2b_ses_filter.py first.")
        sys.exit(1)

    by_domain: dict[str, list[dict]] = defaultdict(list)
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_domain[r["domain"]].append(r)

    # Sort each domain by ambiguity desc and take top N (oversample 3x)
    per_domain_target = config.TARGETS_PER_DOMAIN * 3
    if args.dry_run:
        per_domain_target = 3

    selected: dict[str, list[dict]] = {}
    for domain, items in by_domain.items():
        items.sort(key=lambda r: -(r.get("ambiguity_score") or 0))
        selected[domain] = items[:per_domain_target]
        log.info("[%s] selected %d posts (of %d ambiguous)", domain,
                 len(selected[domain]), len(items))

    # ---- Extract
    interrupted = {"flag": False}

    def _sigint(_s, _f):
        interrupted["flag"] = True
        log.warning("Interrupt received; will exit after current post.")

    signal.signal(signal.SIGINT, _sigint)

    start = time.time()
    new_extractions = 0
    failures = defaultdict(int)

    try:
        for domain, posts in selected.items():
            for post in posts:
                if interrupted["flag"]:
                    raise KeyboardInterrupt
                pid = post["post_id"]

                if pid in checkpoint and not args.dry_run:
                    continue

                parsed, err = call_gemini(post)
                time.sleep(1.0)
                if err is not None:
                    log.warning("[%s] %s -> %s", domain, pid, err)
                    failures[err] += 1
                    rec = {
                        "source_post_id": pid,
                        "source_subreddit": post.get("subreddit"),
                        "domain_seed": domain,
                        "ambiguity_score": post.get("ambiguity_score"),
                        "status": err,
                    }
                    if not args.dry_run:
                        append_checkpoint(checkpoint_path, rec)
                    continue

                rec = {
                    "source_post_id":   pid,
                    "source_subreddit": post.get("subreddit"),
                    "domain_seed":      domain,
                    "ambiguity_score":  post.get("ambiguity_score"),
                    "status":           "ok",
                    **parsed,
                }
                # Force domain to be the seed if Gemini drifted
                rec["domain"] = parsed.get("domain") or domain
                new_extractions += 1
                if not args.dry_run:
                    append_checkpoint(checkpoint_path, rec)
                else:
                    log.info("DRY  %s/%s -> %s",
                             domain, pid, parsed.get("trade_off_type"))
    except KeyboardInterrupt:
        log.warning("Stopped early by user.")

    # Re-load checkpoint and dedupe
    checkpoint = load_checkpoint(checkpoint_path)
    valid = [r for r in checkpoint.values() if r.get("status") == "ok"]
    log.info("Total successful extractions on disk: %d", len(valid))

    # Deduplicate by (domain, trade_off_type, resource_constraint), keep max ambiguity
    dedup_key = lambda r: (r.get("domain"),
                           r.get("trade_off_type"),
                           r.get("resource_constraint"))
    best: dict[tuple, dict] = {}
    for r in valid:
        k = dedup_key(r)
        if k not in best or (r.get("ambiguity_score") or 0) > (best[k].get("ambiguity_score") or 0):
            best[k] = r

    duplicates_removed = len(valid) - len(best)

    # Sort by domain, then ambiguity desc; mint archetype IDs
    final = sorted(best.values(),
                   key=lambda r: (r.get("domain") or "",
                                  -(r.get("ambiguity_score") or 0)))

    counters: dict[str, int] = defaultdict(int)
    for r in final:
        domain = r.get("domain")
        prefix = config.DOMAIN_PREFIX.get(domain, "GEN")
        counters[prefix] += 1
        r["archetype_id"] = f"{prefix}_{counters[prefix]:02d}"

    if not args.dry_run:
        out_jsonl = config.ARCHETYPES_DIR / "archetypes.jsonl"
        with out_jsonl.open("w", encoding="utf-8") as f:
            for r in final:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if final:
            cols = sorted({k for r in final for k in r.keys() if not isinstance(r[k], (dict, list))})
            # Always include archetype_id and a couple of structured fields as JSON strings
            with (config.ARCHETYPES_DIR / "archetypes.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols + ["option_A_json", "option_B_json"])
                for r in final:
                    row = [r.get(c, "") for c in cols]
                    row.append(json.dumps(r.get("option_A", {}), ensure_ascii=False))
                    row.append(json.dumps(r.get("option_B", {}), ensure_ascii=False))
                    w.writerow(row)

    elapsed = time.time() - start
    log.info("=" * 60)
    log.info("STEP 3 COMPLETE in %.1fs", elapsed)
    log.info("New extractions this run : %d", new_extractions)
    log.info("Duplicates removed       : %d", duplicates_removed)
    log.info("Final archetype count    : %d", len(final))
    by_domain_count = defaultdict(int)
    for r in final:
        by_domain_count[r.get("domain")] += 1
    for d in config.SUBREDDITS:
        log.info("  %-10s %d", d, by_domain_count.get(d, 0))
    if failures:
        log.info("Failures by type:")
        for k, v in failures.items():
            log.info("  %-30s %d", k, v)


if __name__ == "__main__":
    main()
