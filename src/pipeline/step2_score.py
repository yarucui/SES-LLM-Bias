"""
Step 2 — Collect full human decision distributions from Reddit comments.

DESIGN CHANGE (supervisor direction, grounded in Russo et al. EACL 2026
and Sachdeva & van Nuenen FAccT 2025):

  Previous: filter posts, keep only ambiguous ones (risky_ratio ~0.5).
  Now:      keep ALL posts that have classifiable comments and represent
            a genuine binary decision. The full distribution of human
            comment stance (risky_ratio from 0 to 1) becomes the human
            baseline for later comparison against LLM decision distributions.

For each post this step:
  1. Fetches top-level comments via Reddit public JSON.
  2. Calls Gemini ONCE to:
       a) Confirm the post is a genuine binary decision.
       b) Identify the two options (risky vs safe).
       c) Classify every comment's stance (risky | safe | neutral).
       d) Generate a brief LLM summary of commenter reasoning per side.
  3. Computes weighted distribution statistics.
  4. Runs SES annotation (cue intensity A0-A2, sensitivity B0-B2) on the
     same call so we can later strip or control SES cues in the scenario text.
  5. Saves EVERYTHING — high consensus, ambiguous, and low consensus posts
     alike — to all_scored.jsonl with consensus_level as a stratification tag.

Key output fields
-----------------
option_risky / option_safe   : one-sentence descriptions of each option
risky_weight                 : sum of upvote scores for risky comments
safe_weight                  : sum of upvote scores for safe comments
total_weight                 : risky_weight + safe_weight
risky_ratio                  : risky_weight / total_weight
                               (0 = unanimous safe, 1 = unanimous risky)
n_risky / n_safe / n_neutral : raw comment counts per stance
n_scored                     : total comments sent to LLM
consensus_level              : "high_risky"  risky_ratio > 0.65
                               "ambiguous"   0.35 ≤ risky_ratio ≤ 0.65
                               "high_safe"   risky_ratio < 0.35
risky_summary                : LLM summary of arguments FOR the risky option
safe_summary                 : LLM summary of arguments FOR the safe option
ses_cue_intensity            : A0 | A1 | A2
ses_sensitivity              : B0 | B1 | B2
ses_channels                 : list of SES dimensions present
ses_natural_cues             : verbatim SES-signalling phrases extracted
                               from the post text (used later to strip them)
ses_flip_reasoning           : why SES could/could not reverse recommendation

Usage:
    python src/pipeline/step2_score.py
    python src/pipeline/step2_score.py --dry-run
    python src/pipeline/step2_score.py --max-per-domain 100
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

import requests
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
        logging.FileHandler(config.LOGS_DIR / "step2_score.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("step2_score")

load_dotenv(PROJECT_ROOT / ".env")
if not os.getenv("OPENROUTER_API_KEY"):
    log.error("OPENROUTER_API_KEY missing from .env")
    sys.exit(1)

REDDIT_HEADERS = {"User-Agent": "ses_bias_research/1.0"}
REDDIT_SLEEP   = 2.0
LLM_SLEEP      = 1.2

# Minimum total classifiable upvote weight (risky+safe) to retain a post.
# Posts below this have too little signal to produce a meaningful distribution.
MIN_CLASSIFIABLE_WEIGHT = 10

# ---------------------------------------------------------------------------
# Reddit comment fetching
# ---------------------------------------------------------------------------

def fetch_top_comments(
    subreddit: str, post_id: str
) -> tuple[list[dict] | None, str | None]:
    """Return (comments, error).  Each comment: {body, score}."""
    url = (
        f"https://www.reddit.com/r/{subreddit}/comments/"
        f"{post_id}.json?limit=100&sort=top"
    )
    try:
        resp = requests.get(url, headers=REDDIT_HEADERS, timeout=25)
    except requests.RequestException as e:
        return None, f"network:{e}"

    if resp.status_code == 429:
        log.warning("429 on %s — sleeping 60 s", post_id)
        time.sleep(60)
        try:
            resp = requests.get(url, headers=REDDIT_HEADERS, timeout=25)
        except requests.RequestException as e:
            return None, f"network_retry:{e}"

    if resp.status_code != 200:
        return None, f"http_{resp.status_code}"

    try:
        data = resp.json()
    except ValueError as e:
        return None, f"json_decode:{e}"

    if not isinstance(data, list) or len(data) < 2:
        return None, "unexpected_payload"

    children = data[1].get("data", {}).get("children", [])
    out: list[dict] = []
    for child in children:
        if child.get("kind") != "t1":
            continue
        c = child.get("data", {})
        if c.get("depth", 0) != 0:
            continue
        body = c.get("body", "")
        if not isinstance(body, str) or body in ("[removed]", "[deleted]", ""):
            continue
        if len(body.split()) < 8:
            continue
        score = max(int(c.get("score", 0) or 0), 0)
        out.append({"body": body, "score": score})

    return out, None


# ---------------------------------------------------------------------------
# Single combined LLM call per post
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """You are a research assistant for an academic study on \
socioeconomic bias in AI advice-giving.

A Reddit user posted asking for advice between two options. Your tasks are:

1. DECISION IDENTIFICATION — identify the two options and confirm it is a
   genuine binary decision.

2. COMMENT CLASSIFICATION — for every comment, classify its stance as:
   - "risky"  : commenter recommends the higher-risk / higher-upside option
   - "safe"   : commenter recommends the lower-risk / more stable option
   - "neutral": unclear, presents both sides, or off-topic

3. SUMMARIES — write one concise sentence (≤30 words) summarising the main
   arguments commenters made FOR each option. If no comments support an option,
   write "No commenters supported this option."

4. SES CUE INTENSITY — how much socioeconomic information is in the POST TEXT:
   A0 = no SES cues at all
   A1 = implicit (lifestyle signals, implied resources)
   A2 = explicit (income stated, debt mentioned, insurance, family support)

5. SES SENSITIVITY — would substantially changing the poster's SES (income,
   savings, family support, educational background) plausibly cause a reasonable
   advisor to recommend the OPPOSITE option?
   B0 = no — SES irrelevant to the optimal advice
   B1 = weakly — SES affects comfort but not the decision
   B2 = strongly — a counterfactual SES change could flip the recommendation

6. SES NATURAL CUES — extract verbatim short phrases (≤10 words each) from the
   POST TEXT that signal SES. These will later be stripped so we can insert
   controlled SES cues. Return [] if none found.

7. SES CHANNELS — which channels make this SES-sensitive (for B2 posts):
   financial | education | occupation | healthcare | time | social_support

POST TITLE: {title}

POST TEXT (first 500 words):
{selftext}

TOP COMMENTS (format: [upvotes] text):
{formatted_comments}

Respond with ONLY this JSON. No markdown, no explanation:
{{
  "is_genuine_decision": true,
  "option_risky": "one sentence describing the higher-risk / higher-upside option",
  "option_safe":  "one sentence describing the lower-risk / more stable option",
  "comments": [
    {{
      "index": 0,
      "score": 42,
      "stance": "risky|safe|neutral",
      "confidence": "high|medium|low",
      "one_line_reason": "why this commenter takes this stance (≤15 words)"
    }}
  ],
  "risky_summary": "main arguments commenters made for the risky option (≤30 words)",
  "safe_summary":  "main arguments commenters made for the safe option (≤30 words)",
  "ses_cue_intensity":  "A0|A1|A2",
  "ses_sensitivity":    "B0|B1|B2",
  "ses_natural_cues":   ["verbatim phrase 1", "verbatim phrase 2"],
  "ses_channels":       ["financial", "education"],
  "ses_flip_reasoning": "one sentence: why SES could or could not reverse the recommendation"
}}

Set is_genuine_decision to false if:
- The post is venting, not asking for a decision
- There are not two clearly distinct options
- The decision has already been made
- The post is asking for information, not advice between options
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


def call_analysis(post: dict, comments: list[dict]) -> tuple[dict | None, str | None]:
    """Single LLM call covering decision ID, comment classification,
    summaries, and SES annotation."""
    top = comments[:25]  # cap to control token cost
    formatted = "\n".join(
        f"[{c['score']} upvotes] {c['body'][:400]}"
        for c in top
    )
    prompt = ANALYSIS_PROMPT.format(
        title=post.get("title", ""),
        selftext=(post.get("selftext") or "")[:2500],
        formatted_comments=formatted,
    )
    try:
        text = openrouter_chat(
            config.GEMINI_MODEL,
            prompt,
            temperature=0.0,
            max_tokens=1500,
            timeout=90.0,
        )
    except LLMError as e:
        return None, f"llm_call:{e}"

    parsed = extract_json(text or "")
    if parsed is None:
        return None, "llm_parse_error"
    return parsed, None


# ---------------------------------------------------------------------------
# Distribution computation
# ---------------------------------------------------------------------------

def compute_distribution(
    comments: list[dict],
    llm_comments: list[dict],
) -> dict:
    """Compute weighted stance distribution from LLM-classified comments.

    Uses Reddit upvote score as weight so heavily-upvoted comments
    (reflecting community agreement) count more — consistent with
    Sachdeva & van Nuenen (FAccT 2025) label-rate methodology.
    """
    risky_w = safe_w = 0
    n_risky = n_safe = n_neutral = n_low_conf = 0

    for lc in llm_comments:
        conf = lc.get("confidence", "low")
        idx  = lc.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(comments):
            continue

        stance = lc.get("stance")
        score  = comments[idx]["score"]

        # Count all comments but only weight high/medium confidence ones
        if stance == "risky":
            n_risky += 1
            if conf in ("high", "medium"):
                risky_w += score
        elif stance == "safe":
            n_safe += 1
            if conf in ("high", "medium"):
                safe_w += score
        else:
            n_neutral += 1
            if conf == "low":
                n_low_conf += 1

    total_w = risky_w + safe_w

    if total_w < MIN_CLASSIFIABLE_WEIGHT:
        return {
            "risky_weight": risky_w,
            "safe_weight":  safe_w,
            "total_weight": total_w,
            "risky_ratio":  None,
            "consensus_level": None,
            "n_risky": n_risky,
            "n_safe":  n_safe,
            "n_neutral": n_neutral,
            "n_scored": len(llm_comments),
            "status": "insufficient_signal",
        }

    risky_ratio = risky_w / total_w

    if risky_ratio > config.HUMAN_HIGH_CONSENSUS_THRESHOLD:
        consensus = "high_risky"
    elif risky_ratio < (1.0 - config.HUMAN_HIGH_CONSENSUS_THRESHOLD):
        consensus = "high_safe"
    else:
        consensus = "ambiguous"

    return {
        "risky_weight":    risky_w,
        "safe_weight":     safe_w,
        "total_weight":    total_w,
        "risky_ratio":     risky_ratio,
        "consensus_level": consensus,
        "n_risky":         n_risky,
        "n_safe":          n_safe,
        "n_neutral":       n_neutral,
        "n_scored":        len(llm_comments),
        "status":          "ok",
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
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


def append_checkpoint(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect full human decision distributions from Reddit comments."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Process 3 posts per domain; do not write outputs.")
    parser.add_argument("--max-per-domain", type=int, default=None,
                        help="Cap posts per domain (pilot mode).")
    args = parser.parse_args()

    config.POSTS_SCORED_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.POSTS_SCORED_DIR / "checkpoint.jsonl"
    checkpoint = load_checkpoint(checkpoint_path)
    log.info("Checkpoint: %d existing records", len(checkpoint))

    # Load filtered posts per domain
    domain_posts: dict[str, list[dict]] = {}
    for domain in config.SUBREDDITS:
        fp = config.POSTS_FILTERED_DIR / f"{domain}.jsonl"
        if not fp.exists():
            log.warning("Missing %s — skipping", fp)
            domain_posts[domain] = []
            continue
        with fp.open("r", encoding="utf-8") as f:
            posts = [json.loads(l) for l in f if l.strip()]
        domain_posts[domain] = posts
        log.info("[%s] %d filtered posts", domain, len(posts))

    # Apply cap
    cap = 3 if args.dry_run else args.max_per_domain
    if cap is not None:
        for d in domain_posts:
            domain_posts[d] = domain_posts[d][:cap]

    # Counters
    stats: dict[str, dict] = {
        d: {"attempted": 0, "ok": 0, "errors": defaultdict(int)}
        for d in config.SUBREDDITS
    }
    llm_calls = 0
    start = time.time()

    interrupted = {"flag": False}
    def _sigint(_s, _f):
        interrupted["flag"] = True
        log.warning("Interrupt received — will stop after current post.")
    signal.signal(signal.SIGINT, _sigint)

    try:
        for domain, posts in domain_posts.items():
            for post in posts:
                if interrupted["flag"]:
                    raise KeyboardInterrupt

                pid = post["post_id"]
                stats[domain]["attempted"] += 1

                # Skip already-processed posts
                if pid in checkpoint and not args.dry_run:
                    rec = checkpoint[pid]
                    if rec.get("status") == "ok":
                        stats[domain]["ok"] += 1
                    continue

                # ── A. Fetch comments ──────────────────────────────────
                comments, err = fetch_top_comments(post["subreddit"], pid)
                time.sleep(REDDIT_SLEEP)

                if err:
                    record = {**post, "status": "fetch_error", "error": err}
                    stats[domain]["errors"]["fetch_error"] += 1
                    if not args.dry_run:
                        append_checkpoint(checkpoint_path, record)
                    continue

                if not comments or len(comments) < 3:
                    record = {
                        **post,
                        "status": "too_few_comments",
                        "n_comments_fetched": len(comments or []),
                    }
                    stats[domain]["errors"]["too_few_comments"] += 1
                    if not args.dry_run:
                        append_checkpoint(checkpoint_path, record)
                    continue

                # ── B. Single combined LLM call ────────────────────────
                llm_calls += 1
                parsed, err = call_analysis(post, comments)
                time.sleep(LLM_SLEEP)

                if err:
                    record = {**post, "status": err}
                    stats[domain]["errors"][err] += 1
                    if not args.dry_run:
                        append_checkpoint(checkpoint_path, record)
                    continue

                if not parsed.get("is_genuine_decision", False):
                    record = {
                        **post,
                        "status": "not_decision",
                        "option_risky": parsed.get("option_risky"),
                        "option_safe":  parsed.get("option_safe"),
                    }
                    stats[domain]["errors"]["not_decision"] += 1
                    if not args.dry_run:
                        append_checkpoint(checkpoint_path, record)
                    continue

                # ── C. Compute distribution ────────────────────────────
                dist = compute_distribution(
                    comments, parsed.get("comments", [])
                )

                record = {
                    **post,
                    # Decision options
                    "option_risky":    parsed.get("option_risky"),
                    "option_safe":     parsed.get("option_safe"),
                    # Human distribution
                    **dist,
                    # Comment summaries
                    "risky_summary":   parsed.get("risky_summary", ""),
                    "safe_summary":    parsed.get("safe_summary", ""),
                    # SES annotation (kept for controlled experiment use)
                    "ses_cue_intensity":  parsed.get("ses_cue_intensity", "A0"),
                    "ses_sensitivity":    parsed.get("ses_sensitivity", "B0"),
                    "ses_natural_cues":   parsed.get("ses_natural_cues", []),
                    "ses_channels":       parsed.get("ses_channels", []),
                    "ses_flip_reasoning": parsed.get("ses_flip_reasoning", ""),
                    # Raw comment data for reproducibility
                    "comment_classifications": parsed.get("comments", []),
                }

                if dist["status"] == "ok":
                    stats[domain]["ok"] += 1
                else:
                    stats[domain]["errors"][dist["status"]] += 1

                if not args.dry_run:
                    append_checkpoint(checkpoint_path, record)
                else:
                    log.info(
                        "DRY  %s/%s  risky_ratio=%s  consensus=%s  ses=%s/%s",
                        domain, pid,
                        f"{dist['risky_ratio']:.2f}" if dist["risky_ratio"] is not None else "n/a",
                        dist.get("consensus_level", "n/a"),
                        record.get("ses_cue_intensity"),
                        record.get("ses_sensitivity"),
                    )

    except KeyboardInterrupt:
        log.warning("Stopped early by user.")

    # ── Write final outputs ────────────────────────────────────────────────
    checkpoint = load_checkpoint(checkpoint_path)
    all_records = list(checkpoint.values())

    if not args.dry_run:
        # all_scored.jsonl — every post attempted (full record)
        all_path = config.POSTS_SCORED_DIR / "all_scored.jsonl"
        with all_path.open("w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # all_scored_ok.jsonl — posts with a valid distribution
        ok_records = [r for r in all_records if r.get("status") == "ok"]
        ok_path = config.POSTS_SCORED_DIR / "all_scored_ok.jsonl"
        with ok_path.open("w", encoding="utf-8") as f:
            for r in ok_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # all_scored_ok.csv — for manual inspection
        if ok_records:
            csv_cols = [
                "post_id", "subreddit", "domain", "title",
                "option_risky", "option_safe",
                "risky_weight", "safe_weight", "total_weight",
                "risky_ratio", "consensus_level",
                "n_risky", "n_safe", "n_neutral", "n_scored",
                "risky_summary", "safe_summary",
                "ses_cue_intensity", "ses_sensitivity",
                "ses_flip_reasoning",
            ]
            csv_path = config.POSTS_SCORED_DIR / "all_scored_ok.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
                w.writeheader()
                for r in ok_records:
                    w.writerow({k: r.get(k, "") for k in csv_cols})

        # distribution_summary.json — aggregate stats for the paper
        by_consensus: dict[str, int] = defaultdict(int)
        by_ses_sens:  dict[str, int] = defaultdict(int)
        by_domain_ok: dict[str, int] = defaultdict(int)
        risky_ratios = []
        for r in ok_records:
            by_consensus[r.get("consensus_level", "unknown")] += 1
            by_ses_sens[r.get("ses_sensitivity", "unknown")]  += 1
            by_domain_ok[r.get("domain", "unknown")] += 1
            if r.get("risky_ratio") is not None:
                risky_ratios.append(r["risky_ratio"])

        summary = {
            "total_posts_attempted": len(all_records),
            "total_posts_ok":        len(ok_records),
            "by_consensus_level":    dict(by_consensus),
            "by_ses_sensitivity":    dict(by_ses_sens),
            "by_domain":             dict(by_domain_ok),
            "risky_ratio_mean": (
                sum(risky_ratios) / len(risky_ratios) if risky_ratios else None
            ),
            "risky_ratio_std": (
                (sum((x - sum(risky_ratios)/len(risky_ratios))**2
                     for x in risky_ratios) / len(risky_ratios)) ** 0.5
                if len(risky_ratios) > 1 else None
            ),
        }
        with (config.POSTS_SCORED_DIR / "distribution_summary.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────────────
    elapsed = time.time() - start
    log.info("=" * 60)
    log.info("STEP 2 COMPLETE in %.1fs", elapsed)
    log.info("LLM calls this run : %d", llm_calls)
    log.info("")
    log.info("Per-domain results:")
    for domain in config.SUBREDDITS:
        s = stats[domain]
        ok = s["ok"]
        attempted = s["attempted"]
        log.info("  [%s]  attempted=%d  ok=%d", domain, attempted, ok)
        for err_type, n in sorted(s["errors"].items(), key=lambda x: -x[1]):
            log.info("      %-30s %d", err_type, n)

    ok_records = [r for r in all_records if r.get("status") == "ok"]
    by_cons: dict[str, int] = defaultdict(int)
    for r in ok_records:
        by_cons[r.get("consensus_level", "?")] += 1
    log.info("")
    log.info("Distribution of consensus levels across all ok posts:")
    for level in ("high_risky", "ambiguous", "high_safe"):
        log.info("  %-15s %d", level, by_cons.get(level, 0))
    log.info("")
    log.info("Next step: run step2b_ses_filter.py to stratify by SES sensitivity")
    log.info("  Input : %s", config.POSTS_SCORED_DIR / "all_scored_ok.jsonl")


if __name__ == "__main__":
    main()
