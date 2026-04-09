"""
Step 2 — Fetch top-level comments from Reddit's public JSON endpoint, then
ask Gemini to (a) confirm the post is a genuine two-option decision and
(b) classify each comment's stance. Compute a comment-weighted ambiguity
score for each post and write out the candidates that look ambiguous enough.

Crash-safe: progress is appended to a checkpoint file after every post,
both successes and failures, and reruns skip post_ids that already exist.

Usage:
    python src/pipeline/step2_score.py
    python src/pipeline/step2_score.py --dry-run
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

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Setup
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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    log.error("OPENROUTER_API_KEY missing from .env")
    sys.exit(1)

from _llm import LLMError, openrouter_chat  # noqa: E402

REDDIT_HEADERS = {"User-Agent": "ses_bias_research/1.0"}
REDDIT_SLEEP   = 2.0
GEMINI_SLEEP   = 1.0

# ---------------------------------------------------------------------------
# Reddit fetch
# ---------------------------------------------------------------------------
def fetch_top_comments(subreddit: str, post_id: str) -> tuple[list[dict] | None, str | None]:
    """Return (comments_list, error). Each comment is a dict with body & score."""
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json?limit=50&sort=top"
    try:
        resp = requests.get(url, headers=REDDIT_HEADERS, timeout=20)
    except requests.RequestException as e:
        return None, f"network: {e}"

    if resp.status_code == 429:
        log.warning("429 rate-limit, sleeping 60s and retrying once")
        time.sleep(60)
        try:
            resp = requests.get(url, headers=REDDIT_HEADERS, timeout=20)
        except requests.RequestException as e:
            return None, f"network: {e}"

    if resp.status_code != 200:
        return None, f"http_{resp.status_code}"

    try:
        data = resp.json()
    except ValueError as e:
        return None, f"json_decode: {e}"

    if not isinstance(data, list) or len(data) < 2:
        return None, "unexpected_payload"

    comments_listing = data[1].get("data", {}).get("children", [])
    out: list[dict] = []
    for child in comments_listing:
        if child.get("kind") != "t1":
            continue
        c = child.get("data", {})
        if c.get("depth", 0) != 0:
            continue
        body = c.get("body", "")
        if not isinstance(body, str) or body in ("[removed]", "[deleted]"):
            continue
        if len(body.split()) < 5:
            continue
        score = int(c.get("score", 0) or 0)
        if score <= 0:
            continue
        out.append({"body": body, "score": score})
    return out, None


# ---------------------------------------------------------------------------
# Gemini scoring
# ---------------------------------------------------------------------------
GEMINI_PROMPT = """You are a research assistant for an academic study on \
decision-making. A Reddit user posted asking for advice between two options. \
Analyze the post and its comments.

POST TITLE: {title}

POST TEXT (first 400 words):
{selftext}

TOP COMMENTS:
{formatted_comments}

Instructions:
1. Identify what the two main options are (risky/ambitious vs safe/conservative).
2. For each comment, classify its stance.
3. Assess if this is a genuine two-option decision post.

Respond with ONLY this JSON (no markdown, no explanation):
{{
  "option_risky": "one sentence describing the riskier option",
  "option_safe": "one sentence describing the safer option",
  "is_genuine_decision": true,
  "domain_confirmed": "education|career|finance|health|social",
  "comments": [
    {{
      "index": 0,
      "score": {upvote_count},
      "stance": "risky|safe|neutral",
      "confidence": "high|medium|low"
    }}
  ]
}}

Set is_genuine_decision to false if:
- The post is venting, not asking for a decision
- There are not clearly two distinct options
- The decision has already been made
"""


def extract_json(text: str) -> dict | None:
    """Robustly find the first JSON object in a Gemini response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()
    # Greedy match for the outermost {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def call_gemini(post: dict, comments: list[dict]) -> tuple[dict | None, str | None]:
    top = comments[:15]
    formatted = "\n".join(
        f"[{i}] [{c['score']} upvotes] {c['body']}"
        for i, c in enumerate(top)
    )
    upvote_count = top[0]["score"] if top else 0
    prompt = GEMINI_PROMPT.format(
        title=post.get("title", ""),
        selftext=post.get("selftext", "")[:2000],
        formatted_comments=formatted,
        upvote_count=upvote_count,
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
# Ambiguity calculation
# ---------------------------------------------------------------------------
def compute_ambiguity(comments: list[dict], gemini_comments: list[dict]) -> dict:
    risky_w = 0
    safe_w  = 0
    used    = 0
    for gc in gemini_comments:
        if gc.get("confidence") not in ("high", "medium"):
            continue
        idx = gc.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(comments):
            continue
        score = comments[idx]["score"]
        stance = gc.get("stance")
        if stance == "risky":
            risky_w += score
            used += 1
        elif stance == "safe":
            safe_w += score
            used += 1
    total = risky_w + safe_w
    if total == 0:
        return {
            "risky_weight": 0, "safe_weight": 0,
            "risky_ratio": None, "ambiguity_score": None,
            "num_comments_scored": used,
            "status": "no_classifiable_comments",
        }
    risky_ratio = risky_w / total
    ambiguity = 1 - abs(risky_ratio - 0.5) * 2
    return {
        "risky_weight": risky_w,
        "safe_weight":  safe_w,
        "risky_ratio":  risky_ratio,
        "ambiguity_score": ambiguity,
        "num_comments_scored": used,
        "status": "ok",
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
    parser.add_argument("--max-per-domain", type=int, default=None,
                        help="Cap posts processed per domain (pilot mode).")
    args = parser.parse_args()

    config.POSTS_SCORED_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.POSTS_SCORED_DIR / "checkpoint.jsonl"
    checkpoint = load_checkpoint(checkpoint_path)
    log.info("Loaded %d existing checkpoint records", len(checkpoint))

    # Build the list of posts to process per domain
    domain_to_posts: dict[str, list[dict]] = {}
    for domain in config.SUBREDDITS:
        path = config.POSTS_FILTERED_DIR / f"{domain}.jsonl"
        if not path.exists():
            log.warning("Missing %s, skipping domain", path)
            domain_to_posts[domain] = []
            continue
        with path.open("r", encoding="utf-8") as f:
            posts = [json.loads(line) for line in f if line.strip()]
        domain_to_posts[domain] = posts
        log.info("[%s] %d filtered posts", domain, len(posts))

    if args.dry_run:
        for d in domain_to_posts:
            domain_to_posts[d] = domain_to_posts[d][:3]
    elif args.max_per_domain is not None:
        for d in domain_to_posts:
            domain_to_posts[d] = domain_to_posts[d][:args.max_per_domain]
        log.info("Pilot mode: capped to %d posts per domain", args.max_per_domain)

    # Counters
    attempted     = defaultdict(int)
    scored_ok     = defaultdict(int)
    error_counts  = defaultdict(int)
    gemini_calls  = 0
    start = time.time()

    interrupted = {"flag": False}

    def _sigint(_s, _f):
        interrupted["flag"] = True
        log.warning("Interrupt received; will exit after current post.")

    signal.signal(signal.SIGINT, _sigint)

    try:
        for domain, posts in domain_to_posts.items():
            for post in posts:
                if interrupted["flag"]:
                    raise KeyboardInterrupt
                pid = post["post_id"]
                attempted[domain] += 1

                if pid in checkpoint and not args.dry_run:
                    rec = checkpoint[pid]
                    if rec.get("status") == "ok":
                        scored_ok[domain] += 1
                    else:
                        error_counts[rec.get("status", "unknown")] += 1
                    continue

                # ---- A. fetch comments
                comments, err = fetch_top_comments(post["subreddit"], pid)
                time.sleep(REDDIT_SLEEP)
                if err is not None:
                    record = {**post, "status": "fetch_error", "error": err}
                    error_counts["fetch_error"] += 1
                    if not args.dry_run:
                        append_checkpoint(checkpoint_path, record)
                    continue
                if comments is None or len(comments) < 5:
                    record = {**post, "status": "insufficient_comments",
                              "num_qualifying_comments": len(comments or [])}
                    error_counts["insufficient_comments"] += 1
                    if not args.dry_run:
                        append_checkpoint(checkpoint_path, record)
                    continue

                # ---- B. Gemini classification
                gemini_calls += 1
                parsed, err = call_gemini(post, comments)
                time.sleep(GEMINI_SLEEP)
                if err is not None:
                    record = {**post, "status": err}
                    error_counts[err] += 1
                    if not args.dry_run:
                        append_checkpoint(checkpoint_path, record)
                    continue

                if not parsed.get("is_genuine_decision", False):
                    record = {**post, "status": "not_decision",
                              "option_risky": parsed.get("option_risky"),
                              "option_safe":  parsed.get("option_safe")}
                    error_counts["not_decision"] += 1
                    if not args.dry_run:
                        append_checkpoint(checkpoint_path, record)
                    continue

                # ---- C. ambiguity
                amb = compute_ambiguity(comments, parsed.get("comments", []))
                record = {
                    **post,
                    "option_risky":     parsed.get("option_risky"),
                    "option_safe":      parsed.get("option_safe"),
                    "domain_confirmed": parsed.get("domain_confirmed"),
                    **amb,
                }
                if amb["status"] == "ok":
                    scored_ok[domain] += 1
                else:
                    error_counts[amb["status"]] += 1
                if not args.dry_run:
                    append_checkpoint(checkpoint_path, record)
                else:
                    log.info("DRY  %s/%s ambiguity=%s", domain, pid,
                             amb.get("ambiguity_score"))
    except KeyboardInterrupt:
        log.warning("Stopped early by user.")

    # Re-load the (now-updated) checkpoint and emit final outputs
    checkpoint = load_checkpoint(checkpoint_path)
    all_records = list(checkpoint.values())

    if not args.dry_run:
        all_path = config.POSTS_SCORED_DIR / "all_scored.jsonl"
        with all_path.open("w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        ambiguous = [
            r for r in all_records
            if r.get("status") == "ok"
            and r.get("ambiguity_score") is not None
            and r["ambiguity_score"] >= 0.55
        ]
        ambiguous.sort(key=lambda r: (r["domain"], -r["ambiguity_score"]))

        amb_jsonl = config.POSTS_SCORED_DIR / "ambiguous.jsonl"
        with amb_jsonl.open("w", encoding="utf-8") as f:
            for rec in ambiguous:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        amb_csv = config.POSTS_SCORED_DIR / "ambiguous.csv"
        if ambiguous:
            cols = sorted({k for r in ambiguous for k in r.keys()})
            with amb_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in ambiguous:
                    w.writerow({k: r.get(k, "") for k in cols})

    # Summary
    elapsed = time.time() - start
    log.info("=" * 60)
    log.info("STEP 2 COMPLETE in %.1fs", elapsed)
    log.info("Gemini calls this run: %d", gemini_calls)
    log.info("Errors by type:")
    for k, v in sorted(error_counts.items(), key=lambda x: -x[1]):
        log.info("  %-30s %d", k, v)

    by_domain: dict[str, list[dict]] = defaultdict(list)
    for r in all_records:
        if r.get("status") == "ok" and r.get("ambiguity_score") is not None:
            by_domain[r["domain"]].append(r)

    for domain in config.SUBREDDITS:
        records = by_domain[domain]
        ambiguous = [r for r in records if r["ambiguity_score"] >= 0.55]
        mean_amb = sum(r["ambiguity_score"] for r in records) / len(records) if records else 0
        log.info("[%s] attempted=%d ok=%d ambiguous=%d mean_amb=%.3f",
                 domain, attempted[domain], len(records), len(ambiguous), mean_amb)
        top3 = sorted(records, key=lambda r: -r["ambiguity_score"])[:3]
        for r in top3:
            log.info("    %.3f  %s", r["ambiguity_score"], r["title"][:80])


if __name__ == "__main__":
    main()
