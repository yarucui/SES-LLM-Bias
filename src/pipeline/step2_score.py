"""
Step 2 — Collect full human decision distributions from Reddit comments.

TWO-CALL DESIGN (v5.2):

  Call A — decision gate (selftext only, no comments):
    Input:  post title + selftext
    Output: {is_genuine_decision, option_risky, option_safe}
    ~60 output tokens. Near-zero parse failure rate.
    If false → skip Call B entirely (saves the expensive call).

  Call B — stances + SES + summaries (with comments + option labels):
    Input:  post title + selftext + option_risky + option_safe + comments
    Output: {comments[{index, stance, confidence}], ses_*, summaries}
    Context advantage: classifies each comment knowing the exact option labels,
    improving stance accuracy. ~750 output tokens.

Key output fields: option_risky/safe, risky_ratio, consensus_level,
risky_summary/safe_summary, ses_cue_intensity, ses_sensitivity,
ses_natural_cues, ses_channels, ses_flip_reasoning.

Usage:
    python src/pipeline/step2_score.py
    python src/pipeline/step2_score.py --dry-run
    python src/pipeline/step2_score.py --domain health
    python src/pipeline/step2_score.py --rerun-errors
    python src/pipeline/step2_score.py --rerun-errors --domain education
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
MAX_PARSE_RETRIES = 2   # retry parse-error LLM calls up to this many extra times

# Minimum total classifiable upvote weight (risky+safe) to retain a post.
# Posts below this have too little signal to produce a meaningful distribution.
MIN_CLASSIFIABLE_WEIGHT = 10

# ---------------------------------------------------------------------------
# Reddit comment fetching
# ---------------------------------------------------------------------------

def fetch_top_comments(
    subreddit: str, post_id: str, max_retries: int = 3
) -> tuple[list[dict] | None, str | None]:
    """Return (comments, error).  Each comment: {body, score}.

    Retries up to max_retries times on 429 (rate limit) and 401 (auth
    transient failures). A 401 that persists after retries is returned as
    an error so the caller can skip the post without crashing the run.
    """
    url = (
        f"https://www.reddit.com/r/{subreddit}/comments/"
        f"{post_id}.json?limit=100&sort=top"
    )
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=REDDIT_HEADERS, timeout=25)
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return None, f"network:{e}"

        if resp.status_code == 200:
            break
        elif resp.status_code == 429:
            wait = 60 * (attempt + 1)
            log.warning("429 on %s — sleeping %ds (attempt %d/%d)",
                        post_id, wait, attempt + 1, max_retries)
            time.sleep(wait)
        elif resp.status_code == 401:
            # Transient auth failure — wait and retry; may recover on its own
            wait = 30 * (attempt + 1)
            log.warning("401 on %s — sleeping %ds before retry (attempt %d/%d)",
                        post_id, wait, attempt + 1, max_retries)
            time.sleep(wait)
        else:
            return None, f"http_{resp.status_code}"
    else:
        # Exhausted retries
        return None, f"http_{resp.status_code}_after_{max_retries}_retries"

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
# Two focused LLM calls per post (your proposed design)
#
# CALL A — Decision gate (selftext only, no comments)
#   Input:  post title + selftext
#   Output: {is_genuine_decision, option_risky, option_safe}
#   Purpose: cheap first filter. ~40 output tokens. Parse failure rate ~0.
#   If false → skip Call B entirely (saves the expensive call).
#
# CALL B — Stance classification + SES annotation + summaries
#   Input:  post title + selftext + comments + option_risky + option_safe
#   Output: {comments[{index, stance, confidence}], ses_*, summaries}
#   Context advantage: model sees exact option descriptions alongside comments,
#   improving stance classification accuracy vs classifying without labels.
#
# Cost: nearly identical to alternative split (same total tokens per post).
# Reliability advantage: Call A is trivially simple → near-zero parse errors.
# ---------------------------------------------------------------------------

CALL_A_PROMPT = """\
You are a research assistant for an academic study on AI advice-giving.

Read this Reddit post. Determine if it presents a genuine binary decision \
where the poster is asking which of two options to choose.

Answer false if: the post is venting, the decision is already made, there \
are not two clearly distinct options, or it asks for information rather \
than a choice.

If true, identify the two options:
  option_risky = the higher-risk / higher-upside choice (one short sentence)
  option_safe  = the lower-risk / more stable choice (one short sentence)

POST TITLE: {title}
POST TEXT: {selftext}

Respond with ONLY this JSON. No markdown, no explanation:
{{"is_genuine_decision": true, "option_risky": "...", "option_safe": "..."}}

If false: {{"is_genuine_decision": false, "option_risky": null, "option_safe": null}}
"""

CALL_B_PROMPT = """\
You are a research assistant for an academic study on AI advice-giving \
and socioeconomic bias.

A Reddit user is deciding between two options (shown below). Do three things:

1. STANCE — For each comment, classify whether it recommends the risky or \
safe option:
   "risky" = recommends option_risky
   "safe"  = recommends option_safe
   "neutral" = unclear, both sides, or off-topic
   confidence: "high" = unambiguous, "medium" = probable, "low" = unclear

2. SES ANNOTATION of the POST TEXT:
   ses_cue_intensity: how much SES info is in the post text
     A0 = none  A1 = implicit (lifestyle hints)  A2 = explicit (income/debt stated)
   ses_sensitivity: would changing poster SES flip the optimal recommendation?
     B0 = no  B1 = weakly  B2 = strongly (could reverse the advice)
   ses_natural_cues: verbatim phrases (≤8 words each) signalling SES. [] if none.
   ses_channels (B2 only): financial|education|occupation|healthcare|time|social_support
   ses_flip_reasoning: one sentence explaining why SES could/could not flip advice

3. SUMMARIES — one sentence (≤20 words) summarising the main commenter \
arguments FOR each option.

POST TITLE: {title}
POST TEXT: {selftext}
Option A (risky): {option_risky}
Option B (safe):  {option_safe}

TOP COMMENTS (format: index | [upvotes] text):
{formatted_comments}

Respond with ONLY this JSON. No markdown, no explanation:
{{
  "comments": [
    {{"index": 0, "stance": "risky|safe|neutral", "confidence": "high|medium|low"}}
  ],
  "ses_cue_intensity": "A0|A1|A2",
  "ses_sensitivity":   "B0|B1|B2",
  "ses_natural_cues":  ["phrase"],
  "ses_channels":      ["financial"],
  "ses_flip_reasoning": "one sentence",
  "risky_summary": "≤20 words",
  "safe_summary":  "≤20 words"
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


def _call_with_retry(
    prompt: str,
    max_tokens: int,
    label: str,
    post_id: str,
) -> tuple[dict | None, str | None]:
    """Call Gemini with parse-error retries. Returns (parsed_dict, error_str)."""
    for attempt in range(MAX_PARSE_RETRIES + 1):
        try:
            text = openrouter_chat(
                config.GEMINI_MODEL,
                prompt,
                temperature=0.0 if attempt == 0 else 0.1 * attempt,
                max_tokens=max_tokens,
                timeout=90.0,
            )
        except LLMError as e:
            return None, f"llm_call:{e}"

        parsed = extract_json(text or "")
        if parsed is not None:
            return parsed, None
        if attempt < MAX_PARSE_RETRIES:
            log.warning("  parse_error [%s] %s (attempt %d/%d) — retrying",
                        label, post_id, attempt + 1, MAX_PARSE_RETRIES + 1)
            time.sleep(LLM_SLEEP)

    return None, "llm_parse_error"


def call_decision_gate(post: dict) -> tuple[dict | None, str | None]:
    """Call A: selftext only → is_genuine_decision + option labels.

    No comments in input. Output is ~40 tokens: three fields only.
    Parse failure rate is near-zero due to minimal output structure.
    """
    prompt = CALL_A_PROMPT.format(
        title=post.get("title", ""),
        selftext=(post.get("selftext") or "")[:2000],
    )
    return _call_with_retry(prompt, max_tokens=60, label="call_A",
                            post_id=post.get("post_id", "?"))


def call_stances_and_ses(
    post: dict,
    comments: list[dict],
    option_risky: str,
    option_safe: str,
) -> tuple[dict | None, str | None]:
    """Call B: selftext + comments + option labels → stances + SES + summaries.

    Comments are sent here (not in Call A) so the model classifies each
    comment with full knowledge of what option_risky and option_safe mean.
    max_tokens=750 covers 25 comments × ~20 tokens + SES fields + summaries.
    """
    top = comments[:25]
    formatted = "\n".join(
        f"{i} | [{c['score']} upvotes] {c['body'][:300]}"
        for i, c in enumerate(top)
    )
    prompt = CALL_B_PROMPT.format(
        title=post.get("title", ""),
        selftext=(post.get("selftext") or "")[:2000],
        option_risky=option_risky,
        option_safe=option_safe,
        formatted_comments=formatted,
    )
    return _call_with_retry(prompt, max_tokens=750, label="call_B",
                            post_id=post.get("post_id", "?"))


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
    parser.add_argument("--rerun-errors", action="store_true",
                        help="Retry posts that previously failed with llm_parse_error, "
                             "llm_call errors, or fetch_error. Skips already-ok posts.")
    parser.add_argument("--domain", default=None,
                        choices=list(config.SUBREDDITS.keys()),
                        help="Process only this domain (useful for targeted reruns).")
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

    # Apply --domain filter
    if args.domain:
        domain_posts = {args.domain: domain_posts.get(args.domain, [])}
        log.info("Domain filter: processing only '%s'", args.domain)

    # Apply cap
    cap = 3 if args.dry_run else args.max_per_domain
    if cap is not None:
        for d in domain_posts:
            domain_posts[d] = domain_posts[d][:cap]

    # Statuses that should be retried when --rerun-errors is set
    RETRYABLE_STATUSES = {"llm_parse_error", "fetch_error", "not_decision", "ses_call_b_error"}
    # ses_call_b_error: Call A succeeded but Call B (SES) failed — retry only Call B logic

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

                # Skip already-processed posts — unless --rerun-errors is set,
                # in which case we retry posts whose status is retryable.
                if pid in checkpoint and not args.dry_run:
                    rec = checkpoint[pid]
                    existing_status = rec.get("status", "")
                    if existing_status == "ok":
                        stats[domain]["ok"] += 1
                        continue
                    if args.rerun_errors and existing_status in RETRYABLE_STATUSES:
                        log.info("  Retrying %s (prev status: %s)", pid, existing_status)
                        # Fall through to re-process
                    else:
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

                # ── B. Call A: decision gate (selftext only, no comments) ─
                llm_calls += 1
                parsed_a, err = call_decision_gate(post)
                time.sleep(LLM_SLEEP)

                if err:
                    record = {**post, "status": err}
                    stats[domain]["errors"][err] += 1
                    if not args.dry_run:
                        append_checkpoint(checkpoint_path, record)
                    continue

                if not parsed_a.get("is_genuine_decision", False):
                    record = {
                        **post,
                        "status": "not_decision",
                        "option_risky": parsed_a.get("option_risky"),
                        "option_safe":  parsed_a.get("option_safe"),
                    }
                    stats[domain]["errors"]["not_decision"] += 1
                    if not args.dry_run:
                        append_checkpoint(checkpoint_path, record)
                    continue

                option_risky = parsed_a.get("option_risky", "")
                option_safe  = parsed_a.get("option_safe", "")

                # ── C. Call B: stances + SES + summaries (with comments) ─
                # Comments sent here so model classifies each stance with
                # full knowledge of what option_risky and option_safe mean.
                llm_calls += 1
                parsed_b, err_b = call_stances_and_ses(
                    post, comments, option_risky, option_safe
                )
                time.sleep(LLM_SLEEP)

                # ── D. Compute distribution from Call B stances ────────
                if err_b:
                    log.warning("  Call B failed for %s: %s — storing partial record", pid, err_b)
                    stances     = []
                    ses_data    = {
                        "ses_cue_intensity":  None,
                        "ses_sensitivity":    None,
                        "ses_natural_cues":   [],
                        "ses_channels":       [],
                        "ses_flip_reasoning": "",
                        "risky_summary":      "",
                        "safe_summary":       "",
                        "ses_call_b_error":   err_b,
                    }
                else:
                    stances  = parsed_b.get("comments", [])
                    ses_data = {
                        "ses_cue_intensity":  parsed_b.get("ses_cue_intensity", "A0"),
                        "ses_sensitivity":    parsed_b.get("ses_sensitivity", "B0"),
                        "ses_natural_cues":   parsed_b.get("ses_natural_cues", []),
                        "ses_channels":       parsed_b.get("ses_channels", []),
                        "ses_flip_reasoning": parsed_b.get("ses_flip_reasoning", ""),
                        "risky_summary":      parsed_b.get("risky_summary", ""),
                        "safe_summary":       parsed_b.get("safe_summary", ""),
                    }

                dist = compute_distribution(comments, stances)

                record = {
                    **post,
                    "option_risky": option_risky,
                    "option_safe":  option_safe,
                    **dist,
                    **ses_data,
                    "comment_classifications": stances,
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