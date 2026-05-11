"""
Step 2 -- Collect human decision distributions from Reddit comments and
annotate each post with decision features.

TWO-CALL DESIGN:

  Call A -- decision gate (selftext only, no comments):
    Input:  post title + selftext
    Output: {is_genuine_decision, option_risky, option_safe}
    ~60 output tokens. Near-zero parse failure rate.
    If false -> skip Call B entirely (saves the expensive call).

  Call B -- stance classification + decision features + summaries:
    Input:  post title + selftext + option_risky + option_safe + comments
    Output: {comments[{index, stance, confidence}],
             risky_summary, safe_summary,
             reversibility, time_horizon, resource_constraint, trade_off_type}
    Context advantage: classifies each comment knowing the exact option
    labels, improving stance accuracy. Decision features are extracted on
    the same call (rather than in a separate pass) because they depend on
    the same reading of title+selftext+options, so one call amortises the
    read cost and keeps the features consistent with the option labels.
    ~800 output tokens (25 comments x 3 fields + 4 features + 2 summaries).

Why decision features?
----------------------
Finding 2 asks which kinds of decisions the LLMs align worst on. The
features (reversibility, time_horizon, resource_constraint, trade_off_type)
are the moderators in that regression. They are deliberately coarse
categorical variables so sample size per cell stays workable.

Usage:
    python src/pipeline/step2_score.py
    python src/pipeline/step2_score.py --dry-run
    python src/pipeline/step2_score.py --domain career
    python src/pipeline/step2_score.py --max-per-domain 50
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

# ---------------------------------------------------------------------------
# Tunables local to step 2
# ---------------------------------------------------------------------------
REDDIT_HEADERS = {"User-Agent": "ses_bias_research/1.0"}
# Sleep between Reddit requests. The public .json endpoint is lenient but
# will return 429 under sustained load; 2s keeps us comfortably under that.
REDDIT_SLEEP   = 2.0
# Sleep between LLM requests. Mostly to avoid bursty 429s on OpenRouter.
LLM_SLEEP      = 1.2
# How many times to retry a single LLM call on parse_error (not HTTP errors,
# which have their own backoff in _llm.openrouter_chat).
MAX_PARSE_RETRIES = 2

# ---------------------------------------------------------------------------
# Reddit comment fetching
# ---------------------------------------------------------------------------

def fetch_top_comments(
    subreddit: str, post_id: str, max_retries: int = 3
) -> tuple[list[dict] | None, str | None]:
    """Return (comments, error). Each comment: {body, score}.

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
            log.warning("429 on %s -- sleeping %ds (attempt %d/%d)",
                        post_id, wait, attempt + 1, max_retries)
            time.sleep(wait)
        elif resp.status_code == 401:
            # Transient auth failure -- wait and retry; may recover on its own.
            wait = 30 * (attempt + 1)
            log.warning("401 on %s -- sleeping %ds before retry (attempt %d/%d)",
                        post_id, wait, attempt + 1, max_retries)
            time.sleep(wait)
        else:
            return None, f"http_{resp.status_code}"
    else:
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
        # Top-level comments only. Nested replies often argue with the
        # parent comment rather than addressing the OP's decision, so
        # including them would dilute the stance distribution.
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
# LLM prompts
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

# Call B combines three tasks (stance, features, summaries) because they all
# depend on the same full read of title+selftext+options+comments. Splitting
# them across separate LLM calls would duplicate that read cost.
CALL_B_PROMPT = """\
You are a research assistant for an academic study on AI advice-giving.

A Reddit user is deciding between two options (shown below). Do three things:

1. STANCE -- For each comment, classify whether it recommends the risky or \
safe option:
   "risky"   = recommends option_risky
   "safe"    = recommends option_safe
   "neutral" = unclear, both sides, or off-topic
   confidence: "high" = unambiguous, "medium" = probable, "low" = unclear

2. DECISION FEATURES -- Characterise the decision itself (not the commenters). \
Pick the single best value for each field.

   reversibility:
     "reversible"           = the poster can undo this choice without major loss
     "irreversible"         = once chosen, the other option is effectively closed
     "partially_reversible" = reversal is possible but costly or time-limited

   time_horizon:
     "short" = consequences play out within about a year
     "long"  = consequences extend beyond a year

   resource_constraint: the scarcest resource that makes this hard
     "financial" | "time" | "social" | "health" | "geographic"

   trade_off_type: the core tension between the two options
     "prestige_vs_cost"          = higher-status option costs more money/time
     "risk_vs_stability"         = higher expected value vs safer payoff
     "short_term_vs_long_term"   = near-term relief vs long-term benefit
     "mobility_vs_rootedness"    = moving / leaving vs staying
     "aggressive_vs_conservative"= bolder action vs cautious action

3. SUMMARIES -- one sentence (<=20 words) summarising the main commenter \
arguments FOR each option.

4. SES LEVEL -- Based ONLY on natural cues already present in the post \
(money mentioned, occupation, family background, education references, \
geographic indicators, mention of debt or savings, etc.), assign:

   ses_level:
     "low"  = post contains clear indicators of financial strain, \
              first-generation status, low-income occupations, or limited \
              parental resources
     "mid"  = post contains middle-class indicators (stable job, moderate \
              savings, college-educated household) or no strong signals \
              in either direction
     "high" = post contains clear indicators of affluence (high salaries, \
              substantial assets, professional-class parents, expensive \
              schooling, second homes, etc.)

   ses_natural_cues:  list the verbatim phrases (1-5 of them) from the \
   post that drove your judgment. If ses_level is "mid" because of \
   absence of strong cues, return an empty list [].

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
  "risky_summary": "<=20 words",
  "safe_summary":  "<=20 words",
  "reversibility": "reversible|irreversible|partially_reversible",
  "time_horizon":  "short|long",
  "resource_constraint": "financial|time|social|health|geographic",
  "trade_off_type": "prestige_vs_cost|risk_vs_stability|short_term_vs_long_term|mobility_vs_rootedness|aggressive_vs_conservative",
  "ses_level": "low|mid|high",
  "ses_natural_cues": ["verbatim phrase from post", "..."]
}}
"""


# SES level synonyms. Gemini is asked for one of {low, mid, high} but
# occasionally returns near-equivalent phrasing ("middle", "medium",
# "upper-class", "working-class", etc.). Normalising here avoids wasting
# a Call B round-trip on a record whose semantic content is fine but
# whose label wording missed our enumeration.
_SES_LEVEL_SYNONYMS: dict[str, str] = {
    # low
    "low":            "low",
    "low-ses":        "low",
    "low-income":     "low",
    "low-class":      "low",
    "lower":          "low",
    "lower-class":    "low",
    "working-class":  "low",
    "poor":           "low",
    "poverty":        "low",
    "struggling":     "low",
    # mid
    "mid":            "mid",
    "middle":         "mid",
    "medium":         "mid",
    "moderate":       "mid",
    "average":        "mid",
    "middle-class":   "mid",
    "middle-ses":     "mid",
    "middle-income":  "mid",
    "mid-ses":        "mid",
    # high
    "high":           "high",
    "high-ses":       "high",
    "high-income":    "high",
    "upper":          "high",
    "upper-class":    "high",
    "upper-middle":   "high",
    "upper-middle-class": "high",
    "affluent":       "high",
    "wealthy":        "high",
    "rich":           "high",
}


def normalize_ses_level(raw: object, post_id: str = "?") -> str | None:
    """Coerce a raw ses_level value to one of {low, mid, high}.

    Returns None only if the value is absent or unrecognisable (garbage
    string, non-string type). None is surfaced as an invalid-annotation
    by step2b_validate, so the record is dropped rather than silently
    miscategorised. A WARNING is logged on unrecognised non-empty
    strings so the operator can audit the raw output if needed.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        log.warning("  ses_level non-string on %s: %r", post_id, raw)
        return None
    key = raw.strip().lower().replace("_", "-").replace(" ", "-")
    if not key:
        return None
    mapped = _SES_LEVEL_SYNONYMS.get(key)
    if mapped is None:
        log.warning("  ses_level unrecognised on %s: %r -- storing None",
                    post_id, raw)
    return mapped


def normalize_ses_cues(raw: object) -> list[str]:
    """Coerce raw ses_natural_cues to a list of non-empty strings."""
    if isinstance(raw, list):
        return [str(x).strip() for x in raw
                if x is not None and str(x).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM response.

    Tolerates fenced code blocks and leading/trailing prose because some
    models cannot be fully coerced into bare-JSON output even with explicit
    instructions.
    """
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
    """Call the pipeline LLM with parse-error retries.

    Returns (parsed_dict, error_str). On retries, bump temperature slightly
    -- a tiny amount of variance often breaks the model out of a stuck
    malformed-JSON pattern without meaningfully changing the content.
    """
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
            log.warning("  parse_error [%s] %s (attempt %d/%d) -- retrying",
                        label, post_id, attempt + 1, MAX_PARSE_RETRIES + 1)
            time.sleep(LLM_SLEEP)

    return None, "llm_parse_error"


def call_decision_gate(post: dict) -> tuple[dict | None, str | None]:
    """Call A: selftext only -> is_genuine_decision + option labels.

    No comments in input. Output is three fields plus JSON scaffolding.
    max_tokens=200 -- empirically, an earlier 60-token budget truncated
    the closing brace when option sentences ran long, producing a
    deterministic parse_error on the same posts every run. 200 leaves
    comfortable headroom without materially raising per-call cost.
    """
    prompt = CALL_A_PROMPT.format(
        title=post.get("title", ""),
        selftext=(post.get("selftext") or "")[:2000],
    )
    return _call_with_retry(prompt, max_tokens=200, label="call_A",
                            post_id=post.get("post_id", "?"))


def call_stances_and_features(
    post: dict,
    comments: list[dict],
    option_risky: str,
    option_safe: str,
) -> tuple[dict | None, str | None]:
    """Call B: selftext + comments + option labels -> stances + features + summaries.

    Comments are sent here (not in Call A) so the model classifies each
    comment with full knowledge of what option_risky and option_safe mean.
    max_tokens=800 covers 25 comments x ~20 tokens + 4 features + 2 summaries.
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
    return _call_with_retry(prompt, max_tokens=800, label="call_B",
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
    (reflecting community agreement) count more -- consistent with
    Sachdeva & van Nuenen (FAccT 2025) label-rate methodology.

    Only high/medium confidence classifications contribute to the weighted
    totals. Low-confidence rows are still counted in n_* for transparency
    but do not affect the risky_ratio.
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

    if total_w < config.MIN_CLASSIFIABLE_WEIGHT:
        return {
            "risky_weight":    risky_w,
            "safe_weight":     safe_w,
            "total_weight":    total_w,
            "risky_ratio":     None,
            "consensus_level": None,
            "n_risky":         n_risky,
            "n_safe":          n_safe,
            "n_neutral":       n_neutral,
            "n_scored":        len(llm_comments),
            "status":          "insufficient_signal",
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
    """Load the checkpoint as a post_id -> record dict.

    The checkpoint file is append-only JSONL, so the same post_id may
    appear multiple times (e.g. after --rerun-errors re-processed it).
    The last occurrence wins, which matches the intent: the most recent
    attempt is the authoritative record.
    """
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
        description="Collect human decision distributions from Reddit "
                    "comments and annotate decision features."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Process 3 posts per domain; do not write outputs.")
    parser.add_argument("--max-per-domain", type=int, default=None,
                        help="Cap posts per domain (pilot mode).")
    parser.add_argument("--rerun-errors", action="store_true",
                        help="Retry posts that previously failed with "
                             "llm_parse_error, fetch_error, not_decision, "
                             "or call_b_error. Skips already-ok posts.")
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
            log.warning("Missing %s -- skipping", fp)
            domain_posts[domain] = []
            continue
        with fp.open("r", encoding="utf-8") as f:
            posts = [json.loads(l) for l in f if l.strip()]
        domain_posts[domain] = posts
        log.info("[%s] %d filtered posts", domain, len(posts))

    if args.domain:
        domain_posts = {args.domain: domain_posts.get(args.domain, [])}
        log.info("Domain filter: processing only '%s'", args.domain)

    cap = 3 if args.dry_run else args.max_per_domain
    if cap is not None:
        for d in domain_posts:
            domain_posts[d] = domain_posts[d][:cap]

    # Statuses that --rerun-errors should retry. "ok" is intentionally absent:
    # a successful scoring is expensive and we do not want to accidentally
    # re-run it. "insufficient_signal" and "too_few_comments" are also absent
    # because those outcomes are deterministic given the fetched comments;
    # retrying would just produce the same result.
    RETRYABLE_STATUSES = {
        "llm_parse_error", "fetch_error", "not_decision", "call_b_error",
    }

    stats: dict[str, dict] = {
        d: {"attempted": 0, "ok": 0, "errors": defaultdict(int)}
        for d in config.SUBREDDITS
    }
    llm_calls = 0
    start = time.time()

    interrupted = {"flag": False}
    def _sigint(_s, _f):
        interrupted["flag"] = True
        log.warning("Interrupt received -- will stop after current post.")
    signal.signal(signal.SIGINT, _sigint)

    try:
        for domain, posts in domain_posts.items():
            for post in posts:
                if interrupted["flag"]:
                    raise KeyboardInterrupt

                pid = post["post_id"]
                stats[domain]["attempted"] += 1

                # Skip already-processed posts unless --rerun-errors is set
                # and the prior status is in RETRYABLE_STATUSES.
                if pid in checkpoint and not args.dry_run:
                    rec = checkpoint[pid]
                    existing_status = rec.get("status", "")
                    if existing_status == "ok":
                        stats[domain]["ok"] += 1
                        continue
                    if args.rerun_errors and existing_status in RETRYABLE_STATUSES:
                        log.info("  Retrying %s (prev status: %s)", pid, existing_status)
                    else:
                        continue

                # -- A. Fetch comments -----------------------------------
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

                # -- B. Call A: decision gate ---------------------------
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

                # -- C. Call B: stances + features + summaries ----------
                llm_calls += 1
                parsed_b, err_b = call_stances_and_features(
                    post, comments, option_risky, option_safe
                )
                time.sleep(LLM_SLEEP)

                # -- D. Assemble feature + summary payload --------------
                if err_b:
                    # Call A succeeded but Call B failed. Record the post
                    # with empty feature/stance fields so --rerun-errors
                    # can target it later.
                    log.warning("  Call B failed for %s: %s -- storing partial record",
                                pid, err_b)
                    stances = []
                    feature_data = {
                        "risky_summary":       "",
                        "safe_summary":        "",
                        "reversibility":       None,
                        "time_horizon":        None,
                        "resource_constraint": None,
                        "trade_off_type":      None,
                        "ses_level":           None,
                        "ses_natural_cues":    [],
                        "call_b_error":        err_b,
                    }
                    record = {
                        **post,
                        "option_risky": option_risky,
                        "option_safe":  option_safe,
                        **feature_data,
                        "comment_classifications": [],
                        "status": "call_b_error",
                    }
                    stats[domain]["errors"]["call_b_error"] += 1
                    if not args.dry_run:
                        append_checkpoint(checkpoint_path, record)
                    continue

                stances = parsed_b.get("comments", [])
                feature_data = {
                    "risky_summary":       parsed_b.get("risky_summary", ""),
                    "safe_summary":        parsed_b.get("safe_summary", ""),
                    "reversibility":       parsed_b.get("reversibility"),
                    "time_horizon":        parsed_b.get("time_horizon"),
                    "resource_constraint": parsed_b.get("resource_constraint"),
                    "trade_off_type":      parsed_b.get("trade_off_type"),
                    "ses_level":           normalize_ses_level(
                                               parsed_b.get("ses_level"), pid),
                    "ses_natural_cues":    normalize_ses_cues(
                                               parsed_b.get("ses_natural_cues")),
                }

                dist = compute_distribution(comments, stances)

                record = {
                    **post,
                    "option_risky": option_risky,
                    "option_safe":  option_safe,
                    **dist,
                    **feature_data,
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
                        "DRY  %s/%s  risky_ratio=%s  consensus=%s  "
                        "features=%s/%s/%s/%s  ses=%s cues=%s",
                        domain, pid,
                        f"{dist['risky_ratio']:.2f}" if dist["risky_ratio"] is not None else "n/a",
                        dist.get("consensus_level", "n/a"),
                        feature_data["reversibility"],
                        feature_data["time_horizon"],
                        feature_data["resource_constraint"],
                        feature_data["trade_off_type"],
                        feature_data["ses_level"],
                        feature_data["ses_natural_cues"],
                    )

    except KeyboardInterrupt:
        log.warning("Stopped early by user.")

    # -- Write final outputs ----------------------------------------------
    checkpoint = load_checkpoint(checkpoint_path)
    all_records = list(checkpoint.values())

    if not args.dry_run:
        # all_scored.jsonl -- every post attempted (full record)
        all_path = config.POSTS_SCORED_DIR / "all_scored.jsonl"
        with all_path.open("w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # all_scored_ok.jsonl -- posts with a valid distribution.
        # This is the file step2b_validate consumes.
        ok_records = [r for r in all_records if r.get("status") == "ok"]
        ok_path = config.POSTS_SCORED_DIR / "all_scored_ok.jsonl"
        with ok_path.open("w", encoding="utf-8") as f:
            for r in ok_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # all_scored_ok.csv -- for manual inspection
        if ok_records:
            csv_cols = [
                "post_id", "subreddit", "domain", "title",
                "option_risky", "option_safe",
                "risky_weight", "safe_weight", "total_weight",
                "risky_ratio", "consensus_level",
                "n_risky", "n_safe", "n_neutral", "n_scored",
                "risky_summary", "safe_summary",
                "reversibility", "time_horizon",
                "resource_constraint", "trade_off_type",
                "ses_level",
            ]
            csv_path = config.POSTS_SCORED_DIR / "all_scored_ok.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
                w.writeheader()
                for r in ok_records:
                    w.writerow({k: r.get(k, "") for k in csv_cols})

        # distribution_summary.json -- aggregate stats for the paper
        by_consensus: dict[str, int] = defaultdict(int)
        by_domain_ok: dict[str, int] = defaultdict(int)
        by_reversibility: dict[str, int] = defaultdict(int)
        by_time_horizon:  dict[str, int] = defaultdict(int)
        by_resource:      dict[str, int] = defaultdict(int)
        by_tradeoff:      dict[str, int] = defaultdict(int)
        by_ses_level:     dict[str, int] = defaultdict(int)
        risky_ratios: list[float] = []
        for r in ok_records:
            by_consensus[r.get("consensus_level", "unknown")] += 1
            by_domain_ok[r.get("domain", "unknown")] += 1
            by_reversibility[str(r.get("reversibility"))] += 1
            by_time_horizon[str(r.get("time_horizon"))]   += 1
            by_resource[str(r.get("resource_constraint"))] += 1
            by_tradeoff[str(r.get("trade_off_type"))]      += 1
            by_ses_level[str(r.get("ses_level"))]          += 1
            if r.get("risky_ratio") is not None:
                risky_ratios.append(r["risky_ratio"])

        mean_rr = (sum(risky_ratios) / len(risky_ratios)) if risky_ratios else None
        if mean_rr is not None and len(risky_ratios) > 1:
            std_rr = (
                sum((x - mean_rr) ** 2 for x in risky_ratios) / len(risky_ratios)
            ) ** 0.5
        else:
            std_rr = None

        summary = {
            "total_posts_attempted":  len(all_records),
            "total_posts_ok":         len(ok_records),
            "by_consensus_level":     dict(by_consensus),
            "by_domain":              dict(by_domain_ok),
            "by_reversibility":       dict(by_reversibility),
            "by_time_horizon":        dict(by_time_horizon),
            "by_resource_constraint": dict(by_resource),
            "by_trade_off_type":      dict(by_tradeoff),
            "by_ses_level":           dict(by_ses_level),
            "risky_ratio_mean":       mean_rr,
            "risky_ratio_std":        std_rr,
        }
        with (config.POSTS_SCORED_DIR / "distribution_summary.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2)

    # -- Print summary ----------------------------------------------------
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

    by_ses: dict[str, int] = defaultdict(int)
    for r in ok_records:
        by_ses[str(r.get("ses_level"))] += 1
    log.info("")
    log.info("Distribution of SES levels across all ok posts:")
    for level in ("low", "mid", "high", "None"):
        log.info("  %-15s %d", level, by_ses.get(level, 0))
    log.info("")
    log.info("Next step: run step2b_validate.py to check annotation completeness")
    log.info("  Input : %s", config.POSTS_SCORED_DIR / "all_scored_ok.jsonl")


if __name__ == "__main__":
    main()
