"""
Step 5 -- LLM experiment: query four study models on unmodified Reddit posts
and compare their choice distributions against the human risky_ratio from
Step 2.

For each post in all_scored_valid.jsonl, this script builds a single
forced-choice prompt that pins "Option A" to option_risky and "Option B"
to option_safe (as identified by Step 2). It then queries each of the four
study models N_SAMPLES times at temperature 0.7, parses every response to
A / B / unparseable, and aggregates into per-cell and overall tables.

The explicit A/B mapping is what makes cross-model comparison valid: all
four models see the same assignment, and the parse_choice output can be
interpreted the same way across models.

Provider routing is handled in config.STUDY_MODELS -- gpt goes directly
to the OpenAI API (cheaper, more direct) and the other three go through
OpenRouter. Both speak the OpenAI /chat/completions schema, so one call
helper handles both.

Output files (under data/experiment/):
    raw_responses.jsonl            -- one row per individual API call
    raw_responses_checkpoint.jsonl -- append-only checkpoint for resume
    parse_failures.jsonl           -- every unparseable response
    choice_rates.jsonl             -- one row per (post x model)
    comparison_table.csv           -- analysis-ready table (main output)
    experiment_summary.json        -- high-level run stats

Usage:
    python src/pipeline/step5_experiment.py
    python src/pipeline/step5_experiment.py --dry-run
    python src/pipeline/step5_experiment.py --model gpt
    python src/pipeline/step5_experiment.py --post-id abc123
    python src/pipeline/step5_experiment.py --n-samples 3
    python src/pipeline/step5_experiment.py --workers 4
    python src/pipeline/step5_experiment.py --max-posts 50
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import signal
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from _llm import LLMError, chat_completion

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
config.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "step5_experiment.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("step5_experiment")

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RAW_PATH          = config.EXPERIMENT_DIR / "raw_responses.jsonl"
CHECKPOINT_PATH   = config.EXPERIMENT_DIR / "raw_responses_checkpoint.jsonl"
PARSE_FAIL_PATH   = config.EXPERIMENT_DIR / "parse_failures.jsonl"
CHOICE_RATES_PATH = config.EXPERIMENT_DIR / "choice_rates.jsonl"
COMPARISON_CSV    = config.EXPERIMENT_DIR / "comparison_table.csv"
SUMMARY_JSON      = config.EXPERIMENT_DIR / "experiment_summary.json"

INPUT_JSONL = config.POSTS_SCORED_DIR / "all_scored_valid.jsonl"

# File locks for concurrent appends from the ThreadPoolExecutor
_CHECKPOINT_LOCK = threading.Lock()
_PARSEFAIL_LOCK  = threading.Lock()

# Cap on total attempts (first try + retries) for a sample whose earlier
# attempts came back non-committal. API errors are transient and still
# retry without limit, but a model that reasons instead of answering is
# usually deterministic -- re-running the same (post, model, sample_index)
# ten times at T=0.7 will mostly produce the same style of reasoning and
# burn cost for no information gain. MAX_PARSE_ATTEMPTS = 2 gives every
# sample one chance to land differently, then writes it off as missing.
MAX_PARSE_ATTEMPTS = 2

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def clean_post_text(title: str, selftext: str) -> str:
    """Light cleanup only. Preserves emotional content, typos, digressions
    so the LLM sees roughly what a human Reddit reader would see.
    """
    text = re.sub(r"https?://\S+", "[link]", selftext or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return f"{title.strip()}\n\n{text.strip()}"


def build_prompt(post: dict) -> str:
    """Build the forced-choice prompt for a single post.

    Locks Option A -> option_risky, Option B -> option_safe. This mapping is
    the same across all four study models, so parsed A/B counts are directly
    comparable.
    """
    body = clean_post_text(post.get("title", ""), post.get("selftext", ""))
    return (
        f"{body}\n\n"
        f"The two options being considered are:\n"
        f"Option A: {post.get('option_risky', '')}\n"
        f"Option B: {post.get('option_safe', '')}\n\n"
        f"{config.FORCED_CHOICE_INSTRUCTION}"
    )


def prompt_hash(prompt: str) -> str:
    """Short sha1 so raw_responses rows are small and still unique enough
    to detect prompt drift between reruns (we would notice if a post's
    text or options changed upstream and the hash flipped).
    """
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# A choice is ONLY accepted when the model commits to one on its own line
# (or the response is exactly a choice). Accepted forms, case-insensitive,
# with optional markdown bold wrappers and optional trailing period/bang:
#     "A"            "B"
#     "Option A"     "Option B."
#     "**Option A**" "*B.*"
#
# Explicitly rejected:
#     "Option A:" (colon = section header)
#     "(Option A)" (parenthetical reference)
#     "Option A: Attend..." (prefix of a bullet / header)
#     descriptive mentions embedded in reasoning prose
#
# This strictness is deliberate. The pilot run surfaced a failure mode
# where a reasoning model emitted section headers like "**Option A:
# Attend an out-of-state university**" and a lenient substring regex
# classified it as a vote for A -- biasing the data toward whichever
# option the model analyses first. Treating those as unparseable is
# safer than treating them as a committal answer.
_CHOICE_PATTERN = re.compile(
    r"^\s*\*{0,2}\s*(?:option\s+)?([AB])\s*\*{0,2}\s*[.!]?\s*$",
    re.IGNORECASE,
)


def parse_choice(response: str | None) -> str | None:
    """Return 'A', 'B', or None (unparseable / non-committal).

    A response qualifies if (a) the whole trimmed response matches the
    strict choice pattern, OR (b) the first non-empty line matches, OR
    (c) the last non-empty line matches. These three positions cover
    the natural places a committal answer appears: a one-word reply,
    a leading answer before reasoning, or a trailing answer after it.
    """
    if not response:
        return None
    s = response.strip()

    m = _CHOICE_PATTERN.match(s)
    if m:
        return m.group(1).upper()

    lines = [l.strip() for l in s.split("\n") if l.strip()]
    if lines:
        for candidate in (lines[0], lines[-1]):
            m = _CHOICE_PATTERN.match(candidate)
            if m:
                return m.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Information-theoretic helpers
# ---------------------------------------------------------------------------

def binary_entropy(p: float | None) -> float:
    """Shannon entropy of Bernoulli(p) in nats. Returns 0 for p in {0, 1}
    or None. Used to measure how 'collapsed' a choice distribution is --
    an LLM that always says A has entropy 0, a coin flip has entropy ln 2.
    """
    if p is None or p <= 0 or p >= 1:
        return 0.0
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def binary_kl(p: float | None, q: float | None, eps: float = 1e-3) -> float | None:
    """KL(Bernoulli(p) || Bernoulli(q)) in nats, with Laplace-style clamping
    on both probabilities so divide-by-zero is impossible. Returns None if
    either input is None. eps=1e-3 caps KL at ~6.9 nats in the worst case,
    which is a reasonable ceiling for a single-post divergence.
    """
    if p is None or q is None:
        return None
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


# ---------------------------------------------------------------------------
# Single model call
# ---------------------------------------------------------------------------

def call_model(
    model_key: str,
    prompt: str,
    temperature: float,
    max_tokens: int = 100,
) -> tuple[str | None, int, str | None]:
    """Call one study model. Returns (response_text, latency_ms, error).

    Routes to OpenAI or OpenRouter based on config.STUDY_MODELS[model_key]
    ['provider']. Both share the /chat/completions schema, so the only
    per-provider differences are the api_key, base_url, and the OpenRouter
    identification headers.

    Reasoning-control is per-model via config.STUDY_MODELS[...]["extra_body"]
    because providers disagree on the knob: GPT-5 uses OpenAI's native
    reasoning_effort, Claude is a thinking model on OpenRouter that needs
    the unified {"reasoning": {"enabled": False}} to keep the 100-token
    budget from being swallowed by its thinking pass, and Mistral/Llama
    have no reasoning mode so their extra_body is empty.
    """
    cfg = config.STUDY_MODELS[model_key]
    extra: dict[str, str] = {}
    if cfg["provider"] == "openrouter":
        extra["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER", "") or ""
        extra["X-Title"]      = os.getenv("OPENROUTER_X_TITLE",      "") or ""

    t0 = time.time()
    try:
        text = chat_completion(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            model=cfg["model_name"],
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=60.0,
            extra_headers=extra,
            extra_body=cfg.get("extra_body") or {},
        )
        return text, int((time.time() - t0) * 1000), None
    except LLMError as e:
        return None, int((time.time() - t0) * 1000), str(e)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: Path,
) -> tuple[dict[tuple[str, int], dict], dict[tuple[str, int], int]]:
    """Return (done, parse_fail_counts).

    done: {(variant_id, sample_index): latest_record}. Later rows win,
    matching the append-only semantics so a rerun for an errored sample
    overwrites the earlier failure record in the in-memory dict.

    parse_fail_counts: {(variant_id, sample_index): count of historical
    parse failures}. Used to cap retries on samples that consistently
    fail to commit. An "API error" attempt does not increment this count
    because API errors are transient and should retry indefinitely.
    """
    done: dict[tuple[str, int], dict] = {}
    parse_fail_counts: dict[tuple[str, int], int] = defaultdict(int)
    if not path.exists():
        return done, parse_fail_counts
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec["variant_id"], int(rec["sample_index"]))
                done[key] = rec
                if rec.get("error") is None \
                        and rec.get("parsed_choice") not in ("A", "B"):
                    parse_fail_counts[key] += 1
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done, parse_fail_counts


def append_jsonl(path: Path, record: dict, lock: threading.Lock) -> None:
    """Thread-safe append of a single JSONL record."""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


# ---------------------------------------------------------------------------
# Cost projection (very rough -- actual cost depends on token counts
# and provider pricing, which change frequently)
# ---------------------------------------------------------------------------

# Approximate USD per call with ~400 input tokens + ~5 output tokens, as of
# May 2026. These are deliberately round ballpark numbers; real billing
# should come from the provider dashboard.
_ROUGH_COST_PER_CALL_USD = {
    "gpt":     0.0005,
    "claude":  0.0012,
    "mistral": 0.0008,
    "llama":   0.0001,
}


def project_cost(n_posts: int, model_keys: list[str], n_samples: int) -> dict:
    """Return a dict with total calls and a rough cost estimate."""
    per_model_calls = n_posts * n_samples
    total_calls     = per_model_calls * len(model_keys)
    est_usd = sum(
        per_model_calls * _ROUGH_COST_PER_CALL_USD.get(mk, 0.001)
        for mk in model_keys
    )
    return {
        "per_model_calls": per_model_calls,
        "total_calls":     total_calls,
        "est_usd":         est_usd,
    }


# ---------------------------------------------------------------------------
# Task processing (thread-pool unit of work)
# ---------------------------------------------------------------------------

def process_task(
    post: dict,
    model_key: str,
    sample_index: int,
    prompt: str,
    p_hash: str,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
) -> dict:
    """Call the model once, parse, and return a record ready for the
    checkpoint. Parse failures are logged via a side-channel file by the
    orchestrator; this function just returns the record.
    """
    variant_id = f"{post['post_id']}__{model_key}"
    if dry_run:
        return {
            "variant_id":    variant_id,
            "post_id":       post["post_id"],
            "model_key":     model_key,
            "sample_index":  sample_index,
            "prompt_hash":   p_hash,
            "raw_response":  "[DRY RUN]",
            "parsed_choice": None,
            "latency_ms":    0,
            "error":         None,
        }

    text, latency_ms, err = call_model(model_key, prompt, temperature, max_tokens)
    parsed = parse_choice(text) if err is None else None

    return {
        "variant_id":    variant_id,
        "post_id":       post["post_id"],
        "model_key":     model_key,
        "sample_index":  sample_index,
        "prompt_hash":   p_hash,
        "raw_response":  text,
        "parsed_choice": parsed,
        "latency_ms":    latency_ms,
        "error":         err,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_per_cell(records: list[dict]) -> dict[tuple[str, str], dict]:
    """Group raw records by (post_id, model_key) and produce a per-cell
    summary: counts, rate, entropy. Error rows count toward n_attempted
    but not toward n_valid.
    """
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_cell[(r["post_id"], r["model_key"])].append(r)

    out: dict[tuple[str, str], dict] = {}
    for key, cell in by_cell.items():
        n_attempted = len(cell)
        valid = [r for r in cell if r["parsed_choice"] in ("A", "B")]
        n_valid = len(valid)
        n_a = sum(1 for r in valid if r["parsed_choice"] == "A")
        n_b = sum(1 for r in valid if r["parsed_choice"] == "B")

        llm_risky_rate = (n_a / n_valid) if n_valid > 0 else None
        llm_entropy    = binary_entropy(llm_risky_rate)

        out[key] = {
            "post_id":        key[0],
            "model_key":      key[1],
            "n_attempted":    n_attempted,
            "n_valid":        n_valid,
            "n_a":            n_a,
            "n_b":            n_b,
            "llm_risky_rate": llm_risky_rate,
            "llm_entropy":    llm_entropy,
        }
    return out


def build_comparison_rows(
    posts_by_id: dict[str, dict],
    cells: dict[tuple[str, str], dict],
) -> list[dict]:
    """Join per-cell aggregates with the Step 2 human distribution and
    compute alignment-gap metrics.
    """
    rows: list[dict] = []
    for (pid, mk), cell in cells.items():
        post = posts_by_id.get(pid)
        if not post:
            continue
        human_rr = post.get("risky_ratio")
        llm_rr   = cell["llm_risky_rate"]

        mean_gap = (llm_rr - human_rr) if (llm_rr is not None and human_rr is not None) else None
        abs_gap  = abs(mean_gap) if mean_gap is not None else None
        h_ent    = binary_entropy(human_rr)
        l_ent    = cell["llm_entropy"]
        ent_gap  = (h_ent - l_ent) if (h_ent is not None and l_ent is not None) else None
        kl       = binary_kl(human_rr, llm_rr)

        qualified = 1 if cell["n_valid"] >= config.MIN_VALID_SAMPLES else 0

        rows.append({
            "post_id":             pid,
            "domain":              post.get("domain"),
            "consensus_level":     post.get("consensus_level"),
            "ses_level":           post.get("ses_level"),
            "reversibility":       post.get("reversibility"),
            "time_horizon":        post.get("time_horizon"),
            "resource_constraint": post.get("resource_constraint"),
            "trade_off_type":      post.get("trade_off_type"),
            "model":               mk,
            "human_risky_ratio":   human_rr,
            "llm_risky_rate":      llm_rr,
            "n_valid":             cell["n_valid"],
            "mean_gap":            mean_gap,
            "abs_gap":             abs_gap,
            "human_entropy":       h_ent,
            "llm_entropy":         l_ent,
            "entropy_gap":         ent_gap,
            "kl_divergence":       kl,
            "qualified_for_analysis": qualified,
        })
    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_raw_responses(records: dict[tuple[str, int], dict], path: Path) -> None:
    """Write the deduped checkpoint to raw_responses.jsonl, sorted for
    stable diffs. The checkpoint file is the in-flight log (may have
    duplicate keys from reruns); this is the clean audit file.
    """
    sorted_records = sorted(
        records.values(),
        key=lambda r: (r["post_id"], r["model_key"], r["sample_index"]),
    )
    with path.open("w", encoding="utf-8") as f:
        for r in sorted_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_choice_rates(cells: dict[tuple[str, str], dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for (_, _), cell in sorted(cells.items()):
            f.write(json.dumps(cell, ensure_ascii=False) + "\n")


_CSV_COLS = [
    "post_id", "domain", "consensus_level", "ses_level",
    "reversibility", "time_horizon", "resource_constraint", "trade_off_type",
    "model",
    "human_risky_ratio",
    "llm_risky_rate",
    "n_valid",
    "mean_gap",
    "abs_gap",
    "human_entropy",
    "llm_entropy",
    "entropy_gap",
    "kl_divergence",
    "qualified_for_analysis",
]


def write_comparison_csv(rows: list[dict], path: Path) -> None:
    rows = sorted(
        rows,
        key=lambda r: (
            r.get("domain") or "",
            r.get("consensus_level") or "",
            r.get("post_id") or "",
            r.get("model") or "",
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({
                k: (f"{r[k]:.6f}" if isinstance(r.get(k), float) else
                    ("" if r.get(k) is None else r.get(k)))
                for k in _CSV_COLS
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 5: query four study models on unmodified Reddit "
                    "posts and compare against human Step 2 distribution."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without calling any API or writing outputs.")
    parser.add_argument("--model", choices=list(config.STUDY_MODELS.keys()),
                        default=None,
                        help="Run only this model key (default: all four).")
    parser.add_argument("--post-id", default=None,
                        help="Run only this post_id (spot check).")
    parser.add_argument("--n-samples", type=int, default=None,
                        help=f"Override N_SAMPLES (default: {config.EXPERIMENT_N_SAMPLES}).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Thread-pool size for parallel API calls (default: 8).")
    parser.add_argument("--max-posts", type=int, default=None,
                        help="Cap number of posts (cost control).")
    args = parser.parse_args()

    n_samples   = args.n_samples if args.n_samples is not None else config.EXPERIMENT_N_SAMPLES
    temperature = config.EXPERIMENT_TEMPERATURE

    # -- Load posts --------------------------------------------------------
    if not INPUT_JSONL.exists():
        log.error("Missing %s -- run step2_score.py and step2b_validate.py first", INPUT_JSONL)
        sys.exit(1)

    # all_posts / all_posts_by_id is the full unfiltered reference, used
    # when rebuilding the output files so prior-run data for other posts
    # or other models is not lost. posts / posts_by_id below is the
    # filtered run scope, used only to decide which API calls to make.
    all_posts: list[dict] = []
    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_posts.append(json.loads(line))
    all_posts_by_id = {p["post_id"]: p for p in all_posts}
    log.info("Loaded %d posts from %s", len(all_posts), INPUT_JSONL.name)

    posts = list(all_posts)
    if args.post_id:
        posts = [p for p in posts if p.get("post_id") == args.post_id]
        log.info("Post-id filter: %d post(s) remaining", len(posts))
    if args.max_posts is not None:
        posts = posts[:args.max_posts]
        log.info("max-posts cap: %d posts", len(posts))

    posts_by_id = {p["post_id"]: p for p in posts}

    # -- Pick models -------------------------------------------------------
    model_keys = [args.model] if args.model else list(config.STUDY_MODELS.keys())
    log.info("Models: %s", model_keys)

    # -- Validate API keys we actually need --------------------------------
    if not args.dry_run:
        missing: list[str] = []
        for mk in model_keys:
            cfg = config.STUDY_MODELS[mk]
            if not cfg.get("api_key"):
                missing.append(f"{mk} ({cfg['provider']})")
        if missing:
            log.error("Missing API key(s) for: %s", ", ".join(missing))
            log.error("Check .env: OPENAI_API_KEY for gpt, OPENROUTER_API_KEY for others")
            sys.exit(1)

    # -- Build task list ---------------------------------------------------
    if args.dry_run:
        done: dict[tuple[str, int], dict] = {}
        parse_fail_counts: dict[tuple[str, int], int] = defaultdict(int)
    else:
        done, parse_fail_counts = load_checkpoint(CHECKPOINT_PATH)
    log.info("Checkpoint: %d records already completed", len(done))

    n_skip_clean = 0
    n_skip_parse_cap = 0
    tasks: list[tuple[dict, str, int, str, str]] = []  # (post, model, idx, prompt, hash)
    for post in posts:
        prompt = build_prompt(post)
        p_hash = prompt_hash(prompt)
        for mk in model_keys:
            variant_id = f"{post['post_id']}__{mk}"
            for i in range(n_samples):
                key = (variant_id, i)
                rec = done.get(key)
                if rec is not None:
                    if rec.get("error") is None \
                            and rec.get("parsed_choice") in ("A", "B"):
                        # Clean sample -- keep it, don't retry.
                        n_skip_clean += 1
                        continue
                    if rec.get("error") is None \
                            and parse_fail_counts[key] >= MAX_PARSE_ATTEMPTS:
                        # Non-committal and already past the retry cap --
                        # accept it as missing data; don't burn more money.
                        n_skip_parse_cap += 1
                        continue
                    # Otherwise fall through: either an API error (always
                    # retry) or parse-failed but still under the cap.
                tasks.append((post, mk, i, prompt, p_hash))

    total_pending = len(tasks)
    log.info("Skipping: %d clean + %d at parse-retry cap",
             n_skip_clean, n_skip_parse_cap)
    log.info("Pending API calls: %d", total_pending)

    # -- Cost projection ---------------------------------------------------
    proj = project_cost(len(posts), model_keys, n_samples)
    log.info("")
    log.info("=" * 60)
    log.info("COST PROJECTION")
    log.info("  Posts                : %d", len(posts))
    log.info("  Models               : %d (%s)", len(model_keys), ",".join(model_keys))
    log.info("  Samples per cell     : %d", n_samples)
    log.info("  Calls per model      : %d", proj["per_model_calls"])
    log.info("  Total API calls      : %d", proj["total_calls"])
    log.info("  Rough estimate (USD) : $%.2f", proj["est_usd"])
    log.info("  NOTE: estimate is a ballpark -- real cost depends on token")
    log.info("        counts and current provider pricing. Check dashboards.")
    log.info("=" * 60)

    if args.dry_run:
        log.info("DRY RUN -- no API calls. Sample prompt for first post:")
        if posts:
            sample_prompt = build_prompt(posts[0])
            log.info("----- PROMPT START -----")
            for ln in sample_prompt.splitlines():
                log.info("  %s", ln)
            log.info("----- PROMPT END -----")
        return

    if total_pending == 0:
        log.info("Nothing to do (checkpoint already complete). Rebuilding outputs.")
    else:
        log.info("Starting in 10 seconds -- press Ctrl+C to abort.")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            log.info("Aborted before start.")
            return

    # -- Signal handling ---------------------------------------------------
    interrupted = threading.Event()
    def _sigint(_s, _f):
        if not interrupted.is_set():
            log.warning("Interrupt received -- finishing in-flight calls and stopping.")
            interrupted.set()
    signal.signal(signal.SIGINT, _sigint)

    # -- Run ---------------------------------------------------------------
    start = time.time()
    completed = 0
    model_progress: Counter = Counter()
    per_model_pending = Counter(t[1] for t in tasks)
    parse_failures_this_run = 0
    api_errors_this_run = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        fut_to_task = {
            ex.submit(
                process_task, post, mk, i, prompt, p_hash,
                temperature, 100, False,
            ): (post["post_id"], mk, i)
            for (post, mk, i, prompt, p_hash) in tasks
        }

        for fut in as_completed(fut_to_task):
            if interrupted.is_set():
                # Let in-flight finish; stop dispatching is automatic with
                # as_completed since we already submitted everything. We
                # break out of the consumption loop to skip remaining.
                for f in fut_to_task:
                    if not f.running() and not f.done():
                        f.cancel()
                # Continue processing results that do come back to keep
                # the checkpoint up to date.
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                pid, mk, i = fut_to_task[fut]
                log.error("Unexpected exception for %s/%s/%d: %s", pid, mk, i, e)
                continue

            append_jsonl(CHECKPOINT_PATH, rec, _CHECKPOINT_LOCK)

            if rec["error"]:
                api_errors_this_run += 1
                log.warning("API error [%s/%s/%d]: %s",
                            rec["post_id"], rec["model_key"], rec["sample_index"],
                            rec["error"])
            elif rec["parsed_choice"] not in ("A", "B"):
                parse_failures_this_run += 1
                append_jsonl(PARSE_FAIL_PATH, {
                    "variant_id":   rec["variant_id"],
                    "post_id":      rec["post_id"],
                    "model_key":    rec["model_key"],
                    "sample_index": rec["sample_index"],
                    "prompt_hash":  rec["prompt_hash"],
                    "raw_response": rec["raw_response"],
                }, _PARSEFAIL_LOCK)
                log.warning("Parse failure [%s/%s/%d]: %r",
                            rec["post_id"], rec["model_key"], rec["sample_index"],
                            (rec["raw_response"] or "")[:120])

            completed += 1
            model_progress[rec["model_key"]] += 1
            if completed % 100 == 0 or completed == total_pending:
                per_model = "  ".join(
                    f"{mk}: {model_progress[mk]}/{per_model_pending[mk]}"
                    for mk in model_keys
                )
                pct = 100.0 * completed / max(total_pending, 1)
                log.info("Progress: %d/%d calls (%.1f%%) | %s",
                         completed, total_pending, pct, per_model)

    # -- Aggregate ---------------------------------------------------------
    # The output files are a complete view of the checkpoint, not just of
    # this run's scope. If we scoped to posts_by_id / model_keys here, a
    # per-model pilot (e.g. --model gpt after a --model llama pilot)
    # would overwrite the llama rows with a gpt-only table. Instead, we
    # include every record whose post_id is in all_scored_valid.jsonl
    # and whose model_key is a recognised study model. Prior-run data is
    # always preserved; the current run simply appends.
    log.info("")
    log.info("Aggregating results...")
    all_done, _ = load_checkpoint(CHECKPOINT_PATH)
    all_model_keys = set(config.STUDY_MODELS.keys())
    cumulative = {k: v for k, v in all_done.items()
                  if v["post_id"] in all_posts_by_id
                  and v["model_key"] in all_model_keys}
    log.info("  Records in checkpoint (all runs): %d", len(cumulative))

    cells = aggregate_per_cell(list(cumulative.values()))
    comparison_rows = build_comparison_rows(all_posts_by_id, cells)

    # -- Write outputs -----------------------------------------------------
    write_raw_responses(cumulative, RAW_PATH)
    log.info("  Wrote %s (%d rows)", RAW_PATH.name, len(cumulative))
    write_choice_rates(cells, CHOICE_RATES_PATH)
    log.info("  Wrote %s (%d rows)", CHOICE_RATES_PATH.name, len(cells))
    write_comparison_csv(comparison_rows, COMPARISON_CSV)
    log.info("  Wrote %s (%d rows)", COMPARISON_CSV.name, len(comparison_rows))

    # -- Summary -----------------------------------------------------------
    # Like the output files, the summary is cumulative across all runs.
    # Per-model stats include every model that has at least one record
    # in the checkpoint, not just the model(s) targeted this run.
    elapsed = time.time() - start
    total_calls_in_scope = len(cumulative)
    models_with_data = sorted({r["model_key"] for r in cumulative.values()})
    posts_with_data  = sorted({r["post_id"]   for r in cumulative.values()})
    by_model_stats: dict[str, dict] = {}
    for mk in models_with_data:
        mk_records = [r for r in cumulative.values() if r["model_key"] == mk]
        mk_errors  = sum(1 for r in mk_records if r["error"])
        mk_parse   = sum(1 for r in mk_records
                         if r["error"] is None and r["parsed_choice"] not in ("A", "B"))
        by_model_stats[mk] = {
            "n_calls":        len(mk_records),
            "api_errors":     mk_errors,
            "parse_failures": mk_parse,
            "valid":          len(mk_records) - mk_errors - mk_parse,
        }

    qualified = sum(1 for r in comparison_rows if r["qualified_for_analysis"] == 1)
    total_api_errors = sum(s["api_errors"]     for s in by_model_stats.values())
    total_parse_fail = sum(s["parse_failures"] for s in by_model_stats.values())
    total_successful = total_calls_in_scope - total_api_errors - total_parse_fail

    summary = {
        "total_posts":            len(posts_with_data),
        "total_models":           len(models_with_data),
        "models":                 models_with_data,
        "n_samples_per_cell":     n_samples,
        "total_api_calls":        total_calls_in_scope,
        "successful_calls":       total_successful,
        "api_errors":             total_api_errors,
        "parse_failures":         total_parse_fail,
        "failure_rate":           ((total_api_errors + total_parse_fail)
                                   / total_calls_in_scope) if total_calls_in_scope else 0.0,
        "by_model":               by_model_stats,
        "qualified_cells":        qualified,
        "unqualified_cells":      len(comparison_rows) - qualified,
        "elapsed_seconds":        round(elapsed, 1),
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("  Wrote %s", SUMMARY_JSON.name)

    # -- Human summary -----------------------------------------------------
    log.info("")
    log.info("=" * 60)
    log.info("STEP 5 COMPLETE in %.1fs  (stats below are CUMULATIVE across all runs)",
             elapsed)
    log.info("  Posts with data    : %d", summary["total_posts"])
    log.info("  Models with data   : %d  (%s)", summary["total_models"],
             ", ".join(models_with_data))
    log.info("  Samples per cell   : %d", n_samples)
    log.info("  Total API calls    : %d", total_calls_in_scope)
    log.info("  Successful         : %d", total_successful)
    log.info("  API errors         : %d", total_api_errors)
    log.info("  Parse failures     : %d", total_parse_fail)
    log.info("  Failure rate       : %.3f%%", summary["failure_rate"] * 100)
    log.info("  Qualified cells    : %d / %d", qualified, len(comparison_rows))
    log.info("")
    log.info("  Per-model breakdown (cumulative):")
    for mk in models_with_data:
        s = by_model_stats[mk]
        log.info("    %-8s n_calls=%-6d valid=%-6d api_errors=%-4d parse_fail=%-4d",
                 mk, s["n_calls"], s["valid"], s["api_errors"], s["parse_failures"])
    log.info("")
    log.info("Outputs under %s/:", config.EXPERIMENT_DIR)
    log.info("  raw_responses.jsonl        -- every individual API call")
    log.info("  raw_responses_checkpoint.jsonl -- append-only log (do not delete)")
    log.info("  parse_failures.jsonl       -- audit trail of unparseable responses")
    log.info("  choice_rates.jsonl         -- per (post x model) aggregates")
    log.info("  comparison_table.csv       -- MAIN ANALYSIS TABLE")
    log.info("  experiment_summary.json    -- run stats")


if __name__ == "__main__":
    main()
