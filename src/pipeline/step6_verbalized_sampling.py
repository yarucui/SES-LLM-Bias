"""
Step 6 -- Direction 1 verbalized sampling diagnostic.

This script probes whether the entropy collapse seen in step5 is primarily
caused by forced-choice decoding or by the models' underlying beliefs.

Conditions:
    baseline -- reused from step5 raw_responses.jsonl (no new API calls)
    vs       -- verbalized sampling, 3 calls, T=0.7
    dv       -- distribution verbalization, 3 calls, T=0.7
    hight    -- high-temperature forced choice, 20 calls, T=1.2

Output files (under data/experiment/ or data/experiment/direction1_pilot/):
    direction1_selected_posts.json
    direction1_checkpoint.jsonl
    direction1_responses.jsonl
    direction1_distributions.csv
    direction1_parse_failures.jsonl
    direction1_summary.json

Usage:
    python src/pipeline/step6_verbalized_sampling.py
    python src/pipeline/step6_verbalized_sampling.py --dry-run
    python src/pipeline/step6_verbalized_sampling.py --pilot
    python src/pipeline/step6_verbalized_sampling.py --condition vs
    python src/pipeline/step6_verbalized_sampling.py --model gpt
    python src/pipeline/step6_verbalized_sampling.py --max-posts 20
    python src/pipeline/step6_verbalized_sampling.py --workers 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import signal
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from _llm import LLMError, chat_completion_with_usage
from src.pipeline.step5_experiment import (  # reuse step5 helpers directly
    binary_entropy,
    binary_kl,
    build_prompt as build_baseline_prompt,
    clean_post_text,
    parse_choice as parse_forced_choice,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "step6_verbalized_sampling.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("step6_verbalized_sampling")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_JSONL = config.POSTS_SCORED_DIR / "all_scored_valid.jsonl"
BASELINE_RAW_PATH = config.EXPERIMENT_DIR / "raw_responses.jsonl"

DEFAULT_TARGET_POSTS = 150
DEFAULT_WORKERS = 8
STRICT_RATIO_LO = 0.35
STRICT_RATIO_HI = 0.65
MIN_RATIO_LO = 0.20
MAX_RATIO_HI = 0.80
TIEBREAK_RNG = random.Random(42)
PILOT_RNG = random.Random(42)
DOMAIN_ORDER = ("education", "career", "finance")
PROGRESS_EVERY = 100
COST_HARD_CAP_USD = 25.0

CONDITION_BASELINE = "baseline"
CONDITION_VS = "vs"
CONDITION_DV = "dv"
CONDITION_HIGHT = "hight"

MODEL_PRICING = {
    "gpt": {"input_per_1m": 1.25, "output_per_1m": 10.0},
    "claude": {"input_per_1m": 3.0, "output_per_1m": 15.0},
    "mistral": {"input_per_1m": 2.0, "output_per_1m": 6.0},
    "llama": {"input_per_1m": 0.31, "output_per_1m": 0.91},
}


@dataclass(frozen=True)
class ConditionSpec:
    condition: str
    n_samples: int
    temperature: float
    max_tokens: int
    input_tokens: int
    output_tokens: int


CONDITION_SPECS = {
    CONDITION_VS: ConditionSpec(CONDITION_VS, 3, 0.7, 200, 750, 30),
    CONDITION_DV: ConditionSpec(CONDITION_DV, 3, 0.7, 200, 750, 30),
    CONDITION_HIGHT: ConditionSpec(CONDITION_HIGHT, 20, 1.2, 20, 700, 10),
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

VS_PROMPT_TEMPLATE = """{post_text}

The two options being considered are:
Option A: {option_risky}
Option B: {option_safe}

Output ONLY the two probability lines below, in this exact format. Do not
write any reasoning, explanation, or analysis. Your response must start
with the literal text "P(Option A) =".

P(Option A) = 0.XX
P(Option B) = 0.XX"""


DV_PROMPT_TEMPLATE = """{post_text}

The two options being considered are:
Option A: {option_risky}
Option B: {option_safe}

Among advice-givers on /r/{subreddit} who saw a post like this, what
fraction would recommend Option A vs Option B?

Output ONLY the two probability lines below, in this exact format. Do not
write any reasoning, explanation, or analysis. Your response must start
with the literal text "P(Option A) =".

P(Option A) = 0.XX
P(Option B) = 0.XX"""


def build_vs_prompt(post: dict) -> str:
    return VS_PROMPT_TEMPLATE.format(
        post_text=clean_post_text(post.get("title", ""), post.get("selftext", "")),
        option_risky=post.get("option_risky", ""),
        option_safe=post.get("option_safe", ""),
    )


def build_dv_prompt(post: dict) -> str:
    return DV_PROMPT_TEMPLATE.format(
        post_text=clean_post_text(post.get("title", ""), post.get("selftext", "")),
        option_risky=post.get("option_risky", ""),
        option_safe=post.get("option_safe", ""),
        subreddit=post.get("subreddit", ""),
    )


def build_hight_prompt(post: dict) -> str:
    # Reuse the exact baseline prompt from step5 so temperature is the only
    # manipulated variable in this control condition.
    return build_baseline_prompt(post)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


def compute_cost_usd(model_key: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING[model_key]
    return (
        (input_tokens / 1_000_000.0) * pricing["input_per_1m"]
        + (output_tokens / 1_000_000.0) * pricing["output_per_1m"]
    )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_output_dir(pilot: bool) -> Path:
    out_dir = config.EXPERIMENT_DIR / "direction1_pilot" if pilot else config.EXPERIMENT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "selected": output_dir / "direction1_selected_posts.json",
        "checkpoint": output_dir / "direction1_checkpoint.jsonl",
        "responses": output_dir / "direction1_responses.jsonl",
        "distributions": output_dir / "direction1_distributions.csv",
        "parse_failures": output_dir / "direction1_parse_failures.jsonl",
        "summary": output_dir / "direction1_summary.json",
    }


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict, lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_elapsed(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def human_risky_ratio(post: dict) -> float | None:
    val = post.get("human_risky_ratio")
    if val is None:
        val = post.get("risky_ratio")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def sort_posts_by_ambiguity(posts: list[dict]) -> list[dict]:
    decorated: list[tuple[float, float, dict]] = []
    for post in posts:
        rr = human_risky_ratio(post)
        if rr is None:
            continue
        decorated.append((abs(rr - 0.5), TIEBREAK_RNG.random(), post))
    decorated.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in decorated]


def proportional_domain_sample(posts: list[dict], target_n: int) -> list[dict]:
    by_domain: dict[str, list[dict]] = {domain: [] for domain in DOMAIN_ORDER}
    for post in posts:
        domain = post.get("domain")
        if domain in by_domain:
            by_domain[domain].append(post)

    total = sum(len(v) for v in by_domain.values())
    if total <= target_n:
        return sort_posts_by_ambiguity(posts)

    quotas: dict[str, int] = {domain: 0 for domain in DOMAIN_ORDER}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for domain in DOMAIN_ORDER:
        domain_total = len(by_domain[domain])
        if domain_total == 0:
            continue
        exact = target_n * domain_total / total
        base = min(domain_total, int(exact))
        quotas[domain] = base
        assigned += base
        remainders.append((exact - base, domain))

    remainders.sort(key=lambda item: (-item[0], item[1]))
    while assigned < target_n:
        progressed = False
        for _, domain in remainders:
            if quotas[domain] < len(by_domain[domain]):
                quotas[domain] += 1
                assigned += 1
                progressed = True
                if assigned >= target_n:
                    break
        if not progressed:
            break

    selected: list[dict] = []
    for domain in DOMAIN_ORDER:
        selected.extend(sort_posts_by_ambiguity(by_domain[domain])[:quotas[domain]])
    return sort_posts_by_ambiguity(selected)


# ---------------------------------------------------------------------------
# Post selection
# ---------------------------------------------------------------------------

def select_ambiguous_posts(
    all_posts: list[dict],
    target_n: int = DEFAULT_TARGET_POSTS,
    ratio_lo: float = STRICT_RATIO_LO,
    ratio_hi: float = STRICT_RATIO_HI,
) -> list[dict]:
    """Select ambiguous posts stratified by domain and sorted by ambiguity."""
    eligible_posts = [
        post for post in all_posts
        if post.get("domain") in DOMAIN_ORDER and human_risky_ratio(post) is not None
    ]

    current_lo = ratio_lo
    current_hi = ratio_hi
    final_sample: list[dict] = []

    while True:
        band_posts = [
            post for post in eligible_posts
            if current_lo <= human_risky_ratio(post) <= current_hi
        ]
        final_sample = proportional_domain_sample(band_posts, target_n)
        if len(final_sample) >= target_n:
            break
        if current_lo <= MIN_RATIO_LO and current_hi >= MAX_RATIO_HI:
            break
        current_lo = max(MIN_RATIO_LO, round(current_lo - 0.05, 2))
        current_hi = min(MAX_RATIO_HI, round(current_hi + 0.05, 2))
        if current_lo == MIN_RATIO_LO and current_hi == MAX_RATIO_HI:
            band_posts = [
                post for post in eligible_posts
                if current_lo <= human_risky_ratio(post) <= current_hi
            ]
            final_sample = proportional_domain_sample(band_posts, target_n)
            break

    counts = Counter(post.get("domain") for post in final_sample)
    log.info(
        "Ambiguous selection band: [%.2f, %.2f]",
        current_lo,
        current_hi,
    )
    log.info(
        "Selected per-domain counts: education=%d career=%d finance=%d",
        counts.get("education", 0),
        counts.get("career", 0),
        counts.get("finance", 0),
    )
    log.info("Total selected posts: %d", len(final_sample))

    select_ambiguous_posts.last_band = (current_lo, current_hi)
    return final_sample


def write_selected_posts(path: Path, posts: list[dict], band: tuple[float, float], pilot: bool) -> None:
    counts = Counter(post.get("domain") for post in posts)
    payload = {
        "created_at": utc_now_iso(),
        "pilot": pilot,
        "selection_band": [band[0], band[1]],
        "n_selected_posts": len(posts),
        "domain_breakdown": {domain: counts.get(domain, 0) for domain in DOMAIN_ORDER},
        "posts": [
            {
                "post_id": post.get("post_id"),
                "domain": post.get("domain"),
                "subreddit": post.get("subreddit"),
                "human_risky_ratio": human_risky_ratio(post),
            }
            for post in posts
        ],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Baseline loader
# ---------------------------------------------------------------------------

def load_baseline_for_selected(post_ids: set[str]) -> dict:
    """Return baseline aggregates from step5 raw_responses.jsonl."""
    if not BASELINE_RAW_PATH.exists():
        raise FileNotFoundError(f"Missing baseline file: {BASELINE_RAW_PATH}")

    grouped: dict[str, dict[str, dict[str, int | float | None]]] = defaultdict(dict)
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"n_calls": 0, "n_a": 0, "n_b": 0, "n_valid": 0})

    with BASELINE_RAW_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("post_id")
            mk = rec.get("model_key")
            if pid not in post_ids or not mk:
                continue
            key = (pid, mk)
            counts[key]["n_calls"] += 1
            parsed = parse_forced_choice(rec.get("raw_response")) if rec.get("error") is None else None
            if parsed == "A":
                counts[key]["n_a"] += 1
                counts[key]["n_valid"] += 1
            elif parsed == "B":
                counts[key]["n_b"] += 1
                counts[key]["n_valid"] += 1

    for (pid, mk), cell in counts.items():
        rr = (cell["n_a"] / cell["n_valid"]) if cell["n_valid"] else None
        grouped[pid][mk] = {
            "n_calls": cell["n_calls"],
            "n_a": cell["n_a"],
            "n_b": cell["n_b"],
            "n_valid": cell["n_valid"],
            "llm_risky_rate": rr,
            "llm_entropy": binary_entropy(rr),
        }
    return grouped


# ---------------------------------------------------------------------------
# Distribution parsers
# ---------------------------------------------------------------------------

PROB_LINE_PATTERN = re.compile(
    r"P\s*\(\s*(?:Option\s*)?(?:community\s+recommends\s+)?([AaBb])\b[^)]*\)\s*=\s*(\d+\.?\d*)",
    re.IGNORECASE,
)


def parse_distribution(response: str) -> tuple[float, float] | None:
    """Strict parser. Only accepts numbers that appear right after a
    P(...) pattern. Returns (p_a, p_b) or None.

    Handles:
      'P(Option A) = 0.6\nP(Option B) = 0.4'
      'P(A) = 0.6, P(B) = 0.4'
      'P(Option A) = 60\nP(Option B) = 40'  (percentages)
      'P(community recommends Option A) = 0.55'

    Rejects:
      Numbers embedded in reasoning text (e.g. "$32,000 in loans")
      Outputs without explicit P(A) / P(B) lines
    """
    if not response:
        return None

    matches = PROB_LINE_PATTERN.findall(response)
    if len(matches) < 2:
        return None

    p_a = p_b = None
    for letter, value in matches:
        v = float(value)
        if letter.upper() == "A" and p_a is None:
            p_a = v
        elif letter.upper() == "B" and p_b is None:
            p_b = v
        if p_a is not None and p_b is not None:
            break

    if p_a is None or p_b is None:
        return None

    # Normalize percentages to fractions.
    if p_a > 1 or p_b > 1:
        if 0 <= p_a <= 100 and 0 <= p_b <= 100:
            p_a /= 100
            p_b /= 100
        else:
            return None

    # Sanity: must be valid probabilities.
    if not (0 <= p_a <= 1 and 0 <= p_b <= 1):
        return None

    # Renormalize ONLY if close to 1.0 (genuine rounding error).
    total = p_a + p_b
    if 0.90 <= total <= 1.10:
        if total > 0:
            p_a /= total
            p_b /= total
        return (p_a, p_b)
    else:
        # Sum is far from 1 - these are probably not valid probabilities.
        return None


def parse_vs_distribution(response: str) -> tuple[float, float] | None:
    return parse_distribution(response)


def parse_dv_distribution(response: str) -> tuple[float, float] | None:
    return parse_distribution(response)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> dict[tuple[str, int], dict]:
    done: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[(rec["variant_id"], int(rec["sample_index"]))] = rec
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done


def _checkpoint_counts_by_model_condition(records: dict[tuple[str, int], dict]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for rec in records.values():
        counts[(rec.get("model_key", ""), rec.get("condition", ""))] += 1
    return dict(sorted(counts.items()))


def purge_llama_failed_distributions(
    checkpoint_path: Path,
    purged_path: Path,
    force_purge: bool = False,
) -> int:
    """Move failed Llama VS/DV records out of the main checkpoint."""
    if not checkpoint_path.exists():
        log.info("No checkpoint found at %s; nothing to purge.", checkpoint_path)
        return 0

    checkpoint = load_checkpoint(checkpoint_path)
    purged: list[dict] = []
    kept: dict[tuple[str, int], dict] = {}

    for key, rec in checkpoint.items():
        is_llama = rec.get("model_key") == "llama"
        is_target_condition = rec.get("condition") in (CONDITION_VS, CONDITION_DV)
        if not (is_llama and is_target_condition):
            kept[key] = rec
            continue

        strict_dist = parse_distribution(rec.get("raw_response") or "")
        parse_failure_flag = bool(rec.get("parse_failure") or rec.get("parse_failed"))
        should_purge = (
            parse_failure_flag
            or rec.get("parsed_value") is None
            or strict_dist is None
        )
        if should_purge:
            purged.append(rec)
        else:
            kept[key] = rec

    log.info("Purging %d Llama records from %s", len(purged), checkpoint_path.name)
    if purged and not force_purge:
        prompt = (
            f"About to move {len(purged)} records to {purged_path.name} and remove them from "
            f"{checkpoint_path.name}. Continue? [y/N]: "
        )
        try:
            response = input(prompt).strip().lower()
        except EOFError:
            response = ""
        if response not in {"y", "yes"}:
            log.info("Purge aborted by user.")
            raise SystemExit(1)

    if purged:
        with purged_path.open("a", encoding="utf-8") as f:
            for rec in purged:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        dedupe_and_write_checkpoint(kept, checkpoint_path)
    else:
        log.info("No Llama VS/DV records required purging.")

    refreshed = load_checkpoint(checkpoint_path)
    remaining_counts = _checkpoint_counts_by_model_condition(refreshed)
    log.info("Remaining checkpoint records by (model, condition):")
    for (model_key, condition), count in remaining_counts.items():
        log.info("  %s / %s: %d", model_key, condition, count)

    return len(purged)


def dedupe_and_write_checkpoint(records: dict[tuple[str, int], dict], path: Path) -> None:
    ordered = sorted(
        records.values(),
        key=lambda rec: (rec["post_id"], rec["model_key"], rec["condition"], rec["sample_index"]),
    )
    with path.open("w", encoding="utf-8") as f:
        for rec in ordered:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Task processing
# ---------------------------------------------------------------------------

def parse_condition_response(condition: str, response: str | None) -> tuple[str | None, list[float] | None]:
    if response is None:
        return None, None
    if condition == CONDITION_HIGHT:
        choice = parse_forced_choice(response)
        return choice, None
    if condition == CONDITION_VS:
        dist = parse_vs_distribution(response)
        return None, list(dist) if dist is not None else None
    if condition == CONDITION_DV:
        dist = parse_dv_distribution(response)
        return None, list(dist) if dist is not None else None
    raise ValueError(f"Unknown condition: {condition}")


def build_prompt_for_condition(post: dict, condition: str) -> str:
    if condition == CONDITION_VS:
        return build_vs_prompt(post)
    if condition == CONDITION_DV:
        return build_dv_prompt(post)
    if condition == CONDITION_HIGHT:
        return build_hight_prompt(post)
    raise ValueError(f"Unknown condition: {condition}")


def call_model_with_usage(
    model_key: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str | None, int, str | None, int, int, float]:
    cfg = config.STUDY_MODELS[model_key]
    extra_headers: dict[str, str] = {}
    if cfg["provider"] == "openrouter":
        extra_headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER", "") or ""
        extra_headers["X-Title"] = os.getenv("OPENROUTER_X_TITLE", "") or ""

    t0 = time.time()
    try:
        result = chat_completion_with_usage(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            model=cfg["model_name"],
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=60.0,
            extra_headers=extra_headers,
            extra_body=cfg.get("extra_body") or {},
        )
        latency_ms = int((time.time() - t0) * 1000)
        usage = result.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        cost_usd = compute_cost_usd(model_key, input_tokens, output_tokens)
        return result.get("content"), latency_ms, None, input_tokens, output_tokens, cost_usd
    except LLMError as exc:
        return None, int((time.time() - t0) * 1000), str(exc), 0, 0, 0.0


def process_task(
    post: dict,
    model_key: str,
    condition: str,
    sample_index: int,
    prompt: str,
    p_hash: str,
    spec: ConditionSpec,
    dry_run: bool,
) -> dict:
    variant_id = f"{post['post_id']}__{model_key}__{condition}"
    if dry_run:
        return {
            "variant_id": variant_id,
            "post_id": post["post_id"],
            "model_key": model_key,
            "condition": condition,
            "sample_index": sample_index,
            "prompt_hash": p_hash,
            "raw_response": "[DRY RUN]",
            "parsed_value": None,
            "parsed_choice": None,
            "latency_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "error": None,
            "timestamp": utc_now_iso(),
        }

    raw_response, latency_ms, error, input_tokens, output_tokens, cost_usd = call_model_with_usage(
        model_key=model_key,
        prompt=prompt,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
    )
    parsed_choice, parsed_value = parse_condition_response(condition, raw_response if error is None else None)

    return {
        "variant_id": variant_id,
        "post_id": post["post_id"],
        "model_key": model_key,
        "condition": condition,
        "sample_index": sample_index,
        "prompt_hash": p_hash,
        "raw_response": raw_response,
        "parsed_value": parsed_value,
        "parsed_choice": parsed_choice,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "error": error,
        "timestamp": utc_now_iso(),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_condition_records(records: list[dict]) -> dict[tuple[str, str, str], dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for rec in records:
        grouped[(rec["post_id"], rec["model_key"], rec["condition"])].append(rec)

    out: dict[tuple[str, str, str], dict] = {}
    for key, group in grouped.items():
        condition = key[2]
        n_calls = len(group)
        valid = [rec for rec in group if rec.get("error") is None]

        if condition in (CONDITION_BASELINE, CONDITION_HIGHT):
            parsed = [rec for rec in valid if rec.get("parsed_choice") in ("A", "B")]
            n_valid = len(parsed)
            n_a = sum(1 for rec in parsed if rec["parsed_choice"] == "A")
            n_b = sum(1 for rec in parsed if rec["parsed_choice"] == "B")
            llm_rr = (n_a / n_valid) if n_valid else None
            out[key] = {
                "n_calls": n_calls,
                "n_valid": n_valid,
                "n_a": n_a,
                "n_b": n_b,
                "llm_risky_rate": llm_rr,
                "llm_entropy": binary_entropy(llm_rr),
            }
            continue

        parsed_dist = [rec["parsed_value"] for rec in valid if isinstance(rec.get("parsed_value"), list) and len(rec["parsed_value"]) == 2]
        n_valid = len(parsed_dist)
        llm_rr = (sum(dist[0] for dist in parsed_dist) / n_valid) if n_valid else None
        out[key] = {
            "n_calls": n_calls,
            "n_valid": n_valid,
            "n_a": None,
            "n_b": None,
            "llm_risky_rate": llm_rr,
            "llm_entropy": binary_entropy(llm_rr),
        }
    return out


CSV_COLUMNS = [
    "post_id",
    "domain",
    "consensus_level",
    "ses_level",
    "subreddit",
    "human_risky_ratio",
    "human_entropy",
    "model",
    "condition",
    "n_calls",
    "n_valid",
    "llm_risky_rate",
    "llm_entropy",
    "gap",
    "abs_gap",
    "entropy_gap",
    "kl_divergence",
]


def build_distribution_rows(
    posts: list[dict],
    model_keys: list[str],
    baseline: dict,
    aggregates: dict[tuple[str, str, str], dict],
) -> list[dict]:
    rows: list[dict] = []
    for post in posts:
        pid = post["post_id"]
        human_rr = human_risky_ratio(post)
        human_ent = binary_entropy(human_rr)
        for model_key in model_keys:
            for condition in (CONDITION_BASELINE, CONDITION_VS, CONDITION_DV, CONDITION_HIGHT):
                if condition == CONDITION_BASELINE:
                    cell = baseline.get(pid, {}).get(model_key, {})
                else:
                    cell = aggregates.get((pid, model_key, condition), {})
                llm_rr = cell.get("llm_risky_rate")
                llm_ent = cell.get("llm_entropy")
                gap = (llm_rr - human_rr) if (llm_rr is not None and human_rr is not None) else None
                rows.append({
                    "post_id": pid,
                    "domain": post.get("domain"),
                    "consensus_level": post.get("consensus_level"),
                    "ses_level": post.get("ses_level"),
                    "subreddit": post.get("subreddit"),
                    "human_risky_ratio": human_rr,
                    "human_entropy": human_ent,
                    "model": model_key,
                    "condition": condition,
                    "n_calls": cell.get("n_calls", 0),
                    "n_valid": cell.get("n_valid", 0),
                    "llm_risky_rate": llm_rr,
                    "llm_entropy": llm_ent,
                    "gap": gap,
                    "abs_gap": abs(gap) if gap is not None else None,
                    "entropy_gap": (human_ent - llm_ent) if (human_ent is not None and llm_ent is not None) else None,
                    "kl_divergence": binary_kl(human_rr, llm_rr),
                })
    return rows


def write_distributions_csv(rows: list[dict], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (row["domain"] or "", row["post_id"], row["model"], row["condition"]))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in ordered:
            formatted = {}
            for key in CSV_COLUMNS:
                value = row.get(key)
                if isinstance(value, float):
                    formatted[key] = f"{value:.6f}"
                elif value is None:
                    formatted[key] = ""
                else:
                    formatted[key] = value
            writer.writerow(formatted)


# ---------------------------------------------------------------------------
# Cost and runtime projection
# ---------------------------------------------------------------------------

def estimate_runtime_minutes(total_calls: int, workers: int, baseline_rows: list[dict]) -> float:
    latencies = [int(rec["latency_ms"]) for rec in baseline_rows if rec.get("latency_ms") is not None]
    avg_latency_ms = (sum(latencies) / len(latencies)) if latencies else 3500.0
    return (total_calls * avg_latency_ms) / max(workers, 1) / 1000.0 / 60.0


def project_cost_and_runtime(posts: list[dict], model_keys: list[str], workers: int, baseline_rows: list[dict]) -> dict:
    n_posts = len(posts)
    per_condition_calls = {
        condition: n_posts * len(model_keys) * spec.n_samples
        for condition, spec in CONDITION_SPECS.items()
    }
    total_calls = sum(per_condition_calls.values())
    est_cost = 0.0
    for model_key in model_keys:
        for spec in CONDITION_SPECS.values():
            est_cost += (
                n_posts
                * spec.n_samples
                * compute_cost_usd(model_key, spec.input_tokens, spec.output_tokens)
            )
    return {
        "n_posts": n_posts,
        "n_models": len(model_keys),
        "per_condition_calls": per_condition_calls,
        "total_calls": total_calls,
        "est_cost_usd": est_cost,
        "eta_minutes": estimate_runtime_minutes(total_calls, workers, baseline_rows),
    }


def print_cost_projection(posts: list[dict], model_keys: list[str], workers: int, baseline_rows: list[dict], cost_cap: float) -> dict:
    projection = project_cost_and_runtime(posts, model_keys, workers, baseline_rows)
    print("")
    print("DIRECTION 1  VERBALIZED SAMPLING DIAGNOSTIC")
    print(f"  Selected posts:    {projection['n_posts']}")
    print(f"  Models:            {projection['n_models']}    ({', '.join(model_keys)})")
    print("  Conditions:        VS (N=3) + DV (N=3) + High-T (N=20)")
    print("                     [Baseline reused from step5]")
    print(
        f"  Total NEW calls:   {projection['total_calls']} = "
        f"{projection['n_posts']} * {projection['n_models']} * (3 + 3 + 20)"
    )
    print("")
    print(f"ESTIMATED COST: ~${projection['est_cost_usd']:.2f}")
    print(f"ESTIMATED RUNTIME with {workers} workers: ~{projection['eta_minutes']:.0f} minutes")
    print(f"HARD COST CAP: ${cost_cap:.2f}")
    print("")
    print("Press Ctrl+C within 10 seconds to abort.")
    print("")
    return projection


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(
    rows: list[dict],
    checkpoint_records: list[dict],
    posts: list[dict],
    selection_band: tuple[float, float],
    elapsed_seconds: float,
) -> dict:
    counts = Counter(post.get("domain") for post in posts)
    parse_failures_by_condition = Counter()
    by_model = defaultdict(lambda: {
        "vs_failures": 0,
        "dv_failures": 0,
        "hight_failures": 0,
        "api_errors": 0,
        "mean_latency_ms": 0.0,
        "n_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    })
    cost_by_condition = defaultdict(lambda: {
        "n_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    })

    latency_by_model: dict[str, list[int]] = defaultdict(list)
    successful_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0

    for rec in checkpoint_records:
        mk = rec["model_key"]
        condition = rec["condition"]
        input_tokens = int(rec.get("input_tokens") or 0)
        output_tokens = int(rec.get("output_tokens") or 0)
        cost_usd = float(rec.get("cost_usd") or 0.0)
        by_model[mk]["n_calls"] += 1
        by_model[mk]["input_tokens"] += input_tokens
        by_model[mk]["output_tokens"] += output_tokens
        by_model[mk]["cost_usd"] += cost_usd
        cost_by_condition[condition]["n_calls"] += 1
        cost_by_condition[condition]["input_tokens"] += input_tokens
        cost_by_condition[condition]["output_tokens"] += output_tokens
        cost_by_condition[condition]["cost_usd"] += cost_usd
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_cost_usd += cost_usd
        if rec.get("latency_ms") is not None:
            latency_by_model[mk].append(int(rec["latency_ms"]))
        if rec.get("error"):
            by_model[mk]["api_errors"] += 1
            continue
        if rec["condition"] in (CONDITION_VS, CONDITION_DV):
            if rec.get("parsed_value") is None:
                parse_failures_by_condition[rec["condition"]] += 1
                by_model[mk][f"{rec['condition']}_failures"] += 1
            else:
                successful_calls += 1
        else:
            if rec.get("parsed_choice") not in ("A", "B"):
                parse_failures_by_condition[rec["condition"]] += 1
                by_model[mk][f"{rec['condition']}_failures"] += 1
            else:
                successful_calls += 1

    for mk in by_model:
        latencies = latency_by_model.get(mk, [])
        by_model[mk]["mean_latency_ms"] = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
        by_model[mk]["cost_usd"] = round(by_model[mk]["cost_usd"], 6)

    for condition in CONDITION_SPECS:
        cost_by_condition[condition]
    for condition in cost_by_condition:
        cost_by_condition[condition]["cost_usd"] = round(cost_by_condition[condition]["cost_usd"], 6)

    n_calls_per_condition = {
        condition: sum(1 for rec in checkpoint_records if rec["condition"] == condition)
        for condition in CONDITION_SPECS
    }

    return {
        "n_selected_posts": len(posts),
        "selection_band": [selection_band[0], selection_band[1]],
        "domain_breakdown": {domain: counts.get(domain, 0) for domain in DOMAIN_ORDER},
        "n_calls_per_condition": n_calls_per_condition,
        "n_calls_total": len(checkpoint_records),
        "successful_calls": successful_calls,
        "parse_failures_by_condition": {
            condition: parse_failures_by_condition.get(condition, 0)
            for condition in CONDITION_SPECS
        },
        "by_model": dict(sorted(by_model.items())),
        "cost_breakdown": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost_usd": round(total_cost_usd, 6),
            "by_condition": dict(sorted(cost_by_condition.items())),
        },
        "elapsed_seconds": round(elapsed_seconds, 1),
    }


def print_final_summary(rows: list[dict], elapsed_seconds: float, output_dir: Path) -> None:
    print("=" * 64)
    print(f"DIRECTION 1 COMPLETE in {format_elapsed(elapsed_seconds)}")
    print("")
    print("Per-condition mean entropy (averaged across all posts and models):")
    for condition, label in (
        (CONDITION_BASELINE, "baseline (T=0.7, forced)"),
        (CONDITION_VS, "vs       (T=0.7, dist)"),
        (CONDITION_DV, "dv       (T=0.7, descr)"),
        (CONDITION_HIGHT, "hight    (T=1.2, forced)"),
    ):
        vals = [row["llm_entropy"] for row in rows if row["condition"] == condition and row["llm_entropy"] is not None]
        mean_val = (sum(vals) / len(vals)) if vals else None
        print(f"  {label:<28} {mean_val:.3f}" if mean_val is not None else f"  {label:<28} n/a")
    human_vals = [row["human_entropy"] for row in rows if row["human_entropy"] is not None]
    human_mean = (sum(human_vals) / len(human_vals)) if human_vals else None
    print(f"  {'human reference:':<28} {human_mean:.3f}" if human_mean is not None else "  human reference:             n/a")
    print("")
    print("Per-condition mean abs_gap from human:")
    for condition in (CONDITION_BASELINE, CONDITION_VS, CONDITION_DV, CONDITION_HIGHT):
        vals = [row["abs_gap"] for row in rows if row["condition"] == condition and row["abs_gap"] is not None]
        mean_val = (sum(vals) / len(vals)) if vals else None
        print(f"  {condition:<8} {mean_val:.3f}" if mean_val is not None else f"  {condition:<8} n/a")
    print("")
    print("Entropy recovery (vs - baseline) / (human - baseline) per model:")
    model_keys = sorted({row["model"] for row in rows})
    for model_key in model_keys:
        baseline_vals = [row["llm_entropy"] for row in rows if row["model"] == model_key and row["condition"] == CONDITION_BASELINE and row["llm_entropy"] is not None]
        vs_vals = [row["llm_entropy"] for row in rows if row["model"] == model_key and row["condition"] == CONDITION_VS and row["llm_entropy"] is not None]
        human_model_vals = [row["human_entropy"] for row in rows if row["model"] == model_key and row["human_entropy"] is not None]
        if not baseline_vals or not vs_vals or not human_model_vals:
            print(f"  {model_key:<8} n/a")
            continue
        baseline_mean = sum(baseline_vals) / len(baseline_vals)
        vs_mean = sum(vs_vals) / len(vs_vals)
        human_mean_model = sum(human_model_vals) / len(human_model_vals)
        denom = human_mean_model - baseline_mean
        if abs(denom) < 1e-9:
            print(f"  {model_key:<8} n/a")
            continue
        recovery = 100.0 * ((vs_mean - baseline_mean) / denom)
        print(f"  {model_key:<8} {recovery:.0f}%")
    print("")
    print("Output files:")
    print(f"  {output_dir / 'direction1_responses.jsonl'}")
    print(f"  {output_dir / 'direction1_distributions.csv'}")
    print(f"  {output_dir / 'direction1_summary.json'}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6: Direction 1 verbalized sampling diagnostic.")
    parser.add_argument("--dry-run", action="store_true", help="Print selection, prompts, and cost projection without API calls.")
    parser.add_argument("--pilot", action="store_true", help="Run 5 random ambiguous posts, gpt only, writing to direction1_pilot/.")
    parser.add_argument("--condition", choices=list(CONDITION_SPECS.keys()), default=None, help="Run only one new condition.")
    parser.add_argument("--model", choices=list(config.STUDY_MODELS.keys()), default=None, help="Run only one study model.")
    parser.add_argument("--max-posts", type=int, default=None, help="Cap the selected posts after ambiguity filtering.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Thread-pool size for parallel API calls.")
    parser.add_argument("--cost-cap", type=float, default=COST_HARD_CAP_USD, help=f"Hard stop once actual checkpointed cost exceeds this USD cap (default: {COST_HARD_CAP_USD}).")
    parser.add_argument("--purge-llama-bad", action="store_true", help="Move failed Llama VS/DV checkpoint records to direction1_checkpoint_purged.jsonl before resuming.")
    parser.add_argument("--force-purge", action="store_true", help="Skip the confirmation prompt for --purge-llama-bad.")
    args = parser.parse_args()

    if not INPUT_JSONL.exists():
        log.error("Missing %s -- run step2_score.py and step2b_validate.py first", INPUT_JSONL)
        sys.exit(1)

    output_dir = get_output_dir(args.pilot)
    paths = get_output_paths(output_dir)
    checkpoint_lock = threading.Lock()
    parsefail_lock = threading.Lock()

    if args.purge_llama_bad:
        purge_llama_failed_distributions(
            paths["checkpoint"],
            output_dir / "direction1_checkpoint_purged.jsonl",
            force_purge=args.force_purge,
        )

    all_posts = load_jsonl(INPUT_JSONL)
    selected_posts = select_ambiguous_posts(all_posts)
    selection_band = getattr(select_ambiguous_posts, "last_band", (STRICT_RATIO_LO, STRICT_RATIO_HI))

    if args.pilot:
        sample_size = min(5, len(selected_posts))
        selected_posts = sorted(PILOT_RNG.sample(selected_posts, sample_size), key=lambda post: post["post_id"])

    if args.max_posts is not None:
        selected_posts = selected_posts[:args.max_posts]

    write_selected_posts(paths["selected"], selected_posts, selection_band, args.pilot)

    model_keys = ["gpt"] if args.pilot else ([args.model] if args.model else list(config.STUDY_MODELS.keys()))
    run_conditions = [args.condition] if args.condition else list(CONDITION_SPECS.keys())

    post_ids = {post["post_id"] for post in selected_posts}
    baseline_rows_raw = load_jsonl(BASELINE_RAW_PATH) if BASELINE_RAW_PATH.exists() else []
    baseline = load_baseline_for_selected(post_ids)

    projection = print_cost_projection(selected_posts, model_keys, args.workers, baseline_rows_raw, args.cost_cap)

    if args.dry_run:
        if selected_posts:
            example = selected_posts[0]
            log.info("Dry run example post_id: %s", example["post_id"])
            log.info("----- VS PROMPT START -----")
            for line in build_vs_prompt(example).splitlines():
                log.info("  %s", line)
            log.info("----- VS PROMPT END -----")
            log.info("----- DV PROMPT START -----")
            for line in build_dv_prompt(example).splitlines():
                log.info("  %s", line)
            log.info("----- DV PROMPT END -----")
            log.info("----- HIGH-T PROMPT START -----")
            for line in build_hight_prompt(example).splitlines():
                log.info("  %s", line)
            log.info("----- HIGH-T PROMPT END -----")
        return

    missing_keys: list[str] = []
    for mk in model_keys:
        cfg = config.STUDY_MODELS[mk]
        if not cfg.get("api_key"):
            missing_keys.append(f"{mk} ({cfg['provider']})")
    if missing_keys:
        log.error("Missing API key(s) for: %s", ", ".join(missing_keys))
        sys.exit(1)

    checkpoint = load_checkpoint(paths["checkpoint"])
    remaining_counts = _checkpoint_counts_by_model_condition(checkpoint)
    if remaining_counts:
        log.info("Checkpoint records remaining by (model, condition):")
        for (model_key, condition), count in remaining_counts.items():
            log.info("  %s / %s: %d", model_key, condition, count)
    running_cost_usd = sum(float(rec.get("cost_usd") or 0.0) for rec in checkpoint.values())
    if running_cost_usd > args.cost_cap:
        log.error(
            "Existing checkpoint cost already exceeds the cap: $%.4f > $%.2f",
            running_cost_usd,
            args.cost_cap,
        )
        raise SystemExit(1)
    tasks: list[tuple[dict, str, str, int, str, str, ConditionSpec]] = []
    for post in selected_posts:
        for model_key in model_keys:
            for condition in run_conditions:
                spec = CONDITION_SPECS[condition]
                prompt = build_prompt_for_condition(post, condition)
                p_hash = prompt_hash(prompt)
                variant_id = f"{post['post_id']}__{model_key}__{condition}"
                for sample_index in range(spec.n_samples):
                    key = (variant_id, sample_index)
                    rec = checkpoint.get(key)
                    if rec is not None and rec.get("error") is None:
                        if condition in (CONDITION_VS, CONDITION_DV) and rec.get("parsed_value") is not None:
                            continue
                        if condition == CONDITION_HIGHT and rec.get("parsed_choice") in ("A", "B"):
                            continue
                    tasks.append((post, model_key, condition, sample_index, prompt, p_hash, spec))

    total_pending = len(tasks)
    log.info("Pending new API calls: %d", total_pending)

    if total_pending > 0:
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            log.info("Aborted before start.")
            return
    else:
        log.info("Nothing to do. Rebuilding outputs from checkpoint and baseline.")

    interrupted = threading.Event()

    def _sigint(_sig: int, _frame: object) -> None:
        if not interrupted.is_set():
            log.warning("Interrupt received -- finishing in-flight calls and stopping.")
            interrupted.set()

    signal.signal(signal.SIGINT, _sigint)

    start = time.time()
    completed = 0
    model_progress = Counter()
    per_model_pending = Counter(task[1] for task in tasks)
    cost_cap_exceeded = False
    pending_tasks: deque[tuple[dict, str, str, int, str, str, ConditionSpec]] = deque(tasks)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        active_futures: dict = {}

        def submit_more() -> None:
            while (
                pending_tasks
                and len(active_futures) < max(1, args.workers)
                and not interrupted.is_set()
                and not cost_cap_exceeded
            ):
                post, mk, condition, idx, prompt, p_hash, spec = pending_tasks.popleft()
                future = executor.submit(process_task, post, mk, condition, idx, prompt, p_hash, spec, False)
                active_futures[future] = (post, mk, condition, idx)

        submit_more()

        while active_futures:
            done_futures, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
            for future in done_futures:
                post, mk, condition, idx = active_futures.pop(future)
                try:
                    rec = future.result()
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Unexpected exception for %s/%s/%s/%d: %s",
                        post["post_id"],
                        mk,
                        condition,
                        idx,
                        exc,
                    )
                    continue

                key = (rec["variant_id"], rec["sample_index"])
                old_cost = float(checkpoint.get(key, {}).get("cost_usd") or 0.0)
                checkpoint[key] = rec
                running_cost_usd += float(rec.get("cost_usd") or 0.0) - old_cost
                append_jsonl(paths["checkpoint"], rec, checkpoint_lock)

                if rec.get("error") is None:
                    parse_failed = (
                        rec["condition"] in (CONDITION_VS, CONDITION_DV) and rec.get("parsed_value") is None
                    ) or (
                        rec["condition"] == CONDITION_HIGHT and rec.get("parsed_choice") not in ("A", "B")
                    )
                    if parse_failed:
                        append_jsonl(paths["parse_failures"], {
                            "variant_id": rec["variant_id"],
                            "condition": rec["condition"],
                            "raw_response": rec.get("raw_response"),
                            "model_key": rec["model_key"],
                        }, parsefail_lock)
                        log.warning(
                            "Parse failure [%s/%s/%s/%d]: %r",
                            rec["post_id"],
                            rec["model_key"],
                            rec["condition"],
                            rec["sample_index"],
                            (rec.get("raw_response") or "")[:160],
                        )
                else:
                    log.warning(
                        "API error [%s/%s/%s/%d]: %s",
                        rec["post_id"],
                        rec["model_key"],
                        rec["condition"],
                        rec["sample_index"],
                        rec["error"],
                    )

                completed += 1
                model_progress[rec["model_key"]] += 1
                if completed % PROGRESS_EVERY == 0 or completed == total_pending:
                    pct = 100.0 * completed / max(total_pending, 1)
                    per_model = " | ".join(
                        f"{mk}: {model_progress[mk]}/{per_model_pending[mk]}"
                        for mk in model_keys
                    )
                    log.info(
                        "Direction 1 progress: %d/%d (%.1f%%) | %s | running cost $%.4f | elapsed %s",
                        completed,
                        total_pending,
                        pct,
                        per_model,
                        running_cost_usd,
                        format_elapsed(time.time() - start),
                    )

                if running_cost_usd > args.cost_cap and not cost_cap_exceeded:
                    cost_cap_exceeded = True
                    interrupted.set()
                    log.error(
                        "Cost cap exceeded: checkpointed cost is now $%.4f > $%.2f. "
                        "Stopping further submissions; resume with a higher --cost-cap if intended.",
                        running_cost_usd,
                        args.cost_cap,
                    )

            if cost_cap_exceeded:
                for future in list(active_futures):
                    if future.cancel():
                        active_futures.pop(future, None)
            submit_more()

    elapsed = time.time() - start

    dedupe_and_write_checkpoint(checkpoint, paths["responses"])
    records_for_scope = [
        rec for rec in checkpoint.values()
        if rec["post_id"] in post_ids and rec["model_key"] in model_keys and rec["condition"] in CONDITION_SPECS
    ]
    aggregates = aggregate_condition_records(records_for_scope)
    rows = build_distribution_rows(selected_posts, model_keys, baseline, aggregates)
    write_distributions_csv(rows, paths["distributions"])

    summary = build_summary(rows, records_for_scope, selected_posts, selection_band, elapsed)
    with paths["summary"].open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_final_summary(rows, elapsed, output_dir)
    if cost_cap_exceeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
