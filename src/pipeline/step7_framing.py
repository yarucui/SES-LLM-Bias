"""
Step 7 -- Direction 2: Oracle vs Advisor framing experiment.

Tests whether the ROLE assigned to the LLM (advisor vs oracle vs personal)
affects the forced-choice distribution. The advisor framing is reused from
step5; oracle and personal framings are new.

If oracle framing produces different distributions than advisor framing on
the same posts, the collapse is partly role-induced. Combined with
Direction 1, this gives a 2x2 mechanism map (prescriptive/descriptive role
x single-token/distribution output).

Sample selection: ALL 811 posts in all_scored_valid.jsonl (no ambiguity
filtering -- the hypothesis is about systematic framing effects across the
entire distribution).

Output files (under data/experiment/ or data/experiment/direction2_pilot/):
    direction2_checkpoint.jsonl
    direction2_responses.jsonl
    direction2_parse_failures.jsonl
    direction2_choice_rates.jsonl
    direction2_comparison_table.csv
    direction2_summary.json

Usage:
    python src/pipeline/step7_framing.py
    python src/pipeline/step7_framing.py --dry-run
    python src/pipeline/step7_framing.py --pilot
    python src/pipeline/step7_framing.py --framing oracle
    python src/pipeline/step7_framing.py --model gpt
    python src/pipeline/step7_framing.py --max-posts 50
    python src/pipeline/step7_framing.py --workers 8
    python src/pipeline/step7_framing.py --cost-cap 40
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import signal
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from _llm import LLMError, chat_completion_with_usage
from src.pipeline.step5_experiment import (
    binary_entropy,
    binary_kl,
    clean_post_text,
    parse_choice,
    prompt_hash,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

config.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "step7_framing.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("step7_framing")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_JSONL = config.POSTS_SCORED_DIR / "all_scored_valid.jsonl"
BASELINE_RAW_PATH = config.EXPERIMENT_DIR / "raw_responses.jsonl"

FRAMING_ADVISOR = "advisor"
FRAMING_ORACLE = "oracle"
FRAMING_PERSONAL = "personal"
ALL_FRAMINGS = [FRAMING_ADVISOR, FRAMING_ORACLE, FRAMING_PERSONAL]
NEW_FRAMINGS = [FRAMING_ORACLE, FRAMING_PERSONAL]

DEFAULT_N_SAMPLES = 10
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 30
DEFAULT_WORKERS = 8
DEFAULT_COST_CAP = 40.0
PILOT_N_POSTS = 5
PILOT_N_SAMPLES = 5
PILOT_RNG = random.Random(42)
PROGRESS_EVERY = 200

MODEL_PRICING = {
    "gpt": {"input_per_1m": 1.25, "output_per_1m": 10.0},
    "claude": {"input_per_1m": 3.0, "output_per_1m": 15.0},
    "mistral": {"input_per_1m": 2.0, "output_per_1m": 6.0},
    "llama": {"input_per_1m": 0.31, "output_per_1m": 0.91},
}

STOP_SEQUENCES = ["\n\n"]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ORACLE_PROMPT_TEMPLATE = """{post_text}

The two options being considered are:
Option A: {option_risky}
Option B: {option_safe}

Among advice-givers on /r/{subreddit} who saw a post like this, which option would receive more upvoted recommendations?

Respond with exactly two words: "Option A" or "Option B". Do not explain. Do not restate the options. Do not add any other text."""

PERSONAL_PROMPT_TEMPLATE = """{post_text}

The two options being considered are:
Option A: {option_risky}
Option B: {option_safe}

If you personally were facing this decision, which option would you choose?

Respond with exactly two words: "Option A" or "Option B". Do not explain. Do not restate the options. Do not add any other text."""


def build_oracle_prompt(post: dict) -> str:
    return ORACLE_PROMPT_TEMPLATE.format(
        post_text=clean_post_text(post.get("title", ""), post.get("selftext", "")),
        option_risky=post.get("option_risky", ""),
        option_safe=post.get("option_safe", ""),
        subreddit=post.get("subreddit", ""),
    )


def build_personal_prompt(post: dict) -> str:
    return PERSONAL_PROMPT_TEMPLATE.format(
        post_text=clean_post_text(post.get("title", ""), post.get("selftext", "")),
        option_risky=post.get("option_risky", ""),
        option_safe=post.get("option_safe", ""),
    )


def build_advisor_prompt(post: dict) -> str:
    body = clean_post_text(post.get("title", ""), post.get("selftext", ""))
    return (
        f"{body}\n\n"
        f"The two options being considered are:\n"
        f"Option A: {post.get('option_risky', '')}\n"
        f"Option B: {post.get('option_safe', '')}\n\n"
        f"{config.FORCED_CHOICE_INSTRUCTION}"
    )


def build_prompt_for_framing(post: dict, framing: str) -> str:
    if framing == FRAMING_ORACLE:
        return build_oracle_prompt(post)
    if framing == FRAMING_PERSONAL:
        return build_personal_prompt(post)
    if framing == FRAMING_ADVISOR:
        return build_advisor_prompt(post)
    raise ValueError(f"Unknown framing: {framing}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def format_elapsed(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


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


def compute_cost_usd(model_key: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING[model_key]
    return (
        (input_tokens / 1_000_000.0) * pricing["input_per_1m"]
        + (output_tokens / 1_000_000.0) * pricing["output_per_1m"]
    )


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


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_output_dir(pilot: bool) -> Path:
    out_dir = config.EXPERIMENT_DIR / "direction2_pilot" if pilot else config.EXPERIMENT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "checkpoint": output_dir / "direction2_checkpoint.jsonl",
        "responses": output_dir / "direction2_responses.jsonl",
        "parse_failures": output_dir / "direction2_parse_failures.jsonl",
        "choice_rates": output_dir / "direction2_choice_rates.jsonl",
        "comparison": output_dir / "direction2_comparison_table.csv",
        "summary": output_dir / "direction2_summary.json",
    }


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


def dedupe_and_write_checkpoint(records: dict[tuple[str, int], dict], path: Path) -> None:
    ordered = sorted(
        records.values(),
        key=lambda rec: (rec["post_id"], rec["model_key"], rec["framing"], rec["sample_index"]),
    )
    with path.open("w", encoding="utf-8") as f:
        for rec in ordered:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Baseline loader (advisor framing from step5)
# ---------------------------------------------------------------------------

def load_advisor_baseline(post_ids: set[str], model_keys: list[str]) -> dict[tuple[str, str], list[dict]]:
    """Load step5 raw_responses.jsonl and return grouped records for advisor framing.

    Returns {(post_id, model_key): [records]} where each record is reshaped
    to match the direction2 schema.
    """
    if not BASELINE_RAW_PATH.exists():
        log.warning("Baseline file not found: %s -- advisor framing will be empty", BASELINE_RAW_PATH)
        return {}

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with BASELINE_RAW_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("post_id")
            mk = rec.get("model_key")
            if pid not in post_ids or mk not in model_keys:
                continue
            parsed = parse_choice(rec.get("raw_response")) if rec.get("error") is None else None
            grouped[(pid, mk)].append({
                "post_id": pid,
                "model_key": mk,
                "framing": FRAMING_ADVISOR,
                "sample_index": rec.get("sample_index", 0),
                "parsed_choice": parsed,
                "error": rec.get("error"),
            })
    return grouped


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

def call_model_with_retry(
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

    extra_body = dict(cfg.get("extra_body") or {})
    extra_body["stop"] = STOP_SEQUENCES

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
            extra_body=extra_body,
        )
        latency_ms = int((time.time() - t0) * 1000)
        usage = result.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        cost_usd = compute_cost_usd(model_key, input_tokens, output_tokens)
        return result.get("content"), latency_ms, None, input_tokens, output_tokens, cost_usd
    except LLMError as exc:
        return None, int((time.time() - t0) * 1000), str(exc), 0, 0, 0.0


# ---------------------------------------------------------------------------
# Task processing
# ---------------------------------------------------------------------------

def process_task(
    post: dict,
    model_key: str,
    framing: str,
    sample_index: int,
    prompt: str,
    p_hash: str,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
) -> dict:
    variant_id = f"{post['post_id']}__{model_key}__{framing}"
    if dry_run:
        return {
            "variant_id": variant_id,
            "post_id": post["post_id"],
            "model_key": model_key,
            "framing": framing,
            "sample_index": sample_index,
            "prompt_hash": p_hash,
            "raw_response": "[DRY RUN]",
            "parsed_choice": None,
            "latency_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "error": None,
            "timestamp": utc_now_iso(),
        }

    raw_response, latency_ms, error, input_tokens, output_tokens, cost_usd = call_model_with_retry(
        model_key=model_key,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed = parse_choice(raw_response) if error is None else None

    return {
        "variant_id": variant_id,
        "post_id": post["post_id"],
        "model_key": model_key,
        "framing": framing,
        "sample_index": sample_index,
        "prompt_hash": p_hash,
        "raw_response": raw_response,
        "parsed_choice": parsed,
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

def aggregate_per_cell(records: list[dict]) -> dict[tuple[str, str, str], dict]:
    """Group records by (post_id, model_key, framing) and compute per-cell stats."""
    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_cell[(r["post_id"], r["model_key"], r["framing"])].append(r)

    out: dict[tuple[str, str, str], dict] = {}
    for key, cell in by_cell.items():
        n_attempted = len(cell)
        valid = [r for r in cell if r.get("parsed_choice") in ("A", "B")]
        n_valid = len(valid)
        n_a = sum(1 for r in valid if r["parsed_choice"] == "A")
        n_b = sum(1 for r in valid if r["parsed_choice"] == "B")
        llm_risky_rate = (n_a / n_valid) if n_valid > 0 else None
        llm_entropy = binary_entropy(llm_risky_rate)

        out[key] = {
            "post_id": key[0],
            "model_key": key[1],
            "framing": key[2],
            "n_attempted": n_attempted,
            "n_valid": n_valid,
            "n_a": n_a,
            "n_b": n_b,
            "llm_risky_rate": llm_risky_rate,
            "llm_entropy": llm_entropy,
        }
    return out


def build_comparison_rows(
    posts_by_id: dict[str, dict],
    cells: dict[tuple[str, str, str], dict],
) -> list[dict]:
    rows: list[dict] = []
    for (pid, mk, framing), cell in cells.items():
        post = posts_by_id.get(pid)
        if not post:
            continue
        human_rr = human_risky_ratio(post)
        llm_rr = cell["llm_risky_rate"]

        mean_gap = (llm_rr - human_rr) if (llm_rr is not None and human_rr is not None) else None
        abs_gap = abs(mean_gap) if mean_gap is not None else None
        h_ent = binary_entropy(human_rr)
        l_ent = cell["llm_entropy"]
        ent_gap = (h_ent - l_ent) if (h_ent is not None and l_ent is not None) else None
        kl = binary_kl(human_rr, llm_rr)

        qualified = 1 if cell["n_valid"] >= config.MIN_VALID_SAMPLES else 0

        rows.append({
            "post_id": pid,
            "domain": post.get("domain"),
            "consensus_level": post.get("consensus_level"),
            "ses_level": post.get("ses_level"),
            "subreddit": post.get("subreddit"),
            "reversibility": post.get("reversibility"),
            "time_horizon": post.get("time_horizon"),
            "resource_constraint": post.get("resource_constraint"),
            "trade_off_type": post.get("trade_off_type"),
            "human_risky_ratio": human_rr,
            "human_entropy": h_ent,
            "model": mk,
            "framing": framing,
            "n_attempted": cell["n_attempted"],
            "n_valid": cell["n_valid"],
            "n_a": cell["n_a"],
            "n_b": cell["n_b"],
            "llm_risky_rate": llm_rr,
            "llm_entropy": l_ent,
            "mean_gap": mean_gap,
            "abs_gap": abs_gap,
            "entropy_gap": ent_gap,
            "kl_divergence": kl,
            "qualified_for_analysis": qualified,
        })
    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "post_id", "domain", "consensus_level", "ses_level", "subreddit",
    "reversibility", "time_horizon", "resource_constraint", "trade_off_type",
    "human_risky_ratio", "human_entropy",
    "model", "framing",
    "n_attempted", "n_valid", "n_a", "n_b",
    "llm_risky_rate", "llm_entropy",
    "mean_gap", "abs_gap", "entropy_gap", "kl_divergence",
    "qualified_for_analysis",
]


def write_comparison_csv(rows: list[dict], path: Path) -> None:
    ordered = sorted(
        rows,
        key=lambda r: (
            r.get("domain") or "",
            r.get("post_id") or "",
            r.get("model") or "",
            r.get("framing") or "",
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in ordered:
            w.writerow({
                k: (f"{r[k]:.6f}" if isinstance(r.get(k), float) else
                    ("" if r.get(k) is None else r.get(k)))
                for k in CSV_COLUMNS
            })


def write_choice_rates(cells: dict[tuple[str, str, str], dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for key in sorted(cells.keys()):
            f.write(json.dumps(cells[key], ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Cost and runtime projection
# ---------------------------------------------------------------------------

def project_cost(
    n_posts: int,
    model_keys: list[str],
    n_samples: int,
    framings: list[str],
    workers: int,
) -> dict:
    n_framings = len(framings)
    total_calls = n_posts * len(model_keys) * n_samples * n_framings
    est_input_tokens = 700
    est_output_tokens = 10
    est_usd = sum(
        n_posts * n_samples * n_framings * compute_cost_usd(mk, est_input_tokens, est_output_tokens)
        for mk in model_keys
    )
    avg_latency_s = 3.5
    eta_minutes = (total_calls * avg_latency_s) / max(workers, 1) / 60.0
    return {
        "n_posts": n_posts,
        "n_models": len(model_keys),
        "n_framings": n_framings,
        "n_samples": n_samples,
        "total_calls": total_calls,
        "est_usd": est_usd,
        "eta_minutes": eta_minutes,
    }


def print_cost_projection(
    n_posts: int,
    model_keys: list[str],
    n_samples: int,
    framings: list[str],
    workers: int,
    cost_cap: float,
) -> dict:
    proj = project_cost(n_posts, model_keys, n_samples, framings, workers)
    framing_label = " + ".join(f.capitalize() for f in framings)
    print("")
    print("DIRECTION 2 -- ORACLE vs ADVISOR FRAMING")
    print(f"  Posts:             {proj['n_posts']}")
    print(f"  Models:            {proj['n_models']}    ({', '.join(model_keys)})")
    print(f"  Framings:          {framing_label} (Advisor reused from step5)")
    print(f"  Samples per cell:  {n_samples}")
    print(f"  Workers:           {workers}")
    print(f"  Total NEW calls:   {proj['total_calls']} = {n_posts} x {proj['n_models']} x {n_samples} x {proj['n_framings']}")
    print("")
    print(f"ESTIMATED COST: ~${proj['est_usd']:.2f}")
    print(f"ESTIMATED RUNTIME with {workers} workers: ~{proj['eta_minutes']:.0f} minutes")
    print(f"HARD COST CAP: ${cost_cap:.2f}")
    print("")
    print("Press Ctrl+C within 10 seconds to abort.")
    print("")
    return proj


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(
    comparison_rows: list[dict],
    checkpoint_records: list[dict],
    advisor_cell_count: int,
    elapsed_seconds: float,
    n_posts: int,
    model_keys: list[str],
    n_samples: int,
) -> dict:
    by_model: dict[str, dict] = defaultdict(lambda: {
        "n_calls": 0, "api_errors": 0, "parse_failures": 0,
        "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
    })
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0

    for rec in checkpoint_records:
        mk = rec["model_key"]
        by_model[mk]["n_calls"] += 1
        by_model[mk]["input_tokens"] += int(rec.get("input_tokens") or 0)
        by_model[mk]["output_tokens"] += int(rec.get("output_tokens") or 0)
        by_model[mk]["cost_usd"] += float(rec.get("cost_usd") or 0.0)
        total_input_tokens += int(rec.get("input_tokens") or 0)
        total_output_tokens += int(rec.get("output_tokens") or 0)
        total_cost_usd += float(rec.get("cost_usd") or 0.0)
        if rec.get("error"):
            by_model[mk]["api_errors"] += 1
        elif rec.get("parsed_choice") not in ("A", "B"):
            by_model[mk]["parse_failures"] += 1

    for mk in by_model:
        by_model[mk]["cost_usd"] = round(by_model[mk]["cost_usd"], 6)

    parse_failures_by_framing = Counter()
    for rec in checkpoint_records:
        if rec.get("error") is None and rec.get("parsed_choice") not in ("A", "B"):
            parse_failures_by_framing[rec["framing"]] += 1

    qualified = sum(1 for r in comparison_rows if r.get("qualified_for_analysis") == 1)

    return {
        "n_posts": n_posts,
        "n_models": len(model_keys),
        "models": model_keys,
        "n_samples_per_cell": n_samples,
        "new_api_calls": len(checkpoint_records),
        "advisor_cells_reused": advisor_cell_count,
        "parse_failures_by_framing": dict(parse_failures_by_framing),
        "by_model": dict(sorted(by_model.items())),
        "cost_breakdown": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost_usd": round(total_cost_usd, 6),
        },
        "comparison_rows": len(comparison_rows),
        "qualified_cells": qualified,
        "unqualified_cells": len(comparison_rows) - qualified,
        "elapsed_seconds": round(elapsed_seconds, 1),
    }


def print_final_summary(comparison_rows: list[dict], elapsed_seconds: float, total_cost_usd: float, projected_cost: float) -> None:
    print("")
    print("=" * 64)
    print(f"DIRECTION 2 COMPLETE in {format_elapsed(elapsed_seconds)}")
    print("")

    print("Per-framing mean entropy (averaged across all posts and models):")
    for framing in ALL_FRAMINGS:
        vals = [r["llm_entropy"] for r in comparison_rows if r["framing"] == framing and r.get("llm_entropy") is not None]
        mean_val = (sum(vals) / len(vals)) if vals else None
        if mean_val is not None:
            print(f"  {framing:<20} {mean_val:.3f}")
        else:
            print(f"  {framing:<20} n/a")
    human_vals = [r["human_entropy"] for r in comparison_rows if r.get("human_entropy") is not None]
    human_mean = (sum(human_vals) / len(human_vals)) if human_vals else None
    if human_mean is not None:
        print(f"  {'human reference:':<20} {human_mean:.3f}")
    else:
        print(f"  {'human reference:':<20} n/a")
    print("")

    print("Per-framing mean abs_gap from human:")
    for framing in ALL_FRAMINGS:
        vals = [r["abs_gap"] for r in comparison_rows if r["framing"] == framing and r.get("abs_gap") is not None]
        mean_val = (sum(vals) / len(vals)) if vals else None
        if mean_val is not None:
            print(f"  {framing:<20} {mean_val:.3f}")
        else:
            print(f"  {framing:<20} n/a")
    print("")

    print("Per-framing mean entropy_gap (positive = LLM more confident than humans):")
    for framing in ALL_FRAMINGS:
        vals = [r["entropy_gap"] for r in comparison_rows if r["framing"] == framing and r.get("entropy_gap") is not None]
        mean_val = (sum(vals) / len(vals)) if vals else None
        if mean_val is not None:
            print(f"  {framing:<20} {mean_val:.3f}")
        else:
            print(f"  {framing:<20} n/a")
    print("")

    model_keys = sorted({r["model"] for r in comparison_rows})
    print("Per-model mean abs_gap by framing:")
    header = f"  {'':12}" + "".join(f"{f:<12}" for f in ALL_FRAMINGS)
    print(header)
    for mk in model_keys:
        parts = []
        for framing in ALL_FRAMINGS:
            vals = [r["abs_gap"] for r in comparison_rows
                    if r["model"] == mk and r["framing"] == framing and r.get("abs_gap") is not None]
            mean_val = (sum(vals) / len(vals)) if vals else None
            parts.append(f"{mean_val:.3f}" if mean_val is not None else "n/a")
        print(f"  {mk:<12}" + "".join(f"{p:<12}" for p in parts))
    print("")

    print(f"COST: ${total_cost_usd:.2f} (projected: ${projected_cost:.2f})")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 7: Direction 2 oracle vs advisor framing experiment."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print cost projection and sample prompts without API calls.")
    parser.add_argument("--pilot", action="store_true",
                        help="Run 5 random posts, all models, oracle+personal, N=5 samples. Writes to direction2_pilot/.")
    parser.add_argument("--framing", choices=NEW_FRAMINGS, default=None,
                        help="Run only one new framing (default: both oracle and personal).")
    parser.add_argument("--model", choices=list(config.STUDY_MODELS.keys()), default=None,
                        help="Run only one model key (default: all four).")
    parser.add_argument("--max-posts", type=int, default=None,
                        help="Cap number of posts (cost control).")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Thread-pool size for parallel API calls (default: {DEFAULT_WORKERS}).")
    parser.add_argument("--cost-cap", type=float, default=DEFAULT_COST_CAP,
                        help=f"Hard cost cap in USD (default: {DEFAULT_COST_CAP}).")
    args = parser.parse_args()

    n_samples = PILOT_N_SAMPLES if args.pilot else DEFAULT_N_SAMPLES
    temperature = DEFAULT_TEMPERATURE
    max_tokens = DEFAULT_MAX_TOKENS

    # -- Load posts --------------------------------------------------------
    if not INPUT_JSONL.exists():
        log.error("Missing %s -- run step2_score.py and step2b_validate.py first", INPUT_JSONL)
        sys.exit(1)

    all_posts = load_jsonl(INPUT_JSONL)
    all_posts_by_id = {p["post_id"]: p for p in all_posts}
    log.info("Loaded %d posts from %s", len(all_posts), INPUT_JSONL.name)

    posts = list(all_posts)
    if args.pilot:
        sample_size = min(PILOT_N_POSTS, len(posts))
        posts = sorted(PILOT_RNG.sample(posts, sample_size), key=lambda p: p["post_id"])
        log.info("Pilot mode: %d random posts (seed=42)", len(posts))
    if args.max_posts is not None:
        posts = posts[:args.max_posts]
        log.info("max-posts cap: %d posts", len(posts))

    posts_by_id = {p["post_id"]: p for p in posts}
    post_ids = set(posts_by_id.keys())

    # -- Pick models and framings ------------------------------------------
    model_keys = [args.model] if args.model else list(config.STUDY_MODELS.keys())
    run_framings = [args.framing] if args.framing else list(NEW_FRAMINGS)
    log.info("Models: %s", model_keys)
    log.info("New framings to run: %s", run_framings)

    # -- Validate API keys -------------------------------------------------
    if not args.dry_run:
        missing: list[str] = []
        for mk in model_keys:
            cfg = config.STUDY_MODELS[mk]
            if not cfg.get("api_key"):
                missing.append(f"{mk} ({cfg['provider']})")
        if missing:
            log.error("Missing API key(s) for: %s", ", ".join(missing))
            sys.exit(1)

    # -- Output paths ------------------------------------------------------
    output_dir = get_output_dir(args.pilot)
    paths = get_output_paths(output_dir)
    checkpoint_lock = threading.Lock()
    parsefail_lock = threading.Lock()

    # -- Cost projection ---------------------------------------------------
    proj = print_cost_projection(
        len(posts), model_keys, n_samples, run_framings, args.workers, args.cost_cap,
    )

    # -- Dry run -----------------------------------------------------------
    if args.dry_run:
        if posts:
            example = posts[0]
            log.info("Dry run -- sample prompts for post_id: %s", example["post_id"])
            for label, framing in [("ADVISOR", FRAMING_ADVISOR), ("ORACLE", FRAMING_ORACLE), ("PERSONAL", FRAMING_PERSONAL)]:
                prompt = build_prompt_for_framing(example, framing)
                log.info("----- %s PROMPT START -----", label)
                for ln in prompt.splitlines():
                    log.info("  %s", ln)
                log.info("----- %s PROMPT END -----", label)
        return

    # -- Load checkpoint and build task list --------------------------------
    checkpoint = load_checkpoint(paths["checkpoint"])
    log.info("Checkpoint: %d records already completed", len(checkpoint))
    running_cost_usd = sum(float(rec.get("cost_usd") or 0.0) for rec in checkpoint.values())
    if running_cost_usd > args.cost_cap:
        log.error(
            "Existing checkpoint cost already exceeds the cap: $%.4f > $%.2f",
            running_cost_usd, args.cost_cap,
        )
        sys.exit(1)

    tasks: list[tuple[dict, str, str, int, str, str]] = []
    for post in posts:
        for mk in model_keys:
            for framing in run_framings:
                prompt = build_prompt_for_framing(post, framing)
                p_hash = prompt_hash(prompt)
                variant_id = f"{post['post_id']}__{mk}__{framing}"
                for sample_index in range(n_samples):
                    key = (variant_id, sample_index)
                    rec = checkpoint.get(key)
                    if rec is not None and rec.get("error") is None and rec.get("parsed_choice") in ("A", "B"):
                        continue
                    tasks.append((post, mk, framing, sample_index, prompt, p_hash))

    total_pending = len(tasks)
    log.info("Pending API calls: %d", total_pending)

    if total_pending > 0:
        log.info("Starting in 10 seconds -- press Ctrl+C to abort.")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            log.info("Aborted before start.")
            return
    else:
        log.info("Nothing to do. Rebuilding outputs from checkpoint and baseline.")

    # -- Signal handling ---------------------------------------------------
    interrupted = threading.Event()

    def _sigint(_sig: int, _frame: object) -> None:
        if not interrupted.is_set():
            log.warning("Interrupt received -- finishing in-flight calls and stopping.")
            interrupted.set()

    signal.signal(signal.SIGINT, _sigint)

    # -- Run ---------------------------------------------------------------
    start = time.time()
    completed = 0
    model_progress: Counter = Counter()
    framing_progress: Counter = Counter()
    per_model_pending = Counter(t[1] for t in tasks)
    per_framing_pending = Counter(t[2] for t in tasks)
    cost_cap_exceeded = False
    from collections import deque
    pending_tasks: deque = deque(tasks)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        active_futures: dict = {}

        def submit_more() -> None:
            while (
                pending_tasks
                and len(active_futures) < max(1, args.workers)
                and not interrupted.is_set()
                and not cost_cap_exceeded
            ):
                post, mk, framing, idx, prompt, p_hash = pending_tasks.popleft()
                future = executor.submit(
                    process_task, post, mk, framing, idx, prompt, p_hash,
                    temperature, max_tokens, False,
                )
                active_futures[future] = (post, mk, framing, idx)

        submit_more()

        while active_futures:
            done_futures, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
            for future in done_futures:
                post, mk, framing, idx = active_futures.pop(future)
                try:
                    rec = future.result()
                except Exception as exc:
                    log.error("Unexpected exception for %s/%s/%s/%d: %s",
                              post["post_id"], mk, framing, idx, exc)
                    continue

                key = (rec["variant_id"], rec["sample_index"])
                old_cost = float(checkpoint.get(key, {}).get("cost_usd") or 0.0)
                checkpoint[key] = rec
                running_cost_usd += float(rec.get("cost_usd") or 0.0) - old_cost
                append_jsonl(paths["checkpoint"], rec, checkpoint_lock)

                if rec.get("error"):
                    log.warning("API error [%s/%s/%s/%d]: %s",
                                rec["post_id"], rec["model_key"], rec["framing"],
                                rec["sample_index"], rec["error"])
                elif rec.get("parsed_choice") not in ("A", "B"):
                    append_jsonl(paths["parse_failures"], {
                        "variant_id": rec["variant_id"],
                        "framing": rec["framing"],
                        "model_key": rec["model_key"],
                        "sample_index": rec["sample_index"],
                        "raw_response": rec.get("raw_response"),
                    }, parsefail_lock)
                    log.warning("Parse failure [%s/%s/%s/%d]: %r",
                                rec["post_id"], rec["model_key"], rec["framing"],
                                rec["sample_index"], (rec.get("raw_response") or "")[:120])

                completed += 1
                model_progress[rec["model_key"]] += 1
                framing_progress[rec["framing"]] += 1

                if completed % PROGRESS_EVERY == 0 or completed == total_pending:
                    pct = 100.0 * completed / max(total_pending, 1)
                    framing_str = ", ".join(
                        f"{f}: {framing_progress[f]}/{per_framing_pending[f]}"
                        for f in run_framings
                    )
                    model_str = ", ".join(
                        f"{mk}: {model_progress[mk]}/{per_model_pending[mk]}"
                        for mk in model_keys
                    )
                    log.info(
                        "Direction 2 progress: %d/%d (%.1f%%) | cost so far: $%.2f | "
                        "%s | %s | elapsed %s",
                        completed, total_pending, pct, running_cost_usd,
                        framing_str, model_str, format_elapsed(time.time() - start),
                    )

                if running_cost_usd > args.cost_cap and not cost_cap_exceeded:
                    cost_cap_exceeded = True
                    interrupted.set()
                    log.error(
                        "Cost cap exceeded: $%.4f > $%.2f. Stopping. "
                        "Resume with a higher --cost-cap if intended.",
                        running_cost_usd, args.cost_cap,
                    )

            if cost_cap_exceeded:
                for f in list(active_futures):
                    if f.cancel():
                        active_futures.pop(f, None)
            submit_more()

    elapsed = time.time() - start

    # -- Load advisor baseline and merge -----------------------------------
    log.info("")
    log.info("Loading advisor baseline from step5...")
    advisor_grouped = load_advisor_baseline(
        set(all_posts_by_id.keys()), list(config.STUDY_MODELS.keys()),
    )
    advisor_records: list[dict] = []
    for recs in advisor_grouped.values():
        advisor_records.extend(recs)
    log.info("  Loaded %d advisor baseline records", len(advisor_records))

    # -- Aggregate ---------------------------------------------------------
    log.info("Aggregating results...")
    new_records = [
        rec for rec in checkpoint.values()
        if rec["post_id"] in all_posts_by_id
    ]
    all_records = advisor_records + new_records
    cells = aggregate_per_cell(all_records)
    comparison_rows = build_comparison_rows(all_posts_by_id, cells)
    log.info("  Comparison rows: %d", len(comparison_rows))

    # -- Write outputs -----------------------------------------------------
    dedupe_and_write_checkpoint(checkpoint, paths["responses"])
    log.info("  Wrote %s (%d rows)", paths["responses"].name, len(checkpoint))
    write_choice_rates(cells, paths["choice_rates"])
    log.info("  Wrote %s (%d rows)", paths["choice_rates"].name, len(cells))
    write_comparison_csv(comparison_rows, paths["comparison"])
    log.info("  Wrote %s (%d rows)", paths["comparison"].name, len(comparison_rows))

    advisor_cell_count = sum(1 for k in cells if k[2] == FRAMING_ADVISOR)
    summary = build_summary(
        comparison_rows, new_records, advisor_cell_count,
        elapsed, len(posts), model_keys, n_samples,
    )
    with paths["summary"].open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("  Wrote %s", paths["summary"].name)

    print_final_summary(comparison_rows, elapsed, running_cost_usd, proj["est_usd"])

    if cost_cap_exceeded:
        sys.exit(1)


if __name__ == "__main__":
    main()
