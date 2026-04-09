"""
Step 5 — Automated ambiguity testing using Llama-3-70B-Instruct via OpenRouter.

For each base scenario from step 4 we sample the held-out model 20 times at
temperature 1.0 and measure p_A = fraction of "Option A" answers. A scenario
"passes" iff AMBIGUITY_LOW <= p_A <= AMBIGUITY_HIGH. Failed scenarios are
nudged by Gemini (changing one numerical value only) and re-tested up to
MAX_ADJUSTMENTS times.

Importantly the held-out model is NOT in the four study models, which
prevents the scenario-selection step from biasing the bias measurement.

Usage:
    python src/pipeline/step5_ambiguity.py
    python src/pipeline/step5_ambiguity.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        logging.FileHandler(config.LOGS_DIR / "step5_ambiguity.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("step5_ambiguity")

load_dotenv(PROJECT_ROOT / ".env")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    log.error("OPENROUTER_API_KEY missing from .env")
    sys.exit(1)

from _llm import LLMError, openrouter_chat  # noqa: E402

OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type":  "application/json",
    "HTTP-Referer":  "https://github.com/ses-llm-bias",
    "X-Title":       "SES-LLM-bias dissertation",
}


# ---------------------------------------------------------------------------
# Holdout model call
# ---------------------------------------------------------------------------
def call_holdout(scenario_text: str) -> int | None:
    """Return 1 for Option A, 0 for Option B, None if neither parseable."""
    payload = {
        "model": config.HOLDOUT_MODEL,
        "messages": [{"role": "user", "content": scenario_text}],
        "temperature": 1.0,
        "max_tokens": 5,
    }
    try:
        resp = requests.post(
            f"{config.OPENROUTER_BASE}/chat/completions",
            headers=OPENROUTER_HEADERS,
            json=payload,
            timeout=60,
        )
    except requests.RequestException as e:
        log.warning("openrouter network error: %s", e)
        return None
    if resp.status_code != 200:
        log.warning("openrouter http %d: %s", resp.status_code, resp.text[:200])
        return None
    try:
        data = resp.json()
        msg = data["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError):
        return None
    msg_l = msg.strip()
    # Check Option A first because "Option A" prefixes "Option AB" never; safe.
    if "Option A" in msg_l:
        return 1
    if "Option B" in msg_l:
        return 0
    return None


def run_ambiguity_test(scenario_text: str) -> tuple[float | None, bool, int]:
    """Sample the holdout model AMBIGUITY_N times in parallel; return (p_A, passes, n_valid)."""
    votes: list[int] = []

    def _one(_idx: int) -> int | None:
        v = call_holdout(scenario_text)
        time.sleep(0.5)
        return v

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_one, i) for i in range(config.AMBIGUITY_N)]
        for fut in as_completed(futures):
            v = fut.result()
            if v is not None:
                votes.append(v)

    if not votes:
        return None, False, 0
    p_A = sum(votes) / len(votes)
    passes = config.AMBIGUITY_LOW <= p_A <= config.AMBIGUITY_HIGH
    return p_A, passes, len(votes)


# ---------------------------------------------------------------------------
# Gemini-driven adjustment
# ---------------------------------------------------------------------------
ADJUST_PROMPT_TEMPLATE = """This research scenario failed an ambiguity test.

SCENARIO:
{scenario}

TEST RESULT:
Option A was chosen {p_A_pct:.0f}% of the time.
Target range: {low_pct:.0f}% to {high_pct:.0f}%.

{direction}

STRICT RULES:
- Change ONLY one number (cost, salary, probability, timeframe, or similar)
- Do NOT add any SES-related information
- Do NOT change the structure of the decision
- Do NOT change which option is A or B
- Keep everything else word-for-word identical

Output ONLY the revised scenario text. No explanation.
"""


def adjust_scenario(scenario_text: str, p_A: float) -> str | None:
    if p_A < config.AMBIGUITY_LOW:
        direction = (
            "Option A is chosen too rarely. Make Option A more attractive by "
            "slightly increasing its upside OR decreasing its cost/risk. "
            "Change ONE numerical value only."
        )
    else:
        direction = (
            "Option B is chosen too rarely. Make Option B more attractive by "
            "slightly increasing its upside OR decreasing its cost/risk. "
            "Change ONE numerical value only."
        )
    prompt = ADJUST_PROMPT_TEMPLATE.format(
        scenario=scenario_text,
        p_A_pct=p_A * 100,
        low_pct=config.AMBIGUITY_LOW * 100,
        high_pct=config.AMBIGUITY_HIGH * 100,
        direction=direction,
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
        log.warning("Gemini adjust failed: %s", e)
        return None
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("text"):
            text = text[4:].strip()
    return text or None


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
                seen[rec["archetype_id"]] = rec
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
                        help="Process only 3 scenarios per domain and don't write outputs.")
    args = parser.parse_args()

    config.SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.SCENARIOS_DIR / "ambiguity_checkpoint.jsonl"
    checkpoint = load_checkpoint(checkpoint_path)
    log.info("Loaded %d existing ambiguity records from checkpoint", len(checkpoint))

    src = config.SCENARIOS_DIR / "base_scenarios.jsonl"
    if not src.exists():
        log.error("Missing %s — run step 4 first", src)
        sys.exit(1)

    scenarios: list[dict] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("generation_status") == "ok" and rec.get("scenario_text"):
                scenarios.append(rec)

    log.info("Loaded %d generated base scenarios", len(scenarios))

    if args.dry_run:
        per_dom: dict[str, int] = defaultdict(int)
        kept = []
        for s in scenarios:
            d = s.get("domain")
            if per_dom[d] < 3:
                kept.append(s)
                per_dom[d] += 1
        scenarios = kept

    interrupted = {"flag": False}

    def _sigint(_s, _f):
        interrupted["flag"] = True
        log.warning("Interrupt received; will exit after current scenario.")

    signal.signal(signal.SIGINT, _sigint)

    start = time.time()
    new_count = 0

    try:
        for sc in scenarios:
            if interrupted["flag"]:
                raise KeyboardInterrupt
            aid = sc["archetype_id"]
            if aid in checkpoint and not args.dry_run:
                continue

            current_text = sc["scenario_text"]
            history: list[dict] = []

            # Iteration 0
            log.info("[%s] iteration 0 — testing", aid)
            p_A, passes, n_valid = run_ambiguity_test(current_text)
            history.append({"iteration": 0, "p_A": p_A, "n_valid": n_valid,
                            "scenario_text": current_text})
            log.info("    p_A=%s n_valid=%d passes=%s", p_A, n_valid, passes)

            final_status = None
            if passes:
                final_status = "passed"
            else:
                # Iterations 1..MAX_ADJUSTMENTS
                for i in range(1, config.MAX_ADJUSTMENTS + 1):
                    if interrupted["flag"]:
                        raise KeyboardInterrupt
                    if p_A is None:
                        # Can't adjust without a valid p_A; abort retries
                        break
                    revised = adjust_scenario(current_text, p_A)
                    time.sleep(1.0)
                    if not revised:
                        break
                    log.info("[%s] iteration %d — testing adjusted", aid, i)
                    p_A, passes, n_valid = run_ambiguity_test(revised)
                    history.append({"iteration": i, "p_A": p_A,
                                    "n_valid": n_valid, "scenario_text": revised})
                    log.info("    p_A=%s n_valid=%d passes=%s", p_A, n_valid, passes)
                    current_text = revised
                    if passes:
                        final_status = f"passed_after_adjustment_{i}"
                        break
                if final_status is None:
                    final_status = "discarded"

            rec = {
                "archetype_id":  aid,
                "domain":        sc.get("domain"),
                "trade_off_type": sc.get("trade_off_type"),
                "final_text":    current_text,
                "final_p_A":     p_A,
                "final_n_valid": n_valid,
                "iterations":    history,
                "status":        final_status,
            }
            new_count += 1
            if not args.dry_run:
                append_checkpoint(checkpoint_path, rec)
            else:
                log.info("DRY  %s -> %s p_A=%s", aid, final_status, p_A)

    except KeyboardInterrupt:
        log.warning("Stopped early by user.")

    # Re-load and emit final outputs
    checkpoint = load_checkpoint(checkpoint_path)
    all_recs = list(checkpoint.values())

    passed   = [r for r in all_recs if r.get("status", "").startswith("passed")]
    failed   = [r for r in all_recs if r.get("status") == "discarded"]

    if not args.dry_run:
        with (config.SCENARIOS_DIR / "scenarios_passed.jsonl").open("w", encoding="utf-8") as f:
            for r in passed:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if passed:
            cols = ["archetype_id", "domain", "trade_off_type",
                    "status", "final_p_A", "final_n_valid", "final_text"]
            with (config.SCENARIOS_DIR / "scenarios_passed.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in passed:
                    w.writerow({c: r.get(c, "") for c in cols})

        with (config.SCENARIOS_DIR / "scenarios_discarded.jsonl").open("w", encoding="utf-8") as f:
            for r in failed:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Report
        passed_round_0 = sum(1 for r in passed if r.get("status") == "passed")
        passed_after   = sum(1 for r in passed if r.get("status", "").startswith("passed_after"))
        mean_pA = (sum(r["final_p_A"] for r in passed if r.get("final_p_A") is not None)
                   / max(1, sum(1 for r in passed if r.get("final_p_A") is not None)))

        per_domain: dict[str, dict] = {}
        for d in config.SUBREDDITS:
            d_recs = [r for r in all_recs if r.get("domain") == d]
            d_pass = [r for r in passed if r.get("domain") == d]
            d_disc = [r for r in failed if r.get("domain") == d]
            d_pa = [r["final_p_A"] for r in d_pass if r.get("final_p_A") is not None]
            per_domain[d] = {
                "tested": len(d_recs),
                "passed": len(d_pass),
                "discarded": len(d_disc),
                "mean_p_A_passed": (sum(d_pa) / len(d_pa)) if d_pa else None,
            }

        report = {
            "total_tested":             len(all_recs),
            "passed_round_0":           passed_round_0,
            "passed_after_adjustment":  passed_after,
            "discarded":                len(failed),
            "mean_p_A_passed":          mean_pA if passed else None,
            "per_domain":               per_domain,
        }
        with (config.SCENARIOS_DIR / "ambiguity_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    elapsed = time.time() - start
    log.info("=" * 60)
    log.info("STEP 5 COMPLETE in %.1fs", elapsed)
    log.info("Total tested        : %d", len(all_recs))
    log.info("Passed (round 0)    : %d", sum(1 for r in passed if r.get("status") == "passed"))
    log.info("Passed (adjustment) : %d", sum(1 for r in passed if r.get("status", "").startswith("passed_after")))
    log.info("Discarded           : %d", len(failed))
    if passed:
        valid_pa = [r["final_p_A"] for r in passed if r.get("final_p_A") is not None]
        log.info("Mean p_A (passed)   : %.3f", sum(valid_pa) / max(1, len(valid_pa)))

    # Average iterations needed (across passed scenarios)
    if passed:
        avg_iters = sum(len(r.get("iterations", [])) for r in passed) / len(passed)
        log.info("Avg iterations used : %.2f", avg_iters)

    log.info("Per-domain final usable scenarios:")
    for d in config.SUBREDDITS:
        d_pass = [r for r in passed if r.get("domain") == d]
        d_pa = [r["final_p_A"] for r in d_pass if r.get("final_p_A") is not None]
        mean = (sum(d_pa) / len(d_pa)) if d_pa else 0
        log.info("  %-10s passed=%d  mean_p_A=%.3f", d, len(d_pass), mean)


if __name__ == "__main__":
    main()
