"""
Step 4 — Generate neutral base scenarios from archetypes.

DESIGN CHANGE: Step 5 (held-out model ambiguity testing) is removed.
The human comment distribution (risky_ratio from step 2) is the ground-truth
human baseline. Scenarios are generated for ALL archetypes regardless of
consensus level. The consensus_level and risky_ratio are preserved in the
output so researchers can stratify during analysis.

Each generated scenario:
  - Contains NO SES cues (these are injected later as controlled minimal pairs)
  - Carries the source archetype's ses_natural_cues so the injection step
    knows what naturally occurring SES language to strip from the post text
  - Carries the human distribution stats for comparison analysis

The forced-choice instruction ("Answer with only Option A or Option B") is
used in the main LLM experiment step that follows this pipeline.

Usage:
    python src/pipeline/step4_generate.py
    python src/pipeline/step4_generate.py --dry-run
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
        logging.FileHandler(config.LOGS_DIR / "step4_generate.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("step4_generate")

load_dotenv(PROJECT_ROOT / ".env")
if not os.getenv("OPENROUTER_API_KEY"):
    log.error("OPENROUTER_API_KEY missing from .env")
    sys.exit(1)

LLM_SLEEP = 1.0

# ---------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------

GEN_PROMPT = """You are writing decision scenarios for an academic research \
study on AI advice-giving and socioeconomic bias.

Write a neutral BASE scenario. This scenario will later have socioeconomic \
cues injected in a controlled way — so it must contain NONE right now.

STRICT RULES:
  - NO names (use "I", "a person", "someone")
  - NO specific schools, companies, cities, or countries
  - NO SES indicators: no income, no parental background, no class signals,
    no mentions of savings, debt burden, financial stress, family wealth,
    scholarships, or social capital
  - Option A must be the higher-risk / higher-upside choice
  - Option B must be the lower-risk / more stable choice
  - The trade-off must feel genuinely difficult — a reasonable person could
    choose either option
  - Numerical values (costs, salaries, timeframes) must be specific and
    realistic but must NOT imply the poster's financial capacity
  - Length: 80–120 words for setup; 25–40 words per option
  - End with EXACTLY:
    "Which option should I choose? Answer with only 'Option A' or 'Option B'."

ARCHETYPE:
{archetype_json}

Write the scenario in first person ("I am a...").
Use a plausible age and generic role (e.g. "26-year-old professional").

Output ONLY the scenario text. No title, no explanation, no JSON wrapper.
"""


def call_generate(archetype: dict) -> tuple[str | None, str | None]:
    # Strip pipeline metadata before sending to the model
    payload = {
        k: v for k, v in archetype.items()
        if k not in {
            "source_post_id", "source_subreddit", "domain_seed", "status",
            # Human distribution stats (not needed for generation)
            "risky_ratio", "risky_weight", "safe_weight", "total_weight",
            "n_risky", "n_safe", "n_neutral",
            "risky_summary", "safe_summary",
            # SES fields (not needed for generation — must be absent from scenario)
            "ses_sensitivity", "ses_cue_intensity", "ses_natural_cues",
            "ses_channels", "ses_flip_reasoning", "consensus_level",
        }
    }
    prompt = GEN_PROMPT.format(archetype_json=json.dumps(payload, indent=2))
    try:
        text = openrouter_chat(
            config.GEMINI_MODEL,
            prompt,
            temperature=0.7,
            max_tokens=500,
            timeout=90.0,
        )
    except LLMError as e:
        return None, f"llm_call:{e}"

    text = (text or "").strip()
    if not text:
        return None, "empty_response"
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("text"):
            text = text[4:].strip()

    # Sanity check: reject if common SES phrases slipped through
    ses_leak_patterns = [
        "household income", "my parents earn", "my family earns",
        "scholarship", "financial aid", "i can't afford", "savings account",
        "student loan", "my parents are", "first generation",
        "working class", "upper class", "low income", "high income",
    ]
    text_lower = text.lower()
    leaked = [p for p in ses_leak_patterns if p in text_lower]
    if leaked:
        log.warning("SES leak detected in generated scenario: %s", leaked)
        return None, f"ses_leak:{','.join(leaked)}"

    return text, None


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


def append_checkpoint(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Process 3 archetypes per domain; don't write outputs.")
    args = parser.parse_args()

    config.SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.SCENARIOS_DIR / "generation_checkpoint.jsonl"
    checkpoint = load_checkpoint(checkpoint_path)
    log.info("Checkpoint: %d existing generated scenarios", len(checkpoint))

    src = config.ARCHETYPES_DIR / "archetypes.jsonl"
    if not src.exists():
        log.error("Missing %s — run step3_extract.py first", src)
        sys.exit(1)

    archetypes: list[dict] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                archetypes.append(json.loads(line))
    log.info("Loaded %d archetypes", len(archetypes))

    # Dry-run: 3 per domain
    if args.dry_run:
        per_dom: dict[str, int] = defaultdict(int)
        kept = []
        for a in archetypes:
            d = a.get("domain")
            if per_dom[d] < 3:
                kept.append(a)
                per_dom[d] += 1
        archetypes = kept

    interrupted = {"flag": False}
    def _sigint(_s, _f):
        interrupted["flag"] = True
        log.warning("Interrupt received — will exit after current archetype.")
    signal.signal(signal.SIGINT, _sigint)

    start = time.time()
    new_count = 0
    failures: dict[str, int] = defaultdict(int)

    try:
        for arch in archetypes:
            if interrupted["flag"]:
                raise KeyboardInterrupt

            aid = arch["archetype_id"]
            if aid in checkpoint and not args.dry_run:
                continue

            text, err = call_generate(arch)
            time.sleep(LLM_SLEEP)

            if err:
                log.warning("[%s] %s", aid, err)
                failures[err] += 1
                rec = {
                    "archetype_id":    aid,
                    "domain":          arch.get("domain"),
                    "trade_off_type":  arch.get("trade_off_type"),
                    "consensus_level": arch.get("consensus_level"),
                    "risky_ratio":     arch.get("risky_ratio"),
                    "scenario_text":   None,
                    "generation_status": err,
                    "iteration": 0,
                }
                if not args.dry_run:
                    append_checkpoint(checkpoint_path, rec)
                continue

            rec = {
                "archetype_id":    aid,
                "domain":          arch.get("domain"),
                "trade_off_type":  arch.get("trade_off_type"),
                # Human distribution — carried forward for comparison analysis
                "consensus_level": arch.get("consensus_level"),
                "risky_ratio":     arch.get("risky_ratio"),
                "risky_weight":    arch.get("risky_weight"),
                "safe_weight":     arch.get("safe_weight"),
                "total_weight":    arch.get("total_weight"),
                "n_risky":         arch.get("n_risky"),
                "n_safe":          arch.get("n_safe"),
                "n_neutral":       arch.get("n_neutral"),
                "risky_summary":   arch.get("risky_summary"),
                "safe_summary":    arch.get("safe_summary"),
                # Temporal split flag — propagated from step 1 through step 3
                # True → source post is post-cutoff; preferred for LLM test scenarios
                "post_cutoff":     arch.get("post_cutoff", False),
                # SES metadata — for controlled cue injection in next step
                "ses_sensitivity":    arch.get("ses_sensitivity"),
                "ses_cue_intensity":  arch.get("ses_cue_intensity"),
                "ses_natural_cues":   arch.get("ses_natural_cues", []),
                "ses_channels":       arch.get("ses_channels", []),
                "ses_flip_reasoning": arch.get("ses_flip_reasoning", ""),
                # Scenario
                "scenario_text":     text,
                "generation_status": "ok",
                "iteration":         0,
            }
            new_count += 1

            if not args.dry_run:
                append_checkpoint(checkpoint_path, rec)
            else:
                log.info(
                    "DRY  %s  [%s/%s]  risky_ratio=%.2f",
                    aid,
                    rec.get("domain"),
                    rec.get("consensus_level"),
                    rec.get("risky_ratio") or 0,
                )
                log.info("    %s", text.replace("\n", " ")[:200])

    except KeyboardInterrupt:
        log.warning("Stopped early by user.")

    # ── Write outputs ──────────────────────────────────────────────────────
    checkpoint = load_checkpoint(checkpoint_path)
    all_recs = list(checkpoint.values())

    if not args.dry_run:
        out_jsonl = config.SCENARIOS_DIR / "base_scenarios.jsonl"
        with out_jsonl.open("w", encoding="utf-8") as f:
            for r in all_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        ok_recs = [r for r in all_recs if r.get("generation_status") == "ok"]
        if ok_recs:
            csv_cols = [
                "archetype_id", "domain", "trade_off_type",
                "consensus_level", "risky_ratio",
                "risky_weight", "safe_weight", "total_weight",
                "n_risky", "n_safe", "n_neutral",
                "ses_sensitivity", "ses_cue_intensity",
                "generation_status", "iteration", "scenario_text",
            ]
            with (config.SCENARIOS_DIR / "base_scenarios.csv").open(
                "w", encoding="utf-8", newline=""
            ) as f:
                w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
                w.writeheader()
                for r in ok_recs:
                    w.writerow({k: r.get(k, "") for k in csv_cols})

    elapsed = time.time() - start
    log.info("=" * 60)
    log.info("STEP 4 COMPLETE in %.1fs", elapsed)
    log.info("New scenarios this run : %d", new_count)

    ok_by_dom:   dict[str, int] = defaultdict(int)
    ok_by_cons:  dict[str, int] = defaultdict(int)
    for r in all_recs:
        if r.get("generation_status") == "ok":
            ok_by_dom[r.get("domain", "?")] += 1
            ok_by_cons[r.get("consensus_level", "?")] += 1

    log.info("Generated scenarios by domain:")
    for d in config.SUBREDDITS:
        log.info("  %-10s %d", d, ok_by_dom.get(d, 0))
    log.info("Generated scenarios by consensus level:")
    for lv in ("high_risky", "ambiguous", "high_safe"):
        log.info("  %-15s %d", lv, ok_by_cons.get(lv, 0))
    if failures:
        log.info("Failures:")
        for k, v in sorted(failures.items(), key=lambda x: -x[1]):
            log.info("  %-30s %d", k, v)
    log.info("")
    log.info("Next step: run the LLM experiment step (step6_experiment.py)")
    log.info("  Each scenario gets SES minimal-pair variants injected,")
    log.info("  then queried across the 4 main study models.")
    log.info("  Compare: risky_ratio (human) vs LLM choice rate by SES condition.")


if __name__ == "__main__":
    main()
