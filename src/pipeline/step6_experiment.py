"""
Step 6 — LLM experiment: strip natural SES cues, inject minimal-pair cues,
query four study models, compute llm_risky_rate, build comparison table.

This is the core measurement step. For each base scenario from step 4 it:

  1. STRIPS natural SES cues from the scenario text using the ses_natural_cues
     field extracted in step 2 (verbatim phrases that signal SES in the
     original post text). Improved cleanup handles orphaned punctuation.

  2. INJECTS controlled minimal-pair SES cues: exactly one sentence per
     dimension (income / parental_education / occupation / first_gen),
     creating 8 variants per prompt family (4 dimensions × 2 levels).

  3. RUNS both prompt families per scenario:
       constraint_matched (CM)  — SES identity only; material constraints fixed
       constraint_varying  (CV) — SES co-varies with material constraints
     CM isolates pure identity signalling; CV tests the full SES effect.

  4. QUERIES all four study models with forced-choice instruction.
     Each prompt → 5 samples at T=0.7 → llm_risky_rate = fraction choosing A.
     Justification suffix applied to a pre-seeded 50% subsample (CM only).

  5. BUILDS the comparison table per scenario per model per SES condition:
       human_risky_ratio      (from step 2 Reddit comments)
       llm_risky_rate_low     (model choice rate under low-SES framing)
       llm_risky_rate_high    (model choice rate under high-SES framing)
       ses_gap                = llm_risky_rate_high - llm_risky_rate_low
       alignment_gap_low      = llm_risky_rate_low  - human_risky_ratio
       alignment_gap_high     = llm_risky_rate_high - human_risky_ratio
       h3_signal              = alignment_gap_high  - alignment_gap_low

  RQ3 (H3): h3_signal > 0 — LLMs selectively push low-SES users further
  from human consensus while keeping high-SES users closer to it.

Output files
------------
data/experiment/raw_responses.jsonl       — every individual LLM call
data/experiment/choice_rates.jsonl        — aggregated choice rates per prompt
data/experiment/comparison_table.csv      — the main analysis table
data/experiment/justifications.jsonl      — justification-condition responses (CM only)

New fields in all records (v5.0)
---------------------------------
  prompt_family  — "constraint_matched" or "constraint_varying"
  post_cutoff    — True if source post is after training cutoff (contamination flag)

Usage
-----
    python src/pipeline/step6_experiment.py
    python src/pipeline/step6_experiment.py --dry-run
    python src/pipeline/step6_experiment.py --model gpt4o
    python src/pipeline/step6_experiment.py --scenario EDU_01
    python src/pipeline/step6_experiment.py --family constraint_matched
    python src/pipeline/step6_experiment.py --post-cutoff-only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import signal
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from _llm import LLMError, openrouter_chat

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
config.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "step6_experiment.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("step6_experiment")

load_dotenv(PROJECT_ROOT / ".env")
if not os.getenv("OPENROUTER_API_KEY"):
    log.error("OPENROUTER_API_KEY missing from .env")
    sys.exit(1)

# Seeded RNG for reproducible justification subsample selection
_RNG = random.Random(42)

# ---------------------------------------------------------------------------
# SES cue stripping
# ---------------------------------------------------------------------------

def strip_ses_cues(scenario_text: str, natural_cues: list[str]) -> str:
    """Remove verbatim SES-signalling phrases from scenario text.

    Each phrase in natural_cues was extracted verbatim by the step 2 LLM.
    We remove the phrase and any surrounding punctuation/whitespace, then
    re-collapse double spaces and fix orphaned sentence-boundary punctuation.
    If a cue is not found verbatim (LLM hallucination), it is silently skipped.
    """
    text = scenario_text
    for phrase in natural_cues:
        if not phrase or len(phrase.strip()) < 4:
            continue
        # Remove the phrase along with optional trailing punctuation and
        # any surrounding whitespace so no orphaned comma/period remains.
        pattern = r"\s*" + re.escape(phrase.strip()) + r"[.,;:]?\s*"
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Fix sentence-boundary artefacts: e.g. ". ." → "." or ", ," → ","
    text = re.sub(r"([.!?,;:])\s+\1", r"\1", text)
    # Collapse multiple spaces and strip
    text = re.sub(r"  +", " ", text).strip()
    # Fix space before punctuation that was exposed by removal
    text = re.sub(r" ([.,;:])", r"\1", text)
    return text


# ---------------------------------------------------------------------------
# SES cue injection (minimal pair)
# ---------------------------------------------------------------------------

def inject_ses_cue(
    scenario_text: str,
    dimension: str,
    level: str,
    prompt_family: str = "constraint_matched",
) -> str:
    """Insert one SES sentence after the first sentence of the scenario.

    For constraint_matched (CM): the injected sentence signals SES identity
    only — material constraints (savings, debt, income) are held constant in
    the scenario body.

    For constraint_varying (CV): the injected sentence is the same, but the
    scenario body may contain co-varying constraint language (set in step 4
    by passing family=cv to the generator). The CV condition is used to test
    the full SES effect, not just identity signalling.

    The insertion point is after the first period/exclamation/question mark
    so the cue sits naturally in the opening context.
    """
    cue_sentence = config.SES_MINIMAL_PAIRS[dimension][level]

    # Find insertion point: end of first sentence
    m = re.search(r"(?<=[.!?])\s+", scenario_text)
    if m:
        insert_at = m.start() + 1  # after the space
        return scenario_text[:insert_at] + cue_sentence + " " + scenario_text[insert_at:]
    else:
        # Fallback: prepend
        return cue_sentence + " " + scenario_text


def build_prompt(scenario_text: str, justification: bool = False) -> str:
    """Append the forced-choice or justification instruction."""
    instruction = (
        config.JUSTIFICATION_INSTRUCTION if justification
        else config.FORCED_CHOICE_INSTRUCTION
    )
    # Replace the existing instruction if already present (from step 4)
    base = re.sub(
        r"Which option should I choose\?.*",
        "",
        scenario_text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    return base + "\n\n" + instruction


# ---------------------------------------------------------------------------
# Single LLM call
# ---------------------------------------------------------------------------

def call_model(
    model_key: str,
    prompt: str,
    temperature: float,
    max_tokens: int = 10,
) -> tuple[str | None, str | None]:
    """Call one study model. Returns (response_text, error)."""
    model_path = config.STUDY_MODELS[model_key]
    try:
        text = openrouter_chat(
            model_path,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=60.0,
        )
        return text, None
    except LLMError as e:
        return None, str(e)


def parse_choice(response: str | None) -> int | None:
    """Return 1 for Option A, 0 for Option B, None if unparseable."""
    if not response:
        return None
    r = response.strip()
    if re.search(r"\bOption\s*A\b", r, re.IGNORECASE):
        return 1
    if re.search(r"\bOption\s*B\b", r, re.IGNORECASE):
        return 0
    return None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> set[str]:
    """Return set of variant_ids already processed."""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["variant_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_scenario(
    scenario: dict,
    model_key: str,
    dimension: str,
    level: str,
    is_justification: bool,
    dry_run: bool,
    prompt_family: str = "constraint_matched",
) -> dict:
    """Run N_SAMPLES calls for one (scenario, model, dimension, level, family) cell.

    Returns an aggregated record with:
      llm_risky_rate   — fraction of valid responses choosing Option A
      n_valid          — number of parseable responses
      raw_responses    — list of individual response texts
      prompt_family    — "constraint_matched" or "constraint_varying"
    """
    natural_cues = scenario.get("ses_natural_cues") or []
    base_text = strip_ses_cues(scenario.get("scenario_text", ""), natural_cues)
    injected  = inject_ses_cue(base_text, dimension, level, prompt_family)
    prompt    = build_prompt(injected, justification=is_justification)

    n_samples = config.EXPERIMENT_N_SAMPLES
    temperature = config.EXPERIMENT_TEMPERATURE
    max_tokens = 200 if is_justification else 50

    votes: list[int] = []
    raw_responses: list[str] = []
    errors: list[str] = []

    if dry_run:
        # Simulate without API calls
        import random as _r
        for _ in range(n_samples):
            votes.append(_r.randint(0, 1))
        raw_responses = ["[DRY RUN]"] * n_samples
    else:
        # Parallel sampling (thread-safe — each call is independent)
        def _one(_i: int) -> tuple[str | None, str | None]:
            return call_model(model_key, prompt, temperature, max_tokens)

        with ThreadPoolExecutor(max_workers=min(n_samples, 4)) as ex:
            futures = [ex.submit(_one, i) for i in range(n_samples)]
            for fut in as_completed(futures):
                text, err = fut.result()
                if err:
                    errors.append(err)
                    log.warning("    API error [%s]: %s", model_key, err)
                    continue
                raw_responses.append(text or "")
                v = parse_choice(text)
                if v is not None:
                    votes.append(v)
                time.sleep(0.3)   # gentle rate limiting

    n_valid = len(votes)
    llm_risky_rate = sum(votes) / n_valid if n_valid > 0 else None

    return {
        "llm_risky_rate": llm_risky_rate,
        "n_valid":        n_valid,
        "n_errors":       len(errors),
        "raw_responses":  raw_responses,
        "prompt":         prompt if dry_run else None,   # save full prompt only in dry-run
        "errors":         errors,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 6: LLM experiment — SES cue injection and comparison table."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate calls, write no outputs.")
    parser.add_argument("--model", choices=list(config.STUDY_MODELS.keys()),
                        default=None, help="Run only this model (default: all four).")
    parser.add_argument("--scenario", default=None,
                        help="Run only this archetype_id (e.g. EDU_01).")
    parser.add_argument("--family",
                        choices=config.PROMPT_FAMILIES,
                        default=None,
                        help="Run only this prompt family (default: both CM and CV).")
    parser.add_argument("--post-cutoff-only", action="store_true",
                        help="Skip scenarios whose source post predates the training cutoff "
                             "(contamination-safe subset for RQ2/RQ3).")
    args = parser.parse_args()

    # ── Load scenarios ─────────────────────────────────────────────────────
    src = config.SCENARIOS_DIR / "base_scenarios.jsonl"
    if not src.exists():
        log.error("Missing %s — run step4_generate.py first", src)
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

    if args.scenario:
        scenarios = [s for s in scenarios if s.get("archetype_id") == args.scenario]
    if args.post_cutoff_only:
        before = len(scenarios)
        scenarios = [s for s in scenarios if s.get("post_cutoff", False)]
        log.info("post-cutoff filter: %d → %d scenarios", before, len(scenarios))
    log.info("Loaded %d scenarios", len(scenarios))

    models_to_run = (
        {args.model: config.STUDY_MODELS[args.model]}
        if args.model else config.STUDY_MODELS
    )
    log.info("Models: %s", list(models_to_run.keys()))

    families_to_run = [args.family] if args.family else config.PROMPT_FAMILIES
    log.info("Prompt families: %s", families_to_run)

    # ── Pre-select justification subsample (reproducible) ─────────────────
    # Justification condition applied only within constraint_matched family
    # to keep the subsample size tractable and interpretation clean.
    all_cm_variant_ids = [
        f"{s['archetype_id']}__{mk}__{dim}__{level}__constraint_matched"
        for s in scenarios
        for mk in models_to_run
        for dim in config.SES_MINIMAL_PAIRS
        for level in ("low", "high")
    ]
    justification_set: set[str] = set(
        _RNG.sample(
            all_cm_variant_ids,
            k=int(len(all_cm_variant_ids) * config.JUSTIFICATION_FRACTION)
        )
    )
    log.info("Justification subsample (CM only): %d / %d variants",
             len(justification_set), len(all_cm_variant_ids))

    # ── Checkpoint ─────────────────────────────────────────────────────────
    checkpoint_path = config.EXPERIMENT_DIR / "raw_responses_checkpoint.jsonl"
    done = load_checkpoint(checkpoint_path)
    log.info("Checkpoint: %d variants already completed", len(done))

    # Output file handles
    raw_path    = config.EXPERIMENT_DIR / "raw_responses.jsonl"
    just_path   = config.EXPERIMENT_DIR / "justifications.jsonl"

    interrupted = {"flag": False}
    def _sigint(_s, _f):
        interrupted["flag"] = True
        log.warning("Interrupt — will finish current variant.")
    signal.signal(signal.SIGINT, _sigint)

    # ── Main loop ──────────────────────────────────────────────────────────
    start = time.time()
    total_variants = 0
    errors_total   = 0

    # Accumulate results for the comparison table.
    # key: (archetype_id, model_key, dimension, level, prompt_family)
    results: dict[tuple, dict] = {}

    try:
        for scenario in scenarios:
            if interrupted["flag"]:
                break
            aid = scenario["archetype_id"]

            for prompt_family in families_to_run:
                for model_key in models_to_run:
                    for dimension in config.SES_MINIMAL_PAIRS:
                        for level in ("low", "high"):
                            if interrupted["flag"]:
                                raise KeyboardInterrupt

                            variant_id = f"{aid}__{model_key}__{dimension}__{level}__{prompt_family}"
                            if variant_id in done and not args.dry_run:
                                continue

                            # Justification condition: CM family only, pre-seeded subsample
                            is_just = variant_id in justification_set
                            log.info(
                                "  %s  family=%-20s  model=%-8s  dim=%-20s  level=%-4s  just=%s",
                                aid, prompt_family, model_key, dimension, level, is_just
                            )

                            result = run_scenario(
                                scenario, model_key, dimension, level,
                                is_justification=is_just,
                                dry_run=args.dry_run,
                                prompt_family=prompt_family,
                            )
                            total_variants += 1
                            errors_total += result["n_errors"]

                            record = {
                                "variant_id":       variant_id,
                                "archetype_id":     aid,
                                "domain":           scenario.get("domain"),
                                "consensus_level":  scenario.get("consensus_level"),
                                "post_cutoff":      scenario.get("post_cutoff", False),
                                "human_risky_ratio":scenario.get("risky_ratio"),
                                "risky_summary":    scenario.get("risky_summary"),
                                "safe_summary":     scenario.get("safe_summary"),
                                "model_key":        model_key,
                                "model_path":       config.STUDY_MODELS[model_key],
                                "ses_dimension":    dimension,
                                "ses_level":        level,
                                "prompt_family":    prompt_family,
                                "is_justification": is_just,
                                "llm_risky_rate":   result["llm_risky_rate"],
                                "n_valid":          result["n_valid"],
                                "n_errors":         result["n_errors"],
                                "errors":           result["errors"],
                                "raw_responses":    result["raw_responses"],
                            }

                            if not args.dry_run:
                                append_jsonl(checkpoint_path, record)
                                append_jsonl(raw_path, record)
                                if is_just:
                                    append_jsonl(just_path, record)
                            else:
                                log.info(
                                    "    llm_risky_rate=%.2f  n_valid=%d  human=%.2f",
                                    result["llm_risky_rate"] or 0,
                                    result["n_valid"],
                                    scenario.get("risky_ratio") or 0,
                                )

                            results[(aid, model_key, dimension, level, prompt_family)] = record

    except KeyboardInterrupt:
        log.warning("Stopped early by user.")

    # ── Reload full checkpoint for comparison table ────────────────────────
    if not args.dry_run:
        all_raw: list[dict] = []
        if raw_path.exists():
            with raw_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_raw.append(json.loads(line))
        # Re-index by (aid, model, dim, level, prompt_family)
        for rec in all_raw:
            k = (
                rec.get("archetype_id"),
                rec.get("model_key"),
                rec.get("ses_dimension"),
                rec.get("ses_level"),
                rec.get("prompt_family", "constraint_matched"),  # back-compat
            )
            if None not in k:
                results[k] = rec

    # ── Build comparison table ─────────────────────────────────────────────
    # Group by (archetype_id, model_key, dimension, prompt_family)
    # For each group, pair low and high to compute gaps.
    comparison_rows: list[dict] = []

    # key: (aid, model, dim, family) → {low: rec, high: rec}
    grouped: dict[tuple, dict] = {}
    for (aid, mk, dim, level, fam), rec in results.items():
        key = (aid, mk, dim, fam)
        if key not in grouped:
            grouped[key] = {}
        grouped[key][level] = rec

    for (aid, mk, dim, fam), levels in grouped.items():
        low  = levels.get("low")
        high = levels.get("high")
        if not low or not high:
            continue

        human_rr = low.get("human_risky_ratio")
        llm_low  = low.get("llm_risky_rate")
        llm_high = high.get("llm_risky_rate")

        ses_gap        = (llm_high - llm_low)  if (llm_high is not None and llm_low is not None)  else None
        align_gap_low  = (llm_low  - human_rr) if (llm_low  is not None and human_rr is not None) else None
        align_gap_high = (llm_high - human_rr) if (llm_high is not None and human_rr is not None) else None
        # H3 signal: positive means high-SES LLM rate is closer to humans than low-SES rate
        # (i.e. LLMs push low-SES users further from human consensus)
        h3_signal = (align_gap_high - align_gap_low) if (align_gap_high is not None and align_gap_low is not None) else None

        comparison_rows.append({
            "archetype_id":     aid,
            "domain":           low.get("domain"),
            "consensus_level":  low.get("consensus_level"),
            "post_cutoff":      low.get("post_cutoff", False),
            "prompt_family":    fam,
            "ses_dimension":    dim,
            "model":            mk,
            # Human baseline
            "human_risky_ratio":    human_rr,
            # LLM choice rates
            "llm_risky_rate_low":   llm_low,
            "llm_risky_rate_high":  llm_high,
            # Primary gaps (the research findings)
            "ses_gap":              ses_gap,           # RQ1
            "alignment_gap_low":    align_gap_low,     # RQ2/RQ3
            "alignment_gap_high":   align_gap_high,    # RQ2/RQ3
            "h3_signal":            h3_signal,         # H3: positive = predicted direction
            # Metadata
            "risky_summary":    low.get("risky_summary"),
            "safe_summary":     low.get("safe_summary"),
            "n_valid_low":      low.get("n_valid"),
            "n_valid_high":     high.get("n_valid"),
        })

    # Sort for readability
    comparison_rows.sort(key=lambda r: (
        r.get("prompt_family", ""),
        r.get("domain", ""),
        r.get("consensus_level", ""),
        r.get("archetype_id", ""),
        r.get("model", ""),
        r.get("ses_dimension", ""),
    ))

    if not args.dry_run and comparison_rows:
        # choice_rates.jsonl
        rates_path = config.EXPERIMENT_DIR / "choice_rates.jsonl"
        with rates_path.open("w", encoding="utf-8") as f:
            for r in comparison_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # comparison_table.csv — the main analysis table
        csv_cols = [
            "archetype_id", "domain", "consensus_level", "post_cutoff",
            "prompt_family", "ses_dimension", "model",
            "human_risky_ratio",
            "llm_risky_rate_low", "llm_risky_rate_high",
            "ses_gap", "alignment_gap_low", "alignment_gap_high", "h3_signal",
            "n_valid_low", "n_valid_high",
            "risky_summary", "safe_summary",
        ]
        csv_path = config.EXPERIMENT_DIR / "comparison_table.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
            w.writeheader()
            for r in comparison_rows:
                w.writerow({k: (f"{r[k]:.4f}" if isinstance(r.get(k), float) else r.get(k, ""))
                            for k in csv_cols})

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    log.info("=" * 60)
    log.info("STEP 6 COMPLETE in %.1fs", elapsed)
    log.info("Variants processed this run : %d", total_variants)
    log.info("Total API errors            : %d", errors_total)
    log.info("Comparison table rows       : %d", len(comparison_rows))

    if comparison_rows:
        # H3 preview — broken out by prompt family to compare CM vs CV
        for fam in config.PROMPT_FAMILIES:
            fam_rows = [r for r in comparison_rows if r.get("prompt_family") == fam]
            if not fam_rows:
                continue
            h3_vals = [r["h3_signal"] for r in fam_rows if r.get("h3_signal") is not None]
            ses_gaps = [r["ses_gap"] for r in fam_rows if r.get("ses_gap") is not None]
            log.info("")
            log.info("[%s]  %d rows", fam.upper(), len(fam_rows))
            if h3_vals:
                mean_h3 = sum(h3_vals) / len(h3_vals)
                pct_pos = sum(1 for v in h3_vals if v > 0) / len(h3_vals) * 100
                log.info("  H3 preview  mean=%.4f  pct_positive=%.1f%%", mean_h3, pct_pos)
            if ses_gaps:
                log.info("  RQ1 preview ses_gap mean=%.4f", sum(ses_gaps) / len(ses_gaps))

    log.info("")
    log.info("Output files:")
    log.info("  raw_responses.jsonl   — all individual LLM calls")
    log.info("  choice_rates.jsonl    — aggregated per variant")
    log.info("  comparison_table.csv  — MAIN ANALYSIS TABLE")
    log.info("  justifications.jsonl  — justification-condition responses (CM only)")


if __name__ == "__main__":
    main()
