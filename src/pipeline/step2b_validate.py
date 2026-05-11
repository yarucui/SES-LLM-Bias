"""
Step 2b -- Validate Step 2 output and export manual-coding samples.

Three things, no LLM calls:

  1. VALIDATE all_scored_ok.jsonl against the Step 5 input contract.
     Every record must carry the full set of Step 2 fields with values
     in the allowed sets (option_risky/option_safe non-empty,
     risky_ratio in [0,1], consensus_level / reversibility / time_horizon
     / resource_constraint / trade_off_type / ses_level in their
     enumerations, n_scored >= 10). Records failing any check are
     flagged as invalid_annotation and excluded.

  2. WRITE all_scored_valid.jsonl -- every record that passes
     validation. This is what Step 5 reads. No further filtering
     (no SES sensitivity filter, no consensus filter).

  3. EXPORT two manual-coding CSVs for Krippendorff alpha measurement
     of Gemini reliability:
       - option_mapping_validation.csv: 150 posts stratified across
         (domain x consensus_level) cells, floor of 5 per cell when
         possible, seed=42.
       - stance_validation.csv: up to 5 stance-classified comments
         per sampled post, comment bodies re-fetched from Reddit
         (Step 2 does not persist them).

If either CSV already has human-filled cells on a subsequent run,
Krippendorff alpha (nominal, two coders) is computed and compared
against the 0.70 threshold.

Usage:
    python src/pipeline/step2b_validate.py
    python src/pipeline/step2b_validate.py --dry-run
    python src/pipeline/step2b_validate.py --alphas-only

Errors in the upstream Step 2 annotation should be fixed via
    python src/pipeline/step2_score.py --rerun-errors
not in this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "step2b_validate.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("step2b_validate")

# ---------------------------------------------------------------------------
# Validation schema
# ---------------------------------------------------------------------------
VALID_CONSENSUS     = {"high_safe", "ambiguous", "high_risky"}
VALID_REVERSIBILITY = {"reversible", "partially_reversible", "irreversible"}
VALID_TIME_HORIZON  = {"short", "long"}
VALID_RESOURCE      = {"financial", "time", "social", "health", "geographic"}
VALID_TRADEOFF      = {
    "prestige_vs_cost", "risk_vs_stability", "short_term_vs_long_term",
    "mobility_vs_rootedness", "aggressive_vs_conservative",
}
VALID_SES_LEVEL     = {"low", "mid", "high"}
MIN_N_SCORED        = 10

SAMPLE_SIZE            = 150
MIN_PER_CELL           = 5
RANDOM_SEED            = 42
COMMENTS_PER_POST      = 5
KRIPPENDORFF_THRESHOLD = 0.70

OPTION_CSV_NAME = "option_mapping_validation.csv"
STANCE_CSV_NAME = "stance_validation.csv"
REPORT_JSON     = "validation_report.json"
OUTPUT_JSONL    = "all_scored_valid.jsonl"

# ---------------------------------------------------------------------------
# Reddit fetching
# ---------------------------------------------------------------------------
# Step 2 does not persist comment bodies, so we must re-fetch top-level
# comments at validation time. This is a re-fetch, not a re-score: no LLM
# calls, and only for the 150-post sample. Re-fetch mirrors step2_score's
# fetching logic so comment indices still line up with what Gemini saw --
# subject to Reddit-side drift (new comments, deletions) since scoring.
REDDIT_HEADERS = {"User-Agent": "ses_bias_research/1.0"}
REDDIT_SLEEP   = 2.0


def fetch_top_comments(
    subreddit: str, post_id: str, max_retries: int = 3
) -> tuple[list[dict] | None, str | None]:
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
            time.sleep(60 * (attempt + 1))
        elif resp.status_code == 401:
            time.sleep(30 * (attempt + 1))
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
# Validation
# ---------------------------------------------------------------------------

def validate_record(rec: dict) -> tuple[bool, list[str]]:
    """Return (is_valid, list_of_reasons).

    Reasons are the field names that failed, so the report can break
    down which fields Step 2 is missing or mis-annotating most often.
    """
    reasons: list[str] = []

    opt_r = rec.get("option_risky")
    if not (isinstance(opt_r, str) and opt_r.strip()):
        reasons.append("option_risky")

    opt_s = rec.get("option_safe")
    if not (isinstance(opt_s, str) and opt_s.strip()):
        reasons.append("option_safe")

    rr = rec.get("risky_ratio")
    if not (isinstance(rr, (int, float))) or isinstance(rr, bool) \
            or not (0.0 <= float(rr) <= 1.0):
        reasons.append("risky_ratio")

    if rec.get("consensus_level") not in VALID_CONSENSUS:
        reasons.append("consensus_level")

    ns = rec.get("n_scored")
    if not (isinstance(ns, int) and not isinstance(ns, bool) and ns >= MIN_N_SCORED):
        reasons.append("n_scored")

    if rec.get("reversibility") not in VALID_REVERSIBILITY:
        reasons.append("reversibility")

    if rec.get("time_horizon") not in VALID_TIME_HORIZON:
        reasons.append("time_horizon")

    if rec.get("resource_constraint") not in VALID_RESOURCE:
        reasons.append("resource_constraint")

    if rec.get("trade_off_type") not in VALID_TRADEOFF:
        reasons.append("trade_off_type")

    if rec.get("ses_level") not in VALID_SES_LEVEL:
        reasons.append("ses_level")

    return (len(reasons) == 0), reasons


# ---------------------------------------------------------------------------
# Krippendorff's alpha (nominal, two coders)
# ---------------------------------------------------------------------------
# Lifted verbatim from the previous step2b implementation. Kept local so
# the new file has no dependency on the file it replaces.

def krippendorff_alpha_nominal(pairs: list[tuple[str, str]]) -> float | None:
    clean = [(a, b) for a, b in pairs if a and b]
    n = len(clean)
    if n < 2:
        return None
    D_o = sum(1 for a, b in clean if a != b) / n
    counts: Counter = Counter()
    for a, b in clean:
        counts[a] += 1
        counts[b] += 1
    total = sum(counts.values())
    D_e = 1.0 - sum(c / total * (c - 1) / (total - 1) for c in counts.values())
    if D_e == 0:
        return 1.0 if D_o == 0 else None
    alpha = 1.0 - D_o / D_e
    return None if (math.isnan(alpha) or math.isinf(alpha)) else alpha


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    valid_posts: list[dict],
    target: int = SAMPLE_SIZE,
    min_per_cell: int = MIN_PER_CELL,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """Stratified sample across (domain x consensus_level) cells.

    Allocation is proportional to cell size with a floor of min_per_cell
    when the cell has at least that many posts. If the summed allocation
    exceeds target (possible when many small cells hit the floor),
    randomly down-sample the union to hit target exactly.
    """
    rng = random.Random(seed)
    cells: dict[tuple, list] = defaultdict(list)
    for p in valid_posts:
        cells[(p.get("domain"), p.get("consensus_level"))].append(p)

    total = sum(len(v) for v in cells.values())
    if total == 0:
        return []

    sample: list[dict] = []
    for items in cells.values():
        prop = round(target * len(items) / total)
        n = max(min(min_per_cell, len(items)), prop)
        n = min(n, len(items))
        sample.extend(rng.sample(items, n))

    if len(sample) > target:
        sample = rng.sample(sample, target)
    return sample


# ---------------------------------------------------------------------------
# CSV schemas
# ---------------------------------------------------------------------------
OPTION_COLS = [
    "post_id", "domain", "consensus_level", "ses_level",
    "title", "selftext_excerpt",
    "gemini_option_risky", "gemini_option_safe",
    "human_options_agree", "human_option_risky", "human_option_safe",
    "human_notes",
]

STANCE_COLS = [
    "post_id", "title", "option_risky", "option_safe",
    "comment_index", "comment_text",
    "gemini_stance", "gemini_confidence",
    "human_stance", "human_confidence", "human_notes",
]


def _read_csv_rows(path: Path) -> list[dict] | None:
    """Read a CSV that may have been re-saved by Excel in the Windows
    locale. Try UTF-8 (with optional BOM) first, then common Windows
    fallbacks. Returns None if every encoding fails."""
    for enc in ("utf-8-sig", "cp1252", "gbk", "latin-1"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
        except Exception as e:
            log.warning("Could not parse %s (%s): %s", path.name, enc, e)
            return None
    log.warning("Could not decode %s in any known encoding", path.name)
    return None


def load_existing_option_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = _read_csv_rows(path)
    if rows is None:
        return {}
    out: dict[str, dict] = {}
    for row in rows:
        pid   = (row.get("post_id") or "").strip()
        agree = (row.get("human_options_agree") or "").strip()
        if pid and agree:
            out[pid] = {
                "human_options_agree": agree,
                "human_option_risky":  (row.get("human_option_risky") or "").strip(),
                "human_option_safe":   (row.get("human_option_safe") or "").strip(),
                "human_notes":         (row.get("human_notes") or "").strip(),
            }
    return out


def load_existing_stance_labels(path: Path) -> dict[tuple[str, int], dict]:
    if not path.exists():
        return {}
    rows = _read_csv_rows(path)
    if rows is None:
        return {}
    out: dict[tuple[str, int], dict] = {}
    for row in rows:
        pid   = (row.get("post_id") or "").strip()
        idx_s = (row.get("comment_index") or "").strip()
        hs    = (row.get("human_stance") or "").strip()
        hc    = (row.get("human_confidence") or "").strip()
        if not pid or not idx_s or not (hs or hc):
            continue
        try:
            idx = int(idx_s)
        except ValueError:
            continue
        out[(pid, idx)] = {
            "human_stance":     hs,
            "human_confidence": hc,
            "human_notes":      (row.get("human_notes") or "").strip(),
        }
    return out


def write_option_csv(
    sample: list[dict], path: Path, existing: dict[str, dict]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OPTION_COLS)
        w.writeheader()
        for r in sample:
            pid     = r["post_id"]
            prior   = existing.get(pid, {})
            excerpt = (r.get("selftext") or "")[:800].replace("\n", " ")
            w.writerow({
                "post_id":             pid,
                "domain":              r.get("domain", ""),
                "consensus_level":     r.get("consensus_level", ""),
                "ses_level":           r.get("ses_level", ""),
                "title":               (r.get("title") or "")[:200],
                "selftext_excerpt":    excerpt,
                "gemini_option_risky": r.get("option_risky", ""),
                "gemini_option_safe":  r.get("option_safe", ""),
                "human_options_agree": prior.get("human_options_agree", ""),
                "human_option_risky":  prior.get("human_option_risky", ""),
                "human_option_safe":   prior.get("human_option_safe", ""),
                "human_notes":         prior.get("human_notes", ""),
            })


def write_stance_csv(
    rows: list[dict], path: Path, existing: dict[tuple[str, int], dict]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STANCE_COLS)
        w.writeheader()
        for row in rows:
            pid   = row["post_id"]
            idx   = row["comment_index"]
            prior = existing.get((pid, idx), {})
            w.writerow({
                "post_id":           pid,
                "title":             row.get("title", ""),
                "option_risky":      row.get("option_risky", ""),
                "option_safe":       row.get("option_safe", ""),
                "comment_index":     idx,
                "comment_text":      row["comment_text"],
                "gemini_stance":     row["gemini_stance"],
                "gemini_confidence": row["gemini_confidence"],
                "human_stance":      prior.get("human_stance", ""),
                "human_confidence":  prior.get("human_confidence", ""),
                "human_notes":       prior.get("human_notes", ""),
            })


# ---------------------------------------------------------------------------
# Stance-row construction (re-fetches comments for each sampled post)
# ---------------------------------------------------------------------------

def build_stance_rows(sample: list[dict], rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    for i, post in enumerate(sample):
        pid            = post["post_id"]
        subreddit      = post.get("subreddit", "")
        classifications = post.get("comment_classifications") or []
        if not classifications:
            log.warning("  [%d/%d] %s: no comment_classifications -- skipped",
                        i + 1, len(sample), pid)
            continue

        pick = (classifications
                if len(classifications) <= COMMENTS_PER_POST
                else rng.sample(classifications, COMMENTS_PER_POST))

        comments, err = fetch_top_comments(subreddit, pid)
        time.sleep(REDDIT_SLEEP)
        if err or not comments:
            log.warning("  [%d/%d] %s: Reddit fetch failed (%s) -- "
                        "writing rows with empty comment_text",
                        i + 1, len(sample), pid, err or "no comments")
            comments = []

        for c in pick:
            idx = c.get("index")
            if not isinstance(idx, int):
                continue
            text = comments[idx]["body"] if 0 <= idx < len(comments) else ""
            rows.append({
                "post_id":           pid,
                "title":             (post.get("title") or "")[:200],
                "option_risky":      post.get("option_risky", ""),
                "option_safe":       post.get("option_safe", ""),
                "comment_index":     idx,
                "comment_text":      text[:500].replace("\n", " "),
                "gemini_stance":     c.get("stance", ""),
                "gemini_confidence": c.get("confidence", ""),
            })

        if (i + 1) % 25 == 0:
            log.info("  Stance rows: %d / %d posts fetched", i + 1, len(sample))

    return rows


# ---------------------------------------------------------------------------
# Alpha computation from existing human labels
# ---------------------------------------------------------------------------

def compute_alphas(
    sample: list[dict],
    existing_option: dict[str, dict],
    existing_stance: dict[tuple[str, int], dict],
) -> tuple[float | None, float | None, float | None]:
    """Return (option_alpha, stance_alpha, confidence_alpha).

    Option alpha: Gemini's option output is implicitly "the options
    match the post" (= "1"), and the human coder either confirms ("1")
    or disputes ("0"). This makes alpha degenerate when Gemini is
    always a single class; report it as requested and note the caveat
    in the report comments section for the final paper.

    Stance / confidence alpha: straight nominal agreement between
    Gemini's label and the human's label for each coded comment.
    """
    alpha_option = None
    if existing_option:
        pairs_opt = [("1", h["human_options_agree"])
                     for h in existing_option.values()
                     if h.get("human_options_agree")]
        alpha_option = krippendorff_alpha_nominal(pairs_opt)

    alpha_stance = alpha_confidence = None
    if existing_stance:
        by_key: dict[tuple[str, int], dict] = {}
        for p in sample:
            pid = p["post_id"]
            for c in (p.get("comment_classifications") or []):
                idx = c.get("index")
                if isinstance(idx, int):
                    by_key[(pid, idx)] = c

        def _norm(x: object) -> str:
            return str(x or "").strip().lower()

        pairs_stance: list[tuple[str, str]] = []
        pairs_conf:   list[tuple[str, str]] = []
        for key, h in existing_stance.items():
            g = by_key.get(key)
            if not g:
                continue
            gs, hs = _norm(g.get("stance")),     _norm(h.get("human_stance"))
            gc, hc = _norm(g.get("confidence")), _norm(h.get("human_confidence"))
            if gs and hs:
                pairs_stance.append((gs, hs))
            if gc and hc:
                pairs_conf.append((gc, hc))

        alpha_stance     = krippendorff_alpha_nominal(pairs_stance)
        alpha_confidence = krippendorff_alpha_nominal(pairs_conf)

    return alpha_option, alpha_stance, alpha_confidence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2b: validate Step 2 output and export "
                    "manual-coding CSVs for Krippendorff alpha."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and select sample; do not write "
                             "files or fetch comments from Reddit.")
    parser.add_argument("--alphas-only", action="store_true",
                        help="Skip Reddit re-fetch and CSV/JSONL rewrite; "
                             "load existing human labels and compute "
                             "Krippendorff alpha only.")
    args = parser.parse_args()

    if args.dry_run and args.alphas_only:
        parser.error("--dry-run and --alphas-only are mutually exclusive")

    config.POSTS_SCORED_DIR.mkdir(parents=True, exist_ok=True)

    src = config.POSTS_SCORED_DIR / "all_scored_ok.jsonl"
    if not src.exists():
        log.error("Missing %s -- run step2_score.py first", src)
        sys.exit(1)

    all_posts: list[dict] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_posts.append(json.loads(line))
    log.info("Loaded %d scored posts", len(all_posts))

    # -- Validate -----------------------------------------------------------
    valid_posts: list[dict] = []
    invalid_breakdown: Counter = Counter()
    invalid_count = 0
    for p in all_posts:
        ok, reasons = validate_record(p)
        if ok:
            valid_posts.append(p)
        else:
            invalid_count += 1
            for r in reasons:
                invalid_breakdown[r] += 1

    log.info("Valid annotations : %d / %d", len(valid_posts), len(all_posts))
    if invalid_count:
        log.warning(
            "%d posts failed validation. Top reasons: %s. "
            "Fix with: python step2_score.py --rerun-errors",
            invalid_count,
            ", ".join(f"{k}={v}" for k, v in invalid_breakdown.most_common(5)),
        )

    # -- Write all_scored_valid.jsonl --------------------------------------
    out_path = config.POSTS_SCORED_DIR / OUTPUT_JSONL
    if not args.dry_run and not args.alphas_only:
        with out_path.open("w", encoding="utf-8") as f:
            for r in valid_posts:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("Wrote %d posts -> %s", len(valid_posts), out_path.name)

    # -- Breakdown counts ---------------------------------------------------
    domain_counts    = Counter(p.get("domain") for p in valid_posts)
    consensus_counts = Counter(p.get("consensus_level") for p in valid_posts)
    ses_level_counts = Counter(p.get("ses_level") for p in valid_posts)
    log.info("")
    log.info("By domain         : %s", dict(domain_counts))
    log.info("By consensus_level: %s", dict(consensus_counts))
    log.info("By ses_level      : %s", dict(ses_level_counts))

    # -- Stratified sample --------------------------------------------------
    sample = stratified_sample(valid_posts)
    log.info("")
    log.info("Stratified sample : %d posts (target %d)",
             len(sample), SAMPLE_SIZE)
    sample_cells = Counter(
        (p.get("domain"), p.get("consensus_level")) for p in sample
    )
    for key, n in sorted(sample_cells.items(), key=lambda x: str(x[0])):
        log.info("  %-40s : %d", str(key), n)

    # -- Build stance rows (requires Reddit fetch for comment text) --------
    option_csv_path = config.POSTS_SCORED_DIR / OPTION_CSV_NAME
    stance_csv_path = config.POSTS_SCORED_DIR / STANCE_CSV_NAME

    stance_rows: list[dict] = []
    if not args.dry_run and not args.alphas_only and sample:
        log.info("")
        log.info("Re-fetching comments for %d sampled posts "
                 "(approx %.1f min at %.1fs per post)...",
                 len(sample), len(sample) * REDDIT_SLEEP / 60.0, REDDIT_SLEEP)
        rng = random.Random(RANDOM_SEED)
        stance_rows = build_stance_rows(sample, rng)
        log.info("Built %d stance rows", len(stance_rows))

    # -- Existing human labels ---------------------------------------------
    existing_option = load_existing_option_labels(option_csv_path)
    existing_stance = load_existing_stance_labels(stance_csv_path)

    # -- Write CSVs ---------------------------------------------------------
    if not args.dry_run and not args.alphas_only:
        write_option_csv(sample, option_csv_path, existing_option)
        log.info("Wrote %s (%d rows)", option_csv_path.name, len(sample))
        write_stance_csv(stance_rows, stance_csv_path, existing_stance)
        log.info("Wrote %s (%d rows)", stance_csv_path.name, len(stance_rows))

    # -- Krippendorff alpha -------------------------------------------------
    alpha_option, alpha_stance, alpha_confidence = compute_alphas(
        sample, existing_option, existing_stance
    )

    for name, a, threshold_applies in (
        ("option",     alpha_option,     True),
        ("stance",     alpha_stance,     True),
        ("confidence", alpha_confidence, False),
    ):
        if a is None:
            status = "n/a (no human labels)"
        elif not threshold_applies:
            status = "exploratory"
        elif a >= KRIPPENDORFF_THRESHOLD:
            status = "OK"
        else:
            status = "BELOW THRESHOLD"
        log.info("Krippendorff alpha (%s): %s  [%s]",
                 name, f"{a:.3f}" if a is not None else "n/a", status)

    # Diagnostic: if stance alpha is suspiciously low, dump the first
    # mismatched pairs with repr() so hidden formatting differences
    # (whitespace, column shifts from mangled CSV quoting, etc.) are
    # visible.
    if alpha_stance is not None and alpha_stance < KRIPPENDORFF_THRESHOLD:
        by_key_diag: dict[tuple[str, int], dict] = {}
        for p in sample:
            for c in (p.get("comment_classifications") or []):
                idx = c.get("index")
                if isinstance(idx, int):
                    by_key_diag[(p["post_id"], idx)] = c

        mismatches: list[tuple[tuple[str, int], str, str]] = []
        label_counts_g: Counter = Counter()
        label_counts_h: Counter = Counter()
        for key, h in existing_stance.items():
            g = by_key_diag.get(key)
            if not g:
                continue
            gs_raw = g.get("stance") or ""
            hs_raw = h.get("human_stance") or ""
            label_counts_g[gs_raw] += 1
            label_counts_h[hs_raw] += 1
            if str(gs_raw).strip().lower() != str(hs_raw).strip().lower():
                mismatches.append((key, gs_raw, hs_raw))

        log.warning("Stance alpha diagnostic:")
        log.warning("  Gemini label distribution: %s",
                    dict(label_counts_g.most_common()))
        log.warning("  Human  label distribution: %s",
                    dict(label_counts_h.most_common()))
        log.warning("  %d mismatched pairs (showing first 10 with repr):",
                    len(mismatches))
        for key, gs, hs in mismatches[:10]:
            log.warning("    %s  gemini=%r  human=%r", key, gs, hs)

    below_threshold = any(
        a is not None and a < KRIPPENDORFF_THRESHOLD
        for a in (alpha_option, alpha_stance)
    )
    if below_threshold:
        log.warning(
            "Below threshold. Revise Step 2 prompt and re-score before "
            "proceeding to Step 5. Preregistration commitment."
        )

    # -- Report -------------------------------------------------------------
    report = {
        "total_input":                   len(all_posts),
        "valid":                         len(valid_posts),
        "invalid":                       invalid_count,
        "invalid_breakdown_by_reason":   dict(invalid_breakdown),
        "domain_counts":                 dict(domain_counts),
        "consensus_counts":              dict(consensus_counts),
        "ses_level_counts":              dict(ses_level_counts),
        "option_validation_sample_size": len(sample),
        "stance_validation_sample_size": len(stance_rows),
        "option_alpha":                  alpha_option,
        "stance_alpha":                  alpha_stance,
        "confidence_alpha":              alpha_confidence,
    }
    if not args.dry_run:
        with (config.POSTS_SCORED_DIR / REPORT_JSON).open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(report, f, indent=2)

    mode_suffix = (" (dry-run)" if args.dry_run
                   else " (alphas-only)" if args.alphas_only else "")
    log.info("")
    log.info("=" * 60)
    log.info("STEP 2b COMPLETE%s", mode_suffix)
    log.info("  valid posts              : %d", len(valid_posts))
    log.info("  invalid posts            : %d", invalid_count)
    log.info("  option validation sample : %d", len(sample))
    log.info("  stance validation sample : %d", len(stance_rows))
    log.info("")
    if args.dry_run:
        file_note = "dry-run -- not written"
    elif args.alphas_only:
        file_note = "alphas-only -- only validation_report.json rewritten"
    else:
        file_note = "written"
    log.info("Output files (%s):", file_note)
    log.info("  %s", OUTPUT_JSONL)
    log.info("  %s", OPTION_CSV_NAME)
    log.info("  %s", STANCE_CSV_NAME)
    log.info("  %s", REPORT_JSON)


if __name__ == "__main__":
    main()
