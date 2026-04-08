"""
Step 1 — Parse Pushshift .zst submission dumps and apply the inclusion filter.

Reads every .zst file under data/raw/pushshift/ (recursively, so existing
sub-folders like raw/pushshift/reddit/submissions/ also work), streams JSON
lines without ever loading a whole file into memory, and writes one .jsonl
per domain plus a summary.json.

Usage:
    python src/pipeline/step1_parse.py
    python src/pipeline/step1_parse.py --dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import zstandard as zstd
from dotenv import load_dotenv

# Make `import config` work no matter where the script is launched from
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "step1_parse.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("step1_parse")

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Filter logic
# ---------------------------------------------------------------------------
TRIGGERS = [t.lower() for t in config.TRIGGER_PHRASES]


def passes_filter(post: dict) -> tuple[bool, str | None]:
    """Return (True, domain) if the submission should be kept, else (False, None)."""
    sub = post.get("subreddit")
    if not isinstance(sub, str):
        return False, None
    domain = config.SUBREDDIT_TO_DOMAIN.get(sub.lower())
    if domain is None:
        return False, None

    if not post.get("is_self"):
        return False, None

    selftext = post.get("selftext", "")
    if not isinstance(selftext, str) or selftext in ("", "[removed]", "[deleted]"):
        return False, None

    if len(selftext.split()) < config.MIN_WORD_COUNT:
        return False, None

    if int(post.get("num_comments", 0) or 0) < config.MIN_COMMENTS:
        return False, None

    haystack = (str(post.get("title", "")) + " " + selftext).lower()
    if not any(trigger in haystack for trigger in TRIGGERS):
        return False, None

    return True, domain


def slim(post: dict, domain: str, source_file: str) -> dict:
    """Project the fields we keep on disk."""
    return {
        "post_id":      post.get("id"),
        "subreddit":    post.get("subreddit"),
        "domain":       domain,
        "title":        post.get("title", ""),
        "selftext":     post.get("selftext", ""),
        "score":        post.get("score"),
        "num_comments": post.get("num_comments"),
        "created_utc":  post.get("created_utc"),
        "source_file":  source_file,
    }


# ---------------------------------------------------------------------------
# .zst streaming
# ---------------------------------------------------------------------------
def iter_zst_lines(path: Path):
    """Yield decoded JSON lines from a .zst file without loading it all."""
    with path.open("rb") as raw:
        # Large window for Pushshift dumps
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(raw) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text:
                line = line.strip()
                if line:
                    yield line


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process at most 3 kept posts per domain and don't write outputs.",
    )
    args = parser.parse_args()

    config.POSTS_FILTERED_DIR.mkdir(parents=True, exist_ok=True)

    # Find all .zst files anywhere under data/raw/pushshift/. Also tolerate
    # the existing layout at <project>/raw/pushshift/.
    search_roots = [config.PUSHSHIFT_DIR, PROJECT_ROOT / "raw" / "pushshift"]
    zst_files: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        if root.exists():
            for p in sorted(root.rglob("*.zst")):
                if p not in seen:
                    zst_files.append(p)
                    seen.add(p)

    if not zst_files:
        log.error("No .zst files found under %s", config.PUSHSHIFT_DIR)
        sys.exit(1)

    log.info("Found %d .zst file(s):", len(zst_files))
    for p in zst_files:
        log.info("  - %s", p)

    # Per-domain output handles
    out_handles: dict[str, "io.TextIOBase"] = {}
    if not args.dry_run:
        for domain in config.SUBREDDITS:
            fp = config.POSTS_FILTERED_DIR / f"{domain}.jsonl"
            out_handles[domain] = fp.open("w", encoding="utf-8")

    summary: dict[str, dict[str, int]] = {d: defaultdict(int) for d in config.SUBREDDITS}
    dry_kept: dict[str, list[dict]] = defaultdict(list)
    total_lines = 0
    total_kept = 0
    start = time.time()

    # Graceful Ctrl-C
    interrupted = {"flag": False}

    def _sigint(_signum, _frame):
        interrupted["flag"] = True
        log.warning("Interrupt received; will stop after current line.")

    signal.signal(signal.SIGINT, _sigint)

    try:
        for zst_path in zst_files:
            log.info("Streaming %s", zst_path.name)
            file_lines = 0
            file_kept = 0
            for line in iter_zst_lines(zst_path):
                total_lines += 1
                file_lines += 1
                if total_lines % 500_000 == 0:
                    elapsed = time.time() - start
                    log.info(
                        "  %d lines read (kept %d so far, %.0f lines/s)",
                        total_lines, total_kept, total_lines / max(elapsed, 1e-6),
                    )

                try:
                    post = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ok, domain = passes_filter(post)
                if not ok:
                    continue

                record = slim(post, domain, zst_path.name)

                if args.dry_run:
                    if len(dry_kept[domain]) < 3:
                        dry_kept[domain].append(record)
                    # Stop early if every domain is full
                    if all(len(dry_kept[d]) >= 3 for d in config.SUBREDDITS):
                        log.info("Dry-run quota met; stopping early.")
                        raise StopIteration
                else:
                    out_handles[domain].write(json.dumps(record, ensure_ascii=False) + "\n")

                summary[domain][record["subreddit"]] += 1
                total_kept += 1
                file_kept += 1

                if interrupted["flag"]:
                    raise KeyboardInterrupt

            log.info("  %s: %d lines, %d kept", zst_path.name, file_lines, file_kept)

    except StopIteration:
        pass
    except KeyboardInterrupt:
        log.warning("Stopped early by user.")
    finally:
        for fh in out_handles.values():
            fh.close()

    # Write summary
    summary_serialisable = {d: dict(counts) for d, counts in summary.items()}
    if not args.dry_run:
        with (config.POSTS_FILTERED_DIR / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary_serialisable, f, indent=2)

    elapsed = time.time() - start
    log.info("=" * 60)
    log.info("STEP 1 COMPLETE")
    log.info("Lines read : %d", total_lines)
    log.info("Total kept : %d", total_kept)
    for domain in config.SUBREDDITS:
        kept = sum(summary[domain].values())
        log.info("  %-10s %d", domain, kept)
        for sub, n in sorted(summary[domain].items(), key=lambda x: -x[1]):
            log.info("      %-25s %d", sub, n)
    log.info("Elapsed    : %.1fs", elapsed)

    if args.dry_run:
        log.info("--- DRY RUN SAMPLE ---")
        for domain, records in dry_kept.items():
            log.info("[%s] %d sample(s)", domain, len(records))
            for r in records:
                log.info("    %s | %s", r["post_id"], r["title"][:80])


if __name__ == "__main__":
    main()
