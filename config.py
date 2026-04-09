"""
Central configuration for the SES-LLM-bias pipeline.
All tunable knobs live here so the individual step scripts stay declarative.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, which is the directory of this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR              = PROJECT_ROOT / "data"
RAW_DIR               = DATA_DIR / "raw"
PUSHSHIFT_DIR         = RAW_DIR / "pushshift"
POSTS_FILTERED_DIR    = RAW_DIR / "posts_filtered"
POSTS_SCORED_DIR      = RAW_DIR / "posts_scored"
ARCHETYPES_DIR        = RAW_DIR / "archetypes"
SCENARIOS_DIR         = RAW_DIR / "scenarios"
LOGS_DIR              = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Subreddit selection by life-decision domain
# ---------------------------------------------------------------------------
SUBREDDITS = {
    "education": [
        "college", "ApplyingToCollege", "StudentLoans",
        "GradSchool", "TransferStudents",
    ],
    "career": [
        "careeradvice", "careerguidance", "jobs",
        "cscareerquestions", "findapath",
    ],
    "finance": [
        "personalfinance", "FinancialPlanning",
        "investing", "povertyfinance",
    ],
    "health": [
        "AskDocs", "HealthInsurance",
        "ChronicPain", "medical_advice",
    ],
    "social": [
        "LifeAdvice", "Advice",
        "relationship_advice", "AskReddit",
    ],
}

# Convenience: lowercase subreddit -> domain
SUBREDDIT_TO_DOMAIN = {
    sub.lower(): domain
    for domain, subs in SUBREDDITS.items()
    for sub in subs
}

# Short prefixes used when minting archetype IDs in step 3
DOMAIN_PREFIX = {
    "education": "EDU",
    "career":    "CAR",
    "finance":   "FIN",
    "health":    "HLT",
    "social":    "SOC",
}

# ---------------------------------------------------------------------------
# Filtering rules
# ---------------------------------------------------------------------------
TRIGGER_PHRASES = [
    "should i", "which should", "deciding between",
    "torn between", "help me decide", "what would you do",
    "not sure if i should", "take the job", "accept the offer",
    "go to", "stay or", "move or", "option a", "option b",
    "thinking about whether", "risk",
]

MIN_WORD_COUNT = 80
MIN_COMMENTS   = 15

# ---------------------------------------------------------------------------
# Ambiguity test thresholds (used by step 2 and step 5)
# ---------------------------------------------------------------------------
AMBIGUITY_LOW    = 0.35
AMBIGUITY_HIGH   = 0.65
AMBIGUITY_N      = 20
MAX_ADJUSTMENTS  = 3

# Per-domain target archetype count (step 3 oversamples by 3x)
TARGETS_PER_DOMAIN = 10

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
GEMINI_MODEL    = "google/gemini-2.0-flash-lite-001"  # OpenRouter path (no direct Gemini SDK)
HOLDOUT_MODEL   = "meta-llama/llama-3-70b-instruct"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Step 2b — SES sensitivity filter
# ---------------------------------------------------------------------------
SES_CUE_LEVELS         = ["A0", "A1", "A2"]
SES_SENSITIVITY_LEVELS = ["B0", "B1", "B2"]
SES_REQUIRED_SENSITIVITY = "B2"
SES_ALLOWED_CUE_LEVELS   = ["A1", "A2"]
SES_CHANNELS = [
    "financial", "education", "occupation",
    "healthcare", "time", "social_support",
]
