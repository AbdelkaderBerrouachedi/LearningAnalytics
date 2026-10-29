"""
Configuration module for the OULAD Student Withdrawal Early Warning Pipeline.
Defines paths, prediction cutoffs, feature definitions, and model hyperparameters.
"""
from pathlib import Path

# Base Paths (resolve relative to Code_Git)
CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent
DATA_DIR = CODE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Distinct output directories to protect submission pack from being overwritten by demo runs
OUTPUTS_DIR = CODE_DIR / "outputs"
DEMO_OUTPUTS_DIR = OUTPUTS_DIR / "demo"
REAL_OUTPUTS_DIR = OUTPUTS_DIR / "real"

DEMO_TABLES_DIR = DEMO_OUTPUTS_DIR / "tables"
DEMO_FIGURES_DIR = DEMO_OUTPUTS_DIR / "figures"
REAL_TABLES_DIR = REAL_OUTPUTS_DIR / "tables"
REAL_FIGURES_DIR = REAL_OUTPUTS_DIR / "figures"

# Default run directories point to demo outputs
TABLES_DIR = DEMO_TABLES_DIR
FIGURES_DIR = DEMO_FIGURES_DIR

# Protected submission pack directory (only updated via explicit release synchronization)
SUBMISSION_PACK_DIR = PROJECT_ROOT / "Submission_Pack"

# Ensure output directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
DEMO_TABLES_DIR.mkdir(parents=True, exist_ok=True)
DEMO_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REAL_TABLES_DIR.mkdir(parents=True, exist_ok=True)
REAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Prediction Cutoffs (Weeks and elapsed days from presentation start)
CUTOFFS = {
    2: 14,   # Week 2: Day 14
    4: 28,   # Week 4: Day 28
    6: 42,   # Week 6: Day 42
    8: 56,   # Week 8: Day 56
}

# Chronological Presentation Partitions
TRAIN_PRESENTATIONS = ["2013B", "2013J"]
VAL_PRESENTATIONS = ["2014B"]
TEST_PRESENTATIONS = ["2014J"]

# Random Seeds
RANDOM_SEED = 42
N_BOOTSTRAP = 1000

# Operational Advisor Contact Budgets (Percentages of enrolled cohort)
CONTACT_BUDGETS = [0.05, 0.10, 0.20]

# Primary Feature List (Strictly observable <= cutoff t, 36 features)
PRIMARY_FEATURES = [
    # Study Background
    "studied_credits",
    "num_of_prev_attempts",
    "date_registration_imputed",
    # Demographics (Audited for fairness)
    "gender_binary",
    "disability_binary",
    "age_band_ord",
    "education_ord",
    "imd_band_ord",
    # Cumulative Engagement
    "total_clicks",
    "active_days",
    "clicks_per_active_day",
    "distinct_sites",
    "pre_course_clicks",
    "pre_course_active_days",
    # Trailing Engagement & Momentum Dynamics
    "clicks_last_7d",
    "active_days_last_7d",
    "clicks_last_14d",
    "active_days_last_14d",
    "days_since_last_click",
    "click_ratio_7d_to_14d",
    "active_day_ratio_7d_to_14d",
    "recent_to_total_click_share",
    # VLE Resource Category Interactions
    "clicks_forumng",
    "clicks_oucontent",
    "clicks_resource",
    "clicks_subpage",
    "clicks_url",
    "clicks_quiz",
    "clicks_homepage",
    "clicks_other",
    # Assessment Submission Behaviors
    "assessments_due",
    "assessments_submitted",
    "missed_assessments",
    "submission_rate",
    "mean_submission_delay",
    "late_submissions_count",
]

# Ablation Feature Subsets
FEATURE_SUBSETS = {
    "Full Primary": PRIMARY_FEATURES,
    "Without Demographics": [
        f for f in PRIMARY_FEATURES if f not in [
            "gender_binary", "disability_binary", "age_band_ord", "education_ord", "imd_band_ord"
        ]
    ],
    "Demographics Only": [
        "gender_binary", "disability_binary", "age_band_ord", "education_ord", "imd_band_ord",
        "studied_credits", "num_of_prev_attempts", "date_registration_imputed"
    ],
    "VLE Engagement Only": [
        f for f in PRIMARY_FEATURES if "clicks" in f or "active_day" in f or "sites" in f
    ],
    "Assessment Participation Only": [
        "assessments_due", "assessments_submitted", "missed_assessments",
        "submission_rate", "mean_submission_delay", "late_submissions_count"
    ],
    "Full Primary + Assessment Scores": PRIMARY_FEATURES + ["mean_assessment_score"]
}

# Module Domains
MODULE_DOMAINS = {
    "AAA": "Social Science/Humanities",
    "BBB": "Social Sciences",
    "CCC": "STEM",
    "DDD": "STEM",
    "EEE": "STEM",
    "FFF": "STEM",
    "GGG": "Social Sciences",
}
