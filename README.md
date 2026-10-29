# Point-in-Time Learning Analytics Student Withdrawal Early Warning Framework (Paper P6)

**Associated Publication:**  
*How Early Can We Predict Student Withdrawal? Explainable Analytics Across Course Presentations*  

---

## 1. Overview and Key Contributions

This repository contains the complete, reproducible Python codebase for our point-in-time student withdrawal early warning system evaluated on the Open University Learning Analytics Dataset (OULAD).

### Core Methodological Safeguards:
1. **Strict Zero-Leakage Feature Firewall**: Predictor variables are strictly restricted to events timestamped on or before the cutoff day ($t \in \{14, 28, 42, 56\}$). Withdrawal dates define cohort eligibility and the ground-truth target, never entering feature extraction.
2. **Point-in-Time Dynamic Risk Sets**: The denominator is updated dynamically at each cutoff. Students who withdrew on or before day $t$ are permanently purged from the active risk pool.
3. **Strict Chronological Generalization**: Models are trained on historical cohorts (`2013B`, `2013J`), tuned and calibrated on intermediate presentation (`2014B`), and evaluated on held-out future cohorts (`2014J`).
4. **Complete Identity Leakage Control**: Recurring students appearing in earlier training or validation presentations (1,329 students) are systematically eliminated from the held-out test partition.
5. **Validation-Based Probability Calibration**: Platt scaling and Isotonic regression are fitted strictly on validation presentation `2014B`, ensuring probabilistic Brier score validity.
6. **Actionable Advisor Contact Budgets**: Evaluates precision and recall under realistic institutional counseling capacities (top 5%, 10%, 20%).
7. **Theory-Grounded Pedagogical Intervention Engine**: Translates SHAP explanation vectors into a 3-tiered intervention catalog mapped to Self-Regulated Learning (SRL), Academic Momentum, and Cognitive Load Theory (Table 10).

---

## 2. Directory Architecture

```
Code_Git/
├── main.py                            # Master command-line orchestrator
├── requirements.txt                   # Pinned Python package dependencies
├── README.md                          # Repository documentation & reproduction guide
├── audit_oulad.py                     # Historical dataset audit script
├── build_features.py                  # Standalone feature extraction script
├── download_oulad.py                  # Standalone UCI download script
├── run_experiments.py                 # Standalone benchmark experiments runner
├── run_ablations_and_subgroups.py     # Standalone feature ablations & fairness script
├── generate_figures.py                # Standalone figure generation script
├── generate_table1.py                 # Standalone Table 1 comparison script
├── generate_table2.py                 # Standalone Table 2 feature dictionary script
└── src/                               # Modular core library
    ├── __init__.py
    ├── config.py                      # Global paths, cutoffs, features, partitions
    ├── data_loader.py                 # Download, load, audit, synthetic generator
    ├── features.py                    # Point-in-time feature extraction pipeline
    ├── models.py                      # Baselines, Logistic Regression, LightGBM, XGBoost, Calibrators
    ├── evaluation.py                  # PR-AUC, AUROC, Brier, ECE, budgets, bootstrap CIs
    ├── explainability.py              # TreeExplainer SHAP, rank stability (Kendall tau, Spearman rho)
    ├── interventions.py               # 3-Tier pedagogical intervention engine (Table 10)
    └── visualization.py               # Generates 300 DPI publication figures & tables
```

---

## 3. Quickstart & Installation

### Step 1: Environment Setup
We recommend Python 3.10+ in a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Instant 5-Second Verification (Demo Mode)
To verify the entire end-to-end pipeline immediately without waiting to download raw logs:
```bash
python main.py --demo
```
This generates a schema-compliant synthetic cohort, verifies point-in-time filtering, extracts 52 observable features across all four cutoffs, trains and calibrates all five models, conducts bootstrap CI analysis, generates pedagogical intervention tickets, and exports all 300 DPI publication figures!

---

## 4. Full Empirical Reproduction on OULAD

### One-Command Full Pipeline:
```bash
python main.py --all
```

### Or Step-by-Step Execution:

#### 1. Download & Unpack OULAD from UCI:
```bash
python main.py --download
```

#### 2. Audit Cohort Eligibility & Attrition:
```bash
python main.py --audit
```

#### 3. Extract Observable Point-in-Time Features (Weeks 2, 4, 6, 8):
```bash
python main.py --extract-features
```

#### 4. Train Models, Calibrate on Validation Set, and Evaluate on Unseen Test Cohort:
```bash
python main.py --train-eval
```

#### 5. Generate Individualized Pedagogical Triage Tickets:
```bash
python main.py --triage
```

#### 6. Export Publication Figures (300 DPI):
```bash
python main.py --figures
```

---

## 5. Description of Models and Baselines

1. **Prevalence Baseline**: Assigns a constant risk probability equal to training set withdrawal prevalence ($\bar{y}_{train}$).
2. **Activity Dormancy Heuristic**: Ranks learners purely by trailing inactivity (`days_since_last_click` normalized to $[0, 1]$).
3. **Logistic Regression**: Interpretable linear statistical benchmark with standardized features, L2 regularization ($C = 1.0$), and Platt sigmoid calibration.
4. **LightGBM (GBDT)**: Gradient-boosted decision trees (`n_estimators=150`, `learning_rate=0.03`, `num_leaves=31`, `min_child_samples=30`, `subsample=0.8`, `colsample_bytree=0.8`) with post-hoc Isotonic regression calibration.
5. **XGBoost**: Tree ensemble benchmark (`n_estimators=150`, `learning_rate=0.03`, `max_depth=5`) with post-hoc Isotonic calibration.

---

## 6. Theory-Grounded Pedagogical Intervention Catalog

The automated triage engine (`src/interventions.py`) translates individual SHAP vectors into actionable educational tickets:
- **Tier 1 (Automated Digital Nudge)**: Triggered at Week 4 when `days_since_last_click > 7`. Grounded in Zimmerman's Self-Regulated Learning (SRL) theory. Delivers low-friction re-engagement checklists and forum links without consuming human advising time.
- **Tier 2 (Proactive Advisor Outreach)**: Triggered at Weeks 4–6 when `missed_tma_count >= 1`. Grounded in Tinto's Student Departure theory and academic momentum. Dispatches 48-hour priority advisor tickets offering diagnostic micro-scaffolding and deadline grace periods.
- **Tier 3 (Adaptive Curricular Scaffolding)**: Triggered at Week 6 when `click_ratio_7d_to_14d < 0.35`. Grounded in Cognitive Load Theory (Sweller, 2011). Recommends modular remedial videos, peer mentoring, and formative quizzes.

---

## 7. Citation & License

  note      = {Under Review}
}
```

**License:** MIT License. Free for academic and research reproducibility.

