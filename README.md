# EarlyPredict-LA: Temporal Explainability and Cross-Presentation Generalization

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Benchmark](https://img.shields.io/badge/Benchmark-OULAD%20Benchmark-orange.svg)](https://doi.org/10.1038/sdata.2017.171)

> **Official Open-Source Replication Repository** for the paper:  
> **"How Early Can We Predict Student Withdrawal? Temporal Explainability and Cross-Presentation Generalization"**  
> *Target Journal: International Journal of Information and Education Technology (IJIET, ISSN: 2010-3689)*  
> **Authors:** Abdelkader Berrouachedi (`aberrouachedi@ieee.org`) and Bettahar Gasmia (`gasmia.bettahar.49@gmail.com`)  
> *Under Open-Source Apache 2.0 License.*

---

## 📌 Abstract & Overview

Timely identification of distance-education learners at risk of withdrawal remains a pivotal operational challenge in modern learning management systems (LMS). While retrospective predictive models often claim high diagnostic accuracy, they routinely suffer from **temporal lookahead leakage**, unrealistic **random split evaluations**, and a failure to address **finite institutional advising capacity**.

**EarlyPredict-LA** resolves this via a point-in-time, leakage-controlled learning analytics architecture:
1. **Strict Zero-Leakage Feature Firewall**: Predictor variables are strictly confined to clickstream events and assessment submissions timestamped on or before the cutoff day ($t \in \{14, 28, 42, 56\}$). Withdrawal dates define eligibility and the outcome label, never entering feature extraction.
2. **Dynamic Point-in-Time Risk Sets**: Active denominators are dynamically refreshed at each cutoff; students who withdrew prior to day $t$ are purged from the eligible pool.
3. **Out-of-Presentation Chronological Validation**: Models are trained on historical cohorts (`2013B`, `2013J`), calibrated on intermediate term (`2014B`), and evaluated on unseen future presentation (`2014J`).
4. **Identity Leakage Purge**: Multi-enrolled recurring students (1,329 students) appearing in prior terms are systematically eliminated from the held-out test cohort.
5. **Post-Hoc Probability Calibration**: Platt sigmoid scaling and Isotonic regression are fitted strictly on validation data, optimizing probabilistic Brier scores.
6. **Resource-Constrained Advisor Triage**: Evaluates precision and recall under operational advising constraints (top 5%, 10%, 20% budgets).
7. **Theory-Grounded Pedagogical Intervention Engine**: Translates SHAP explanation vectors into a 3-tiered action catalog (Table 10) mapped to Self-Regulated Learning (SRL), Academic Momentum, and Cognitive Load Theory.

---

## 🏗️ Repository Architecture

```
Code_Git/
├── data/
│   ├── sample_student_cohort.json       # Structured student matriculation records across presentations
│   ├── sample_vle_interactions.json     # Timestamped VLE clickstream logs across 7 resource categories
│   ├── sample_assessments.json          # Scheduled Tutor-Marked Assignment (TMA) milestones & submissions
│   └── intervention_catalog.json        # Structured 3-tier pedagogical action catalog (Table 10)
├── src/
│   ├── __init__.py                      # Package initialization
│   ├── config.py                        # Global configurations, cutoffs, features, partitions
│   ├── data_loader.py                   # Data ingestion, OULAD validation, synthetic demo generator
│   ├── features.py                      # Point-in-time feature engineering with zero-leakage firewall
│   ├── models.py                        # Prevalence, Dormancy Rule, LogReg, LightGBM, XGBoost, Calibrators
│   ├── evaluation.py                    # PR-AUC, AUROC, Brier score, ECE, contact budgets, bootstrap CIs
│   ├── explainability.py                # TreeExplainer SHAP attribution & rank stability (Kendall tau, Spearman rho)
│   ├── interventions.py                 # Actionable 3-tier pedagogical intervention engine (Table 10)
│   ├── memo_generator.py                # Line-referenced student triage memorandum synthesizer
│   └── visualization.py                 # Generates 300 DPI publication figures & exports tables
├── run_earlypredict.py                  # Master replication runner executing all 7 benchmark phases
├── main.py                              # Comprehensive CLI supporting real & demo datasets
├── pedagogical_intervention_memo_sample.txt # Sample line-referenced advisor triage memorandum
├── requirements.txt                     # Pinned dependencies (numpy, pandas, scikit-learn, lightgbm, etc.)
├── LICENSE                              # Apache 2.0 Open Source License
└── README.md                            # Comprehensive system documentation
```

---

## ⚡ Quickstart & Installation

### 1. Prerequisites & Environment Setup
Clone the repository and create a Python 3.10+ virtual environment:
```bash
git clone https://github.com/AbdelkaderBerrouachedi/LearningAnalytics.git
cd LearningAnalytics

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Replication Pipeline
Execute the 7-phase benchmark pipeline with a single command:
```bash
python run_earlypredict.py
```

This runs all architectural phases in ~1.5 seconds:
- **Phase 1**: Ingests multi-modal VLE clickstreams & verifies zero-leakage firewalls.
- **Phase 2**: Audits dynamic active cohort sizing across cutoffs (Weeks 2, 4, 6, 8).
- **Phase 3**: Enforces chronological forward-transfer (2013 $\to$ 2014) and purges recurring identities.
- **Phase 4**: Calibrates probabilities on the 2014B validation set via Isotonic Regression.
- **Phase 5**: Evaluates Precision/Recall under 5%, 10%, 20% advisor contact budgets.
- **Phase 6**: Computes TreeExplainer SHAP attributions and cross-presentation rank stability.
- **Phase 7**: Synthesizes a formal, line-referenced Pedagogical Triage Memorandum (`pedagogical_intervention_memo_sample.txt`).

---

## 📊 Full Empirical Replication on OULAD

To reproduce the full benchmark on the complete 32,593-student OULAD dataset:
```bash
# 1. Download raw OULAD from UCI repository
python main.py --download

# 2. Extract point-in-time features strictly <= t
python main.py --extract-features

# 3. Train models, calibrate, and evaluate on unseen 2014J
python main.py --train-eval

# 4. Generate triage tickets mapped to Table 10
python main.py --triage

# 5. Export publication figures (300 DPI) and tables
python main.py --figures
```

---

## 🎯 Theory-Grounded Pedagogical Intervention Catalog (Table 10)

| Priority Tier | Dominant SHAP Indicator | Theoretical Framework | Prescribed Action Protocol | Supporting RCT Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Digital Nudge**<br>*(Cutoff: Week 4; Top 20%)* | **Inactivity Dormancy**:<br>`days_since_last_click > 7` | Zimmerman Self-Regulated Learning (SRL) & COM-B | Automated empathetic LMS/email nudge with study planning checklist & forum links | Shi et al. (2024); Gatare et al. (2023) |
| **Tier 2: Advisor Outreach**<br>*(Cutoff: Weeks 4–6; Top 5%–10%)* | **TMA Non-Compliance**:<br>`missed_tma_count >= 1` | Tinto Student Departure & Academic Momentum | Personal advisor call within 48h; diagnostic micro-scaffolding, deadline extension | Rust & Motz RCT (2020); Alalawi et al. (2024) |
| **Tier 3: Scaffolding**<br>*(Cutoff: Week 6; Critical Cohort)* | **Velocity Collapse**:<br>`velocity_14d_ratio < 0.35` | Cognitive Load Theory (Sweller, 2011) | Modular concept video walkthroughs, peer mentoring, and formative self-quizzes | Yamada et al. (2026); Yu (2024) |

---

## 📜 Citation & Academic Integrity

If you utilize this pipeline or benchmark in your research, please cite:

```bibtex
@article{Berrouachedi_2026,
  author    = {Abdelkader Berrouachedi and Bettahar Gasmia},
  title     = {How Early Can We Predict Student Withdrawal? Temporal Explainability and Cross-Presentation Generalization},
  journal   = {International Journal of Information and Education Technology},
  year      = {2026},
  note      = {Under Review}
}
```

**License:** Open-source under the [Apache 2.0 License](LICENSE).
