#!/usr/bin/env python3
"""
Master Orchestration CLI for Paper P6:
'How Early Can We Predict Student Withdrawal? Explainable Analytics Across Course Presentations'
Target Journal: International Journal of Information and Education Technology (IJIET)
Authors: Abdelkader Berrouachedi and Bettahar Gasmia
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to python path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from src.config import (
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, TABLES_DIR, FIGURES_DIR,
    CUTOFFS, TRAIN_PRESENTATIONS, VAL_PRESENTATIONS, TEST_PRESENTATIONS,
    PRIMARY_FEATURES, FEATURE_SUBSETS, RANDOM_SEED, MODULE_DOMAINS
)
from src.data_loader import download_oulad, load_raw_data, generate_synthetic_demo_data
from src.features import extract_features_for_cutoff
from src.models import (
    PrevalenceBaseline, ActivityDormancyRule, LogisticRegressionPipeline,
    LightGBMPipeline, XGBoostPipeline, PlattCalibrator, IsotonicCalibrator
)
from src.evaluation import evaluate_predictions, bootstrap_ci
from src.explainability import compute_shap_explanations
from src.interventions import PedagogicalInterventionEngine
from src.visualization import (
    plot_figure1_pipeline, plot_figure2_chronological_design,
    plot_figure3_performance, plot_figure4_calibration, plot_figure5_shap
)

def step_download():
    print("=" * 70)
    print("STAGE 1: DOWNLOADING OULAD DATASET FROM UCI REPOSITORY")
    print("=" * 70)
    download_oulad(RAW_DATA_DIR)

def step_audit(data_path: Path):
    print("=" * 70)
    print("STAGE 2: DATASET ELIGIBILITY & ATTRITION AUDIT")
    print("=" * 70)
    base_students, svle, assessments, sa, courses = load_raw_data(data_path)
    print(f"\nTotal valid enrollments: {len(base_students):,}")
    print(f"Total VLE interaction records: {len(svle):,}")
    print(f"Total scheduled assessments: {len(assessments):,}")
    print(f"Total student assessment submissions: {len(sa):,}")

    print("\n--- Point-in-Time Active Cohort Sizing (Table 2 Audit) ---")
    for w, day in CUTOFFS.items():
        active = base_students[
            base_students["date_unregistration"].isna() | (base_students["date_unregistration"] > day)
        ]
        withdrawn = active[active["date_unregistration"].notna()]
        print(f"Week {w} (Day {day:2d}): Active N = {len(active):5,d} | Future Drops = {len(withdrawn):4,d} ({len(withdrawn)/len(active)*100:5.2f}%)")

def step_extract_features(data_path: Path, output_dir: Path):
    print("=" * 70)
    print("STAGE 3: POINT-IN-TIME FEATURE EXTRACTION (ZERO-LEAKAGE FIREWALL)")
    print("=" * 70)
    base_students, svle, assessments, sa, courses = load_raw_data(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for w, day in CUTOFFS.items():
        df_feat = extract_features_for_cutoff(w, day, base_students, svle, assessments, sa)
        out_csv = output_dir / f"cohort_week_{w}.csv"
        df_feat.to_csv(out_csv, index=False)
        print(f"  -> Saved Week {w} cohort to {out_csv} ({len(df_feat):,} records)")

def step_train_evaluate(processed_dir: Path, tables_dir: Path):
    print("=" * 70)
    print("STAGE 4: CHRONOLOGICAL MODEL TRAINING & VALIDATION-BASED CALIBRATION")
    print("=" * 70)
    results_t4 = []
    results_t5 = []
    calib_cache = {}

    for w, day in CUTOFFS.items():
        fpath = processed_dir / f"cohort_week_{w}.csv"
        if not fpath.exists():
            print(f"Cohort file {fpath} not found. Run --extract-features first.")
            return

        df = pd.read_csv(fpath)

        train_mask = df["code_presentation"].isin(TRAIN_PRESENTATIONS)
        val_mask = df["code_presentation"].isin(VAL_PRESENTATIONS)
        test_mask = df["code_presentation"].isin(TEST_PRESENTATIONS)

        # Purge recurring students from test partition to ensure zero identity leakage
        seen_students = set(df.loc[train_mask, "id_student"]) | set(df.loc[val_mask, "id_student"])
        test_unseen_mask = test_mask & (~df["id_student"].isin(seen_students))

        train_df = df[train_mask]
        val_df = df[val_mask]
        test_df = df[test_unseen_mask]

        print(f"\n--- Week {w} (Day {day}) Cohort Partitions ---")
        print(f"  Train (2013B/J) : {len(train_df):,} students ({train_df['y_withdrawal'].mean()*100:.1f}% drops)")
        print(f"  Val   (2014B)   : {len(val_df):,} students ({val_df['y_withdrawal'].mean()*100:.1f}% drops)")
        print(f"  Test  (2014J)   : {len(test_df):,} students ({test_df['y_withdrawal'].mean()*100:.1f}% drops; purged {test_mask.sum() - len(test_df)} recurring)")

        feats = PRIMARY_FEATURES
        X_train, y_train = train_df[feats].values, train_df["y_withdrawal"].values
        X_val, y_val = val_df[feats].values, val_df["y_withdrawal"].values
        X_test, y_test = test_df[feats].values, test_df["y_withdrawal"].values

        # 1. Prevalence Baseline
        prev_model = PrevalenceBaseline().fit(X_train, y_train)
        p_prev = prev_model.predict_proba(X_test)
        res_prev = evaluate_predictions(y_test, p_prev)

        # 2. Activity Dormancy Rule
        dorm_idx = feats.index("days_since_last_click")
        dorm_model = ActivityDormancyRule(feature_idx=dorm_idx).fit(X_train, y_train)
        p_dorm = dorm_model.predict_proba(X_test)
        res_dorm = evaluate_predictions(y_test, p_dorm)

        # 3. Logistic Regression
        lr_model = LogisticRegressionPipeline(random_state=RANDOM_SEED).fit(X_train, y_train)
        p_lr_val = lr_model.predict_proba(X_val)
        p_lr_raw = lr_model.predict_proba(X_test)
        cal_lr = PlattCalibrator().fit(p_lr_val, y_val)
        p_lr_cal = cal_lr.transform(p_lr_raw)
        res_lr = evaluate_predictions(y_test, p_lr_cal)

        # 4. LightGBM (GBDT)
        lgb_model = LightGBMPipeline(random_state=RANDOM_SEED).fit(X_train, y_train)
        p_lgb_val = lgb_model.predict_proba(X_val)
        p_lgb_raw = lgb_model.predict_proba(X_test)
        cal_iso = IsotonicCalibrator().fit(p_lgb_val, y_val)
        p_lgb_cal = cal_iso.transform(p_lgb_raw)
        res_lgb = evaluate_predictions(y_test, p_lgb_cal)

        # 5. XGBoost
        xgb_model = XGBoostPipeline(random_state=RANDOM_SEED).fit(X_train, y_train)
        p_xgb_val = xgb_model.predict_proba(X_val)
        p_xgb_raw = xgb_model.predict_proba(X_test)
        cal_xgb = IsotonicCalibrator().fit(p_xgb_val, y_val)
        p_xgb_cal = cal_xgb.transform(p_xgb_raw)
        res_xgb = evaluate_predictions(y_test, p_xgb_cal)

        calib_cache[f"week_{w}"] = {
            "y_test": y_test.tolist(),
            "prob_lgb_raw": p_lgb_raw.tolist(),
            "prob_lgb_cal": p_lgb_cal.tolist(),
            "prob_lr_cal": p_lr_cal.tolist()
        }

        # Store for Table 4 & Table 5
        models_dict = {
            "Prevalence Baseline": (res_prev, p_prev),
            "Activity Heuristic Rule": (res_dorm, p_dorm),
            "Logistic Regression": (res_lr, p_lr_cal),
            "LightGBM (GBDT)": (res_lgb, p_lgb_cal),
            "XGBoost": (res_xgb, p_xgb_cal)
        }

        for m_name, (metrics, prob_arr) in models_dict.items():
            ci = bootstrap_ci(y_test, prob_arr, n_bootstraps=200, random_seed=RANDOM_SEED)
            results_t4.append({
                "Cutoff Week": f"Week {w}",
                "Observation Window": f"Day 0 to {day}",
                "Model": m_name,
                "PR-AUC": f"{ci['pr_auc'][0]:.3f} [{ci['pr_auc'][1]:.3f}, {ci['pr_auc'][2]:.3f}]",
                "AUROC": f"{ci['roc_auc'][0]:.3f} [{ci['roc_auc'][1]:.3f}, {ci['roc_auc'][2]:.3f}]",
                "Brier Score": f"{ci['brier'][0]:.4f} [{ci['brier'][1]:.4f}, {ci['brier'][2]:.4f}]",
                "ECE": f"{metrics['ece']:.4f}"
            })
            results_t5.append({
                "Cutoff Week": f"Week {w}",
                "Model": m_name,
                "Prec@5%": f"{ci['p5'][0]*100:.1f}% [{ci['p5'][1]*100:.1f}%, {ci['p5'][2]*100:.1f}%]",
                "Rec@5%": f"{ci['r5'][0]*100:.1f}% [{ci['r5'][1]*100:.1f}%, {ci['r5'][2]*100:.1f}%]",
                "Prec@10%": f"{ci['p10'][0]*100:.1f}% [{ci['p10'][1]*100:.1f}%, {ci['p10'][2]*100:.1f}%]",
                "Rec@10%": f"{ci['r10'][0]*100:.1f}% [{ci['r10'][1]*100:.1f}%, {ci['r10'][2]*100:.1f}%]",
                "Prec@20%": f"{ci['p20'][0]*100:.1f}% [{ci['p20'][1]*100:.1f}%, {ci['p20'][2]*100:.1f}%]",
                "Rec@20%": f"{ci['r20'][0]*100:.1f}% [{ci['r20'][1]*100:.1f}%, {ci['r20'][2]*100:.1f}%]"
            })

    # Save cache
    with open(tables_dir / "predictions.json", "w") as f:
        json.dump(calib_cache, f)

    pd.DataFrame(results_t4).to_csv(tables_dir / "table4_model_performance.csv", index=False)
    pd.DataFrame(results_t5).to_csv(tables_dir / "table5_contact_budgets.csv", index=False)
    print(f"\nSuccessfully generated and saved Table 4 and Table 5 in {tables_dir}")
    return calib_cache

def step_triage_interventions(processed_dir: Path, output_dir: Path):
    print("=" * 70)
    print("STAGE 5: TRIAGE & PEDAGOGICAL INTERVENTION TICKET GENERATION (TABLE 10)")
    print("=" * 70)
    fpath4 = processed_dir / "cohort_week_4.csv"
    if not fpath4.exists():
        print(f"Week 4 cohort file not found at {fpath4}.")
        return

    df4 = pd.read_csv(fpath4)
    train_mask = df4["code_presentation"].isin(TRAIN_PRESENTATIONS)
    test_mask = df4["code_presentation"].isin(TEST_PRESENTATIONS)

    seen = set(df4.loc[train_mask, "id_student"]) | set(df4.loc[df4["code_presentation"] == "2014B", "id_student"])
    test_unseen = test_mask & (~df4["id_student"].isin(seen))
    test_df = df4[test_unseen].copy()

    feats = PRIMARY_FEATURES
    X_train, y_train = df4.loc[train_mask, feats].values, df4.loc[train_mask, "y_withdrawal"].values
    X_test, y_test = test_df[feats].values, test_df["y_withdrawal"].values

    model = LightGBMPipeline(random_state=RANDOM_SEED).fit(X_train, y_train)
    probs = model.predict_proba(X_test)

    engine = PedagogicalInterventionEngine(top_budget_pct=0.10)
    triage_tickets = engine.triage_cohort(test_df, probs)

    out_json = output_dir / "intervention_triage_sample.json"
    with open(out_json, "w") as f:
        json.dump(triage_tickets[:50], f, indent=2)

    flagged = [t for t in triage_tickets if t["flagged_for_intervention"]]
    print(f"Generated {len(triage_tickets):,} student assessments for Week 4.")
    print(f"Flagged {len(flagged):,} students under top 10% advising budget.")
    print(f"Sample intervention tickets exported to {out_json}")

def step_generate_visuals(processed_dir: Path, tables_dir: Path, figures_dir: Path):
    print("=" * 70)
    print("STAGE 6: GENERATING PUBLICATION FIGURES (300 DPI)")
    print("=" * 70)
    f1 = plot_figure1_pipeline(figures_dir)
    print(f"  -> {f1.name}")
    f2 = plot_figure2_chronological_design(figures_dir)
    print(f"  -> {f2.name}")
    f3 = plot_figure3_performance(tables_dir, figures_dir)
    print(f"  -> {f3.name}")

    pred_json = tables_dir / "predictions.json"
    if pred_json.exists():
        with open(pred_json) as f:
            calib_data = json.load(f)
        f4 = plot_figure4_calibration(calib_data, figures_dir)
        print(f"  -> {f4.name}")

    # Figure 5 SHAP
    fpath4 = processed_dir / "cohort_week_4.csv"
    if fpath4.exists():
        df4 = pd.read_csv(fpath4)
        train_mask = df4["code_presentation"].isin(TRAIN_PRESENTATIONS)
        test_mask = df4["code_presentation"].isin(TEST_PRESENTATIONS)
        seen = set(df4.loc[train_mask, "id_student"]) | set(df4.loc[df4["code_presentation"] == "2014B", "id_student"])
        test_unseen = test_mask & (~df4["id_student"].isin(seen))

        feats = PRIMARY_FEATURES
        X_train = df4.loc[train_mask, feats].values
        y_train = df4.loc[train_mask, "y_withdrawal"].values
        X_test = df4.loc[test_unseen, feats].values

        model = LightGBMPipeline(random_state=RANDOM_SEED).fit(X_train, y_train)
        shap_res = compute_shap_explanations(model.model, X_train, X_test, feats, max_samples=500)
        f5 = plot_figure5_shap(shap_res, figures_dir)
        print(f"  -> {f5.name}")
        print(f"\n  Cross-Presentation SHAP Stability:")
        print(f"    Kendall's tau : {shap_res['stability']['kendall_tau']:.3f} (p = {shap_res['stability']['kendall_p']:.4e})")
        print(f"    Spearman's rho: {shap_res['stability']['spearman_rho']:.3f} (p = {shap_res['stability']['spearman_p']:.4e})")

def main():
    parser = argparse.ArgumentParser(
        description="Point-in-Time Student Withdrawal Early Warning Framework (Paper P6)"
    )
    parser.add_argument("--demo", action="store_true", help="Run rapid synthetic demo end-to-end (~5s)")
    parser.add_argument("--download", action="store_true", help="Download raw OULAD from UCI repository")
    parser.add_argument("--audit", action="store_true", help="Audit dataset eligibility and cohort sizing")
    parser.add_argument("--extract-features", action="store_true", help="Extract point-in-time features for Weeks 2, 4, 6, 8")
    parser.add_argument("--train-eval", action="store_true", help="Run chronological model training, calibration & evaluation")
    parser.add_argument("--triage", action="store_true", help="Generate pedagogical intervention tickets (Table 10)")
    parser.add_argument("--figures", action="store_true", help="Generate publication-ready 300 DPI figures")
    parser.add_argument("--all", action="store_true", help="Run all phases end-to-end on full dataset")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.demo:
        print("\n" + "=" * 70)
        print("RUNNING RAPID SYNTHETIC VERIFICATION DEMO")
        print("=" * 70)
        demo_dir = DATA_DIR / "demo_raw"
        demo_processed = DATA_DIR / "demo_processed"
        generate_synthetic_demo_data(demo_dir, n_students=600)
        step_audit(demo_dir)
        step_extract_features(demo_dir, demo_processed)
        step_train_evaluate(demo_processed, TABLES_DIR)
        step_triage_interventions(demo_processed, TABLES_DIR)
        step_generate_visuals(demo_processed, TABLES_DIR, FIGURES_DIR)
        print("\n[SUCCESS] Rapid synthetic verification run completed perfectly!")
        return

    if args.download or args.all:
        step_download()

    if args.audit or args.all:
        step_audit(RAW_DATA_DIR)

    if args.extract_features or args.all:
        step_extract_features(RAW_DATA_DIR, PROCESSED_DATA_DIR)

    if args.train_eval or args.all:
        step_train_evaluate(PROCESSED_DATA_DIR, TABLES_DIR)

    if args.triage or args.all:
        step_triage_interventions(PROCESSED_DATA_DIR, TABLES_DIR)

    if args.figures or args.all:
        step_generate_visuals(PROCESSED_DATA_DIR, TABLES_DIR, FIGURES_DIR)

if __name__ == "__main__":
    main()
