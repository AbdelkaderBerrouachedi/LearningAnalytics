import pandas as pd
import numpy as np
import os
import json
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
import lightgbm as lgb
import xgboost as xgb

PROCESSED_DIR = r"..\data\processed"
TABLES_DIR = r"..\tables"
os.makedirs(TABLES_DIR, exist_ok=True)

PRIMARY_FEATURES = [
    'studied_credits', 'num_of_prev_attempts', 'date_registration_imputed',
    'gender_binary', 'disability_binary', 'age_band_ord', 'education_ord', 'imd_band_ord',
    'total_clicks', 'active_days', 'clicks_per_active_day', 'distinct_sites',
    'pre_course_clicks', 'pre_course_active_days',
    'clicks_last_7d', 'active_days_last_7d', 'clicks_last_14d', 'active_days_last_14d',
    'days_since_last_click', 'click_ratio_7d_to_14d', 'active_day_ratio_7d_to_14d',
    'recent_to_total_click_share',
    'clicks_forumng', 'clicks_oucontent', 'clicks_resource', 'clicks_subpage',
    'clicks_url', 'clicks_quiz', 'clicks_homepage', 'clicks_other',
    'assessments_due', 'assessments_submitted', 'missed_assessments',
    'submission_rate', 'mean_submission_delay', 'late_submissions_count'
]

def calibrate_platt(val_prob, y_val, test_prob):
    eps = 1e-7
    val_p = np.clip(val_prob, eps, 1 - eps)
    val_logit = np.log(val_p / (1 - val_p)).reshape(-1, 1)
    
    test_p = np.clip(test_prob, eps, 1 - eps)
    test_logit = np.log(test_p / (1 - test_p)).reshape(-1, 1)
    
    lr = LogisticRegression(C=1.0)
    lr.fit(val_logit, y_val)
    return lr.predict_proba(test_logit)[:, 1]

def calibrate_isotonic(val_prob, y_val, test_prob):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(val_prob, y_val)
    return np.clip(iso.predict(test_prob), 0.0, 1.0)

def compute_budget_metrics(y_true, y_prob, budget_pct):
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0
    k = max(1, int(np.floor(budget_pct * n)))
    total_positives = int(np.sum(y_true))
    if total_positives == 0:
        return 0.0, 0.0

    if np.ptp(y_prob) < 1e-9:
        base_rate = float(np.mean(y_true))
        return base_rate, float(budget_pct)

    sort_idx = np.argsort(-y_prob, kind="mergesort")
    scores_sorted = y_prob[sort_idx]
    y_sorted = y_true[sort_idx]

    threshold = scores_sorted[k - 1]
    above_mask = scores_sorted > threshold
    tie_mask = scores_sorted == threshold

    n_above = int(np.sum(above_mask))
    n_tie = int(np.sum(tie_mask))

    pos_above = float(np.sum(y_sorted[above_mask]))
    pos_tie = float(np.sum(y_sorted[tie_mask]))

    k_from_tie = k - n_above
    expected_pos_from_tie = pos_tie * (k_from_tie / n_tie) if n_tie > 0 and k_from_tie > 0 else 0.0
    expected_total_pos = pos_above + expected_pos_from_tie
    precision = float(expected_total_pos / k)
    recall = float(expected_total_pos / total_positives)
    return precision, recall

def evaluate_predictions(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.nan_to_num(np.asarray(y_prob), nan=0.0)
    
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0))
    
    p5, r5 = compute_budget_metrics(y_true, y_prob, 0.05)
    p10, r10 = compute_budget_metrics(y_true, y_prob, 0.10)
    p20, r20 = compute_budget_metrics(y_true, y_prob, 0.20)
    
    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'brier': brier,
        'p5': p5, 'r5': r5,
        'p10': p10, 'r10': r10,
        'p20': p20, 'r20': r20
    }

def bootstrap_ci(y_true, y_prob, n_bootstraps=1000, random_seed=42):
    rng = np.random.RandomState(random_seed)
    n = len(y_true)
    boot_metrics = {k: [] for k in ['pr_auc', 'roc_auc', 'brier', 'p5', 'r5', 'p10', 'r10', 'p20', 'r20']}
    
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        y_b = y_true[idx]
        if np.sum(y_b) == 0 or np.sum(y_b) == n:
            continue
        p_b = y_prob[idx]
        res = evaluate_predictions(y_b, p_b)
        for k in boot_metrics:
            boot_metrics[k].append(res[k])
            
    ci_res = {}
    for k, vals in boot_metrics.items():
        ci_res[f"{k}_low"] = np.percentile(vals, 2.5)
        ci_res[f"{k}_high"] = np.percentile(vals, 97.5)
    return ci_res

def run_all_experiments():
    print("=========================================================")
    print("STARTING MAIN EXPERIMENTAL BENCHMARK (RQ1, RQ2, RQ3)")
    print("=========================================================")
    
    results_table3 = []
    results_table4 = []
    saved_predictions = {}
    
    for week in [2, 4, 6, 8]:
        fpath = os.path.join(PROCESSED_DIR, f"cohort_week_{week}.csv")
        df = pd.read_csv(fpath)
        cutoff_day = {2: 14, 4: 28, 6: 42, 8: 56}[week]
        
        # Split definitions
        train_mask = df['code_presentation'].isin(['2013B', '2013J'])
        val_mask = df['code_presentation'] == '2014B'
        test_mask = df['code_presentation'] == '2014J'
        
        train_students = set(df.loc[train_mask, 'id_student'])
        val_students = set(df.loc[val_mask, 'id_student'])
        seen_students = train_students | val_students
        
        test_unseen_mask = test_mask & (~df['id_student'].isin(seen_students))
        
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_unseen_mask].copy()
        test_all_df = df[test_mask].copy()
        
        X_train = train_df[PRIMARY_FEATURES].values
        y_train = train_df['y_withdrawal'].values
        
        X_val = val_df[PRIMARY_FEATURES].values
        y_val = val_df['y_withdrawal'].values
        
        X_test = test_df[PRIMARY_FEATURES].values
        y_test = test_df['y_withdrawal'].values
        
        prev_train = np.mean(y_train)
        
        print(f"\n--- Week {week} (Day {cutoff_day}) ---")
        print(f"Train N={len(train_df):,} (prev={prev_train*100:.2f}%), Val N={len(val_df):,}, Test Unseen N={len(test_df):,} (prev={np.mean(y_test)*100:.2f}%)")
        
        # 1. Baseline 1: Prevalence
        prob_prev = np.full(len(y_test), prev_train)
        res_prev = evaluate_predictions(y_test, prob_prev)
        ci_prev = bootstrap_ci(y_test, prob_prev)
        
        # 2. Baseline 2: Activity Heuristic Rule
        inact_score = test_df['days_since_last_click'].values * 0.1 - np.log1p(test_df['clicks_last_14d'].values)
        prob_rule = 1.0 / (1.0 + np.exp(-inact_score / (np.std(inact_score) + 1e-5)))
        res_rule = evaluate_predictions(y_test, prob_rule)
        ci_rule = bootstrap_ci(y_test, prob_rule)
        
        # 3. Model 1: Logistic Regression
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        
        lr_base = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        lr_base.fit(X_train_s, y_train)
        
        prob_val_lr = lr_base.predict_proba(X_val_s)[:, 1]
        prob_lr_raw = lr_base.predict_proba(X_test_s)[:, 1]
        prob_lr = calibrate_platt(prob_val_lr, y_val, prob_lr_raw)
        
        res_lr = evaluate_predictions(y_test, prob_lr)
        ci_lr = bootstrap_ci(y_test, prob_lr)
        brier_lr_raw = brier_score_loss(y_test, prob_lr_raw)
        
        # 4. Model 2: LightGBM
        lgb_base = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.03, num_leaves=31,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        lgb_base.fit(X_train, y_train)
        
        prob_val_lgb = lgb_base.predict_proba(X_val)[:, 1]
        prob_lgb_raw = lgb_base.predict_proba(X_test)[:, 1]
        prob_lgb = calibrate_isotonic(prob_val_lgb, y_val, prob_lgb_raw)
        
        res_lgb = evaluate_predictions(y_test, prob_lgb)
        ci_lgb = bootstrap_ci(y_test, prob_lgb)
        brier_lgb_raw = brier_score_loss(y_test, prob_lgb_raw)
        
        # 5. Model 3: XGBoost
        xgb_base = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss'
        )
        xgb_base.fit(X_train, y_train)
        
        prob_val_xgb = xgb_base.predict_proba(X_val)[:, 1]
        prob_xgb_raw = xgb_base.predict_proba(X_test)[:, 1]
        prob_xgb = calibrate_isotonic(prob_val_xgb, y_val, prob_xgb_raw)
        
        res_xgb = evaluate_predictions(y_test, prob_xgb)
        ci_xgb = bootstrap_ci(y_test, prob_xgb)
        brier_xgb_raw = brier_score_loss(y_test, prob_xgb_raw)
        
        print(f"Results for Week {week}:")
        print(f"  LightGBM: PR-AUC={res_lgb['pr_auc']:.3f} [{ci_lgb['pr_auc_low']:.3f}, {ci_lgb['pr_auc_high']:.3f}], AUROC={res_lgb['roc_auc']:.3f}, Prec@10%={res_lgb['p10']*100:.1f}%, Rec@10%={res_lgb['r10']*100:.1f}%")
        print(f"  Logistic: PR-AUC={res_lr['pr_auc']:.3f} [{ci_lr['pr_auc_low']:.3f}, {ci_lr['pr_auc_high']:.3f}], AUROC={res_lr['roc_auc']:.3f}")
        print(f"  ActRule : PR-AUC={res_rule['pr_auc']:.3f}, AUROC={res_rule['roc_auc']:.3f}")
        
        saved_predictions[f"week_{week}"] = {
            'y_test': y_test.tolist(),
            'prob_prev': prob_prev.tolist(),
            'prob_rule': prob_rule.tolist(),
            'prob_lr_raw': prob_lr_raw.tolist(),
            'prob_lr_cal': prob_lr.tolist(),
            'prob_lgb_raw': prob_lgb_raw.tolist(),
            'prob_lgb_cal': prob_lgb.tolist(),
            'prob_xgb_raw': prob_xgb_raw.tolist(),
            'prob_xgb_cal': prob_xgb.tolist(),
            'module': test_df['code_module'].tolist()
        }
        
        models_dict = {
            'Prevalence Baseline': (res_prev, ci_prev, res_prev['brier']),
            'Activity Heuristic Rule': (res_rule, ci_rule, res_rule['brier']),
            'Logistic Regression': (res_lr, ci_lr, brier_lr_raw),
            'LightGBM (GBDT)': (res_lgb, ci_lgb, brier_lgb_raw),
            'XGBoost': (res_xgb, ci_xgb, brier_xgb_raw)
        }
        
        for mname, (res, ci, b_raw) in models_dict.items():
            results_table3.append({
                'Week': f"Week {week}",
                'Cutoff Day': cutoff_day,
                'Model': mname,
                'PR-AUC': f"{res['pr_auc']:.3f} [{ci['pr_auc_low']:.3f}, {ci['pr_auc_high']:.3f}]",
                'AUROC': f"{res['roc_auc']:.3f} [{ci['roc_auc_low']:.3f}, {ci['roc_auc_high']:.3f}]",
                'Raw Brier': f"{b_raw:.4f}",
                'Calibrated Brier': f"{res['brier']:.4f} [{ci['brier_low']:.4f}, {ci['brier_high']:.4f}]"
            })
            
            results_table4.append({
                'Week': f"Week {week}",
                'Cutoff Day': cutoff_day,
                'Model': mname,
                'Prec@5%': f"{res['p5']*100:.1f}% [{ci['p5_low']*100:.1f}, {ci['p5_high']*100:.1f}]",
                'Rec@5%': f"{res['r5']*100:.1f}% [{ci['r5_low']*100:.1f}, {ci['r5_high']*100:.1f}]",
                'Prec@10%': f"{res['p10']*100:.1f}% [{ci['p10_low']*100:.1f}, {ci['p10_high']*100:.1f}]",
                'Rec@10%': f"{res['r10']*100:.1f}% [{ci['r10_low']*100:.1f}, {ci['r10_high']*100:.1f}]",
                'Prec@20%': f"{res['p20']*100:.1f}% [{ci['p20_low']*100:.1f}, {ci['p20_high']*100:.1f}]",
                'Rec@20%': f"{res['r20']*100:.1f}% [{ci['r20_low']*100:.1f}, {ci['r20_high']*100:.1f}]"
            })
            
    df_t3 = pd.DataFrame(results_table3)
    df_t4 = pd.DataFrame(results_table4)
    
    df_t3.to_csv(os.path.join(TABLES_DIR, "table3_model_performance.csv"), index=False)
    df_t3.to_markdown(os.path.join(TABLES_DIR, "table3_model_performance.md"), index=False)
    
    df_t4.to_csv(os.path.join(TABLES_DIR, "table4_contact_budgets.csv"), index=False)
    df_t4.to_markdown(os.path.join(TABLES_DIR, "table4_contact_budgets.md"), index=False)
    
    with open(os.path.join(TABLES_DIR, "predictions.json"), 'w') as f:
        json.dump(saved_predictions, f)
        
    print("\nMain benchmark complete! Table 3 and Table 4 generated successfully.")

if __name__ == "__main__":
    run_all_experiments()
