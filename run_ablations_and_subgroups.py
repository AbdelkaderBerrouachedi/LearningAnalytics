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

PROCESSED_DIR = r"..\data\processed"
TABLES_DIR = r"..\tables"
os.makedirs(TABLES_DIR, exist_ok=True)

# Feature subsets for Table 6 ablations
FEATURE_GROUPS = {
    'Full Primary': [
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
    ],
    'Without Demographics': [
        'studied_credits', 'num_of_prev_attempts', 'date_registration_imputed',
        'total_clicks', 'active_days', 'clicks_per_active_day', 'distinct_sites',
        'pre_course_clicks', 'pre_course_active_days',
        'clicks_last_7d', 'active_days_last_7d', 'clicks_last_14d', 'active_days_last_14d',
        'days_since_last_click', 'click_ratio_7d_to_14d', 'active_day_ratio_7d_to_14d',
        'recent_to_total_click_share',
        'clicks_forumng', 'clicks_oucontent', 'clicks_resource', 'clicks_subpage',
        'clicks_url', 'clicks_quiz', 'clicks_homepage', 'clicks_other',
        'assessments_due', 'assessments_submitted', 'missed_assessments',
        'submission_rate', 'mean_submission_delay', 'late_submissions_count'
    ],
    'Demographics Only': [
        'gender_binary', 'disability_binary', 'age_band_ord', 'education_ord', 'imd_band_ord',
        'studied_credits', 'num_of_prev_attempts', 'date_registration_imputed'
    ],
    'VLE Engagement Only': [
        'total_clicks', 'active_days', 'clicks_per_active_day', 'distinct_sites',
        'pre_course_clicks', 'pre_course_active_days',
        'clicks_last_7d', 'active_days_last_7d', 'clicks_last_14d', 'active_days_last_14d',
        'days_since_last_click', 'click_ratio_7d_to_14d', 'active_day_ratio_7d_to_14d',
        'recent_to_total_click_share',
        'clicks_forumng', 'clicks_oucontent', 'clicks_resource', 'clicks_subpage',
        'clicks_url', 'clicks_quiz', 'clicks_homepage', 'clicks_other'
    ],
    'Assessment Participation Only': [
        'assessments_due', 'assessments_submitted', 'missed_assessments',
        'submission_rate', 'mean_submission_delay', 'late_submissions_count'
    ],
    'Full Primary + Assessment Scores': [
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
        'submission_rate', 'mean_submission_delay', 'late_submissions_count',
        'mean_assessment_score'
    ]
}

def calibrate_isotonic(val_prob, y_val, test_prob):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(val_prob, y_val)
    return np.clip(iso.predict(test_prob), 0.0, 1.0)

def compute_budget_metrics(y_true, y_prob, budget_pct):
    n = len(y_true)
    k = max(1, int(np.floor(budget_pct * n)))
    top_indices = np.argsort(y_prob)[::-1][:k]
    flagged_true = y_true[top_indices]
    precision = np.mean(flagged_true)
    total_positives = np.sum(y_true)
    recall = np.sum(flagged_true) / total_positives if total_positives > 0 else 0.0
    return precision, recall

def evaluate_metrics(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.nan_to_num(np.asarray(y_prob), nan=0.0)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0))
    p10, r10 = compute_budget_metrics(y_true, y_prob, 0.10)
    return {'pr_auc': pr_auc, 'roc_auc': roc_auc, 'brier': brier, 'p10': p10, 'r10': r10}

def run_table5_presentation_transfer():
    print("\n--- Running Table 5: Held-out Presentation Analysis (2014J Modules) ---", flush=True)
    # Using Week 4 (Day 28) and Week 8 (Day 56)
    rows_t5 = []
    
    for week in [4, 8]:
        fpath = os.path.join(PROCESSED_DIR, f"cohort_week_{week}.csv")
        df = pd.read_csv(fpath)
        
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
        
        feats = FEATURE_GROUPS['Full Primary']
        X_train, y_train = train_df[feats].values, train_df['y_withdrawal'].values
        X_val, y_val = val_df[feats].values, val_df['y_withdrawal'].values
        
        # Fit LightGBM
        model = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.03, num_leaves=31,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        model.fit(X_train, y_train)
        
        prob_val = model.predict_proba(X_val)[:, 1]
        
        # Overall 2014J performance
        X_test, y_test = test_df[feats].values, test_df['y_withdrawal'].values
        prob_test = calibrate_isotonic(prob_val, y_val, model.predict_proba(X_test)[:, 1])
        res_overall = evaluate_metrics(y_test, prob_test)
        
        rows_t5.append({
            'Week': f"Week {week}",
            'Module': 'ALL 2014J (Overall)',
            'Domain': 'Multi-Disciplinary',
            'Eligible N': len(y_test),
            'Withdrawn N': int(np.sum(y_test)),
            'Prevalence (%)': f"{np.mean(y_test)*100:.2f}%",
            'PR-AUC': f"{res_overall['pr_auc']:.3f}",
            'AUROC': f"{res_overall['roc_auc']:.3f}",
            'Brier': f"{res_overall['brier']:.4f}",
            'Prec@10%': f"{res_overall['p10']*100:.1f}%",
            'Rec@10%': f"{res_overall['r10']*100:.1f}%"
        })
        
        # Per module in 2014J
        modules = sorted(test_df['code_module'].unique())
        domain_map = {
            'AAA': 'Social Science/Humanities',
            'BBB': 'Social Sciences',
            'CCC': 'STEM',
            'DDD': 'STEM',
            'EEE': 'STEM',
            'FFF': 'STEM',
            'GGG': 'Social Sciences'
        }
        
        for mod in modules:
            m_idx = (test_df['code_module'] == mod).values
            y_m = y_test[m_idx]
            p_m = prob_test[m_idx]
            res_m = evaluate_metrics(y_m, p_m)
            
            rows_t5.append({
                'Week': f"Week {week}",
                'Module': f"Module {mod}",
                'Domain': domain_map.get(mod, 'General'),
                'Eligible N': len(y_m),
                'Withdrawn N': int(np.sum(y_m)),
                'Prevalence (%)': f"{np.mean(y_m)*100:.2f}%",
                'PR-AUC': f"{res_m['pr_auc']:.3f}",
                'AUROC': f"{res_m['roc_auc']:.3f}",
                'Brier': f"{res_m['brier']:.4f}",
                'Prec@10%': f"{res_m['p10']*100:.1f}%",
                'Rec@10%': f"{res_m['r10']*100:.1f}%"
            })
            
    df_t5 = pd.DataFrame(rows_t5)
    df_t5.to_csv(os.path.join(TABLES_DIR, "table5_held_out_presentation.csv"), index=False)
    df_t5.to_markdown(os.path.join(TABLES_DIR, "table5_held_out_presentation.md"), index=False)
    print("Table 5 complete!", flush=True)

def run_table6_ablations_and_fairness():
    print("\n--- Running Table 6: Feature Ablations and Algorithmic Parity Audit ---", flush=True)
    rows_ablations = []
    rows_fairness = []
    
    # 1. Ablation across feature sets at Week 4 and Week 8
    for week in [4, 8]:
        fpath = os.path.join(PROCESSED_DIR, f"cohort_week_{week}.csv")
        df = pd.read_csv(fpath)
        
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
        
        y_train = train_df['y_withdrawal'].values
        y_val = val_df['y_withdrawal'].values
        y_test = test_df['y_withdrawal'].values
        
        for gname, feats in FEATURE_GROUPS.items():
            X_train = train_df[feats].values
            X_val = val_df[feats].values
            X_test = test_df[feats].values
            
            model = lgb.LGBMClassifier(
                n_estimators=150, learning_rate=0.03, num_leaves=31,
                min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1
            )
            model.fit(X_train, y_train)
            prob_val = model.predict_proba(X_val)[:, 1]
            prob_test = calibrate_isotonic(prob_val, y_val, model.predict_proba(X_test)[:, 1])
            res = evaluate_metrics(y_test, prob_test)
            
            rows_ablations.append({
                'Week': f"Week {week}",
                'Ablation Variant': gname,
                'Num Features': len(feats),
                'PR-AUC': f"{res['pr_auc']:.3f}",
                'AUROC': f"{res['roc_auc']:.3f}",
                'Brier': f"{res['brier']:.4f}",
                'Prec@10%': f"{res['p10']*100:.1f}%",
                'Rec@10%': f"{res['r10']*100:.1f}%"
            })
            
    # 2. Algorithmic Parity Audit at Week 4 (Primary Model)
    fpath4 = os.path.join(PROCESSED_DIR, "cohort_week_4.csv")
    df4 = pd.read_csv(fpath4)
    train_mask = df4['code_presentation'].isin(['2013B', '2013J'])
    val_mask = df4['code_presentation'] == '2014B'
    test_mask = df4['code_presentation'] == '2014J'
    seen_students = set(df4.loc[train_mask, 'id_student']) | set(df4.loc[val_mask, 'id_student'])
    test_df = df4[test_mask & (~df4['id_student'].isin(seen_students))].copy()
    
    feats = FEATURE_GROUPS['Full Primary']
    model = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.03, num_leaves=31,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    model.fit(df4.loc[train_mask, feats].values, df4.loc[train_mask, 'y_withdrawal'].values)
    prob_val = model.predict_proba(df4.loc[val_mask, feats].values)[:, 1]
    prob_test = calibrate_isotonic(prob_val, df4.loc[val_mask, 'y_withdrawal'].values, 
                                   model.predict_proba(test_df[feats].values)[:, 1])
    test_df['y_prob'] = prob_test
    
    # Flag top 10% overall
    k10 = int(np.floor(0.10 * len(test_df)))
    cutoff_thresh = np.sort(prob_test)[::-1][k10]
    test_df['flagged_10'] = (test_df['y_prob'] >= cutoff_thresh).astype(int)
    
    subgroups = {
        'Gender: Female': test_df['gender_binary'] == 1,
        'Gender: Male': test_df['gender_binary'] == 0,
        'Disability: Declared': test_df['disability_binary'] == 1,
        'Disability: None': test_df['disability_binary'] == 0,
        'Age: Under 35': test_df['age_band_ord'] == 1,
        'Age: 35-55': test_df['age_band_ord'] == 2,
        'Age: 55+': test_df['age_band_ord'] == 3,
        'IMD: Most Deprived (0-30%)': test_df['imd_band_ord'] <= 2,
        'IMD: Least Deprived (70-100%)': test_df['imd_band_ord'] >= 7
    }
    
    for sname, smask in subgroups.items():
        sub_df = test_df[smask]
        n_sub = len(sub_df)
        y_true = sub_df['y_withdrawal'].values
        y_p = sub_df['y_prob'].values
        flagged = sub_df['flagged_10'].values
        
        selection_rate = np.mean(flagged) * 100
        prec10 = np.sum((flagged == 1) & (y_true == 1)) / max(1, np.sum(flagged)) * 100
        fpr = np.sum((flagged == 1) & (y_true == 0)) / max(1, np.sum(y_true == 0)) * 100
        pr_auc = average_precision_score(y_true, y_p)
        
        rows_fairness.append({
            'Subgroup': sname,
            'Subgroup N': n_sub,
            'Withdrawal Prev (%)': f"{np.mean(y_true)*100:.2f}%",
            'Selection Rate (%)': f"{selection_rate:.2f}%",
            'Prec@10% Flagged': f"{prec10:.1f}%",
            'False Positive Rate (%)': f"{fpr:.2f}%",
            'PR-AUC': f"{pr_auc:.3f}"
        })
        
    # 3. Leakage sensitivity comparison: Unseen vs All in 2014J
    all_test_df = df4[test_mask].copy()
    prob_all_test = calibrate_isotonic(prob_val, df4.loc[val_mask, 'y_withdrawal'].values, 
                                       model.predict_proba(all_test_df[feats].values)[:, 1])
    res_unseen = evaluate_metrics(test_df['y_withdrawal'].values, prob_test)
    res_all = evaluate_metrics(all_test_df['y_withdrawal'].values, prob_all_test)
    
    leakage_rows = [
        {
            'Evaluation Population': 'Strictly Unseen Students (Zero Overlap, Primary)',
            'Cohort N': len(test_df),
            'PR-AUC': f"{res_unseen['pr_auc']:.3f}",
            'AUROC': f"{res_unseen['roc_auc']:.3f}",
            'Brier': f"{res_unseen['brier']:.4f}",
            'Prec@10%': f"{res_unseen['p10']*100:.1f}%",
            'Rec@10%': f"{res_unseen['r10']*100:.1f}%"
        },
        {
            'Evaluation Population': 'All Enrolled Students (With Identity Leakage)',
            'Cohort N': len(all_test_df),
            'PR-AUC': f"{res_all['pr_auc']:.3f}",
            'AUROC': f"{res_all['roc_auc']:.3f}",
            'Brier': f"{res_all['brier']:.4f}",
            'Prec@10%': f"{res_all['p10']*100:.1f}%",
            'Rec@10%': f"{res_all['r10']*100:.1f}%"
        }
    ]
    
    df_t6_abl = pd.DataFrame(rows_ablations)
    df_t6_fair = pd.DataFrame(rows_fairness)
    df_leak = pd.DataFrame(leakage_rows)
    
    df_t6_abl.to_csv(os.path.join(TABLES_DIR, "table6a_ablations.csv"), index=False)
    df_t6_abl.to_markdown(os.path.join(TABLES_DIR, "table6a_ablations.md"), index=False)
    
    df_t6_fair.to_csv(os.path.join(TABLES_DIR, "table6b_fairness_audit.csv"), index=False)
    df_t6_fair.to_markdown(os.path.join(TABLES_DIR, "table6b_fairness_audit.md"), index=False)
    
    df_leak.to_csv(os.path.join(TABLES_DIR, "table6c_leakage_sensitivity.csv"), index=False)
    df_leak.to_markdown(os.path.join(TABLES_DIR, "table6c_leakage_sensitivity.md"), index=False)
    
    print("Table 6 (Ablations, Fairness, Leakage Audit) complete!", flush=True)

if __name__ == "__main__":
    run_table5_presentation_transfer()
    run_table6_ablations_and_fairness()
