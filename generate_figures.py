import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import json
import os
from sklearn.calibration import calibration_curve
import shap
import lightgbm as lgb

FIGURES_DIR = r"..\figures"
TABLES_DIR = r"..\tables"
PROCESSED_DIR = r"..\data\processed"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Publication style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

def plot_figure1_pipeline():
    print("Generating Figure 1: Data and prediction pipeline...", flush=True)
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.axis('off')
    
    # Draw boxes
    boxes = [
        {"x": 0.02, "y": 0.55, "w": 0.16, "h": 0.35, "title": "1. Multi-Modal\nRaw Data\n(OULAD)", "desc": "• VLE clickstream (10.6M)\n• Course registries\n• TMA submission logs\n• Static demographics", "color": "#E3F2FD", "edge": "#1976D2"},
        {"x": 0.22, "y": 0.55, "w": 0.17, "h": 0.35, "title": "2. Point-in-Time\nEligibility Filter", "desc": "• Cutoffs: Days 14, 28, 42, 56\n• Exclude early drops (<= t)\n• Strict cohort at risk\n• Outcome: drop > t", "color": "#FFF3E0", "edge": "#F57C00"},
        {"x": 0.43, "y": 0.55, "w": 0.18, "h": 0.35, "title": "3. Anti-Leakage\nFeature Extraction", "desc": "• Events strictly <= t\n• Inactivity recency\n• Fortnight velocity\n• TMA submission delay", "color": "#E8F5E9", "edge": "#388E3C"},
        {"x": 0.65, "y": 0.55, "w": 0.16, "h": 0.35, "title": "4. Chronological\nValidation Split", "desc": "• Train: 2013B/J (N=12k)\n• Val: 2014B (N=6.6k)\n• Test: 2014J unseen (N=7.9k)\n• Zero student overlap", "color": "#F3E5F5", "edge": "#7B1FA2"},
        {"x": 0.85, "y": 0.55, "w": 0.13, "h": 0.35, "title": "5. Decision\nSupport", "desc": "• Isotonic Calib.\n• Top 5/10/20%\n• Proactive outreach\n• SHAP audibility", "color": "#FFEBEE", "edge": "#D32F2F"},
    ]
    
    for b in boxes:
        rect = patches.FancyBboxPatch((b['x'], b['y']), b['w'], b['h'],
                                      boxstyle="round,pad=0.015,rounding_size=0.02",
                                      facecolor=b['color'], edgecolor=b['edge'], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(b['x'] + b['w']/2, b['y'] + b['h'] - 0.08, b['title'],
                ha='center', va='center', weight='bold', color='#1A237E', fontsize=9.5)
        ax.text(b['x'] + 0.01, b['y'] + 0.12, b['desc'],
                ha='left', va='center', color='#333333', fontsize=7.5, linespacing=1.3)
        
    # Arrows
    arrow_props = dict(facecolor='#555555', edgecolor='#555555', arrowstyle='->', lw=1.5)
    ax.annotate('', xy=(0.21, 0.72), xytext=(0.19, 0.72), arrowprops=arrow_props)
    ax.annotate('', xy=(0.42, 0.72), xytext=(0.40, 0.72), arrowprops=arrow_props)
    ax.annotate('', xy=(0.64, 0.72), xytext=(0.62, 0.72), arrowprops=arrow_props)
    ax.annotate('', xy=(0.84, 0.72), xytext=(0.82, 0.72), arrowprops=arrow_props)
    
    # Sub-caption banner at bottom
    banner = patches.FancyBboxPatch((0.02, 0.1), 0.96, 0.35,
                                    boxstyle="round,pad=0.01,rounding_size=0.015",
                                    facecolor="#FAFAFA", edgecolor="#CCCCCC", linewidth=1.0)
    ax.add_patch(banner)
    summary_text = (
        "Methodological Safeguards Enforced:\n"
        "• Strict Zero-Leakage: Zero target information or post-cutoff interaction observable in feature space.\n"
        "• Point-in-Time Risk Sets: Denominator updated dynamically at each cutoff; pre-cutoff withdrawals removed.\n"
        "• Temporal Generalization: Evaluated on future chronological cohorts with complete student identity separation.\n"
        "• Realistic Decision Framing: Assesses precision and recall under operational advisor contact budgets (5%, 10%, 20%)."
    )
    ax.text(0.04, 0.27, summary_text, ha='left', va='center', color='#263238', fontsize=8.2, linespacing=1.4)
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig1_pipeline.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 1 to {out_path}", flush=True)

def plot_figure2_chronological_design():
    print("Generating Figure 2: Chronological evaluation design...", flush=True)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    
    # Presentations timeline
    presentations = [
        {"name": "2013B (Feb 2013)", "role": "Training Set\n(Historical 1)", "color": "#90CAF9", "y": 3, "start": 0, "len": 8},
        {"name": "2013J (Oct 2013)", "role": "Training Set\n(Historical 2)", "color": "#64B5F6", "y": 2, "start": 3, "len": 8},
        {"name": "2014B (Feb 2014)", "role": "Validation Set\n(Hyperparameter Tuning & Calibration)", "color": "#FFB74D", "y": 1, "start": 7, "len": 8},
        {"name": "2014J (Oct 2014)", "role": "Held-Out Test Set\n(Zero-Leakage Unseen Students)", "color": "#81C784", "y": 0, "start": 10, "len": 8},
    ]
    
    for p in presentations:
        rect = patches.FancyBboxPatch((p['start'], p['y'] - 0.35), p['len'], 0.7,
                                      boxstyle="round,pad=0.02,rounding_size=0.1",
                                      facecolor=p['color'], edgecolor="#333333", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(p['start'] + 0.3, p['y'], p['name'], ha='left', va='center', weight='bold', fontsize=9.5)
        ax.text(p['start'] + p['len'] - 0.3, p['y'], p['role'], ha='right', va='center', style='italic', fontsize=8)
        
    # Overlap arrow and warning
    ax.annotate("Identity Leakage Control:\n1,329 recurring students purged\nfrom Test Partition",
                xy=(10.5, 0.4), xytext=(6.5, -0.6),
                arrowprops=dict(facecolor='#D32F2F', edgecolor='#D32F2F', arrowstyle='->', lw=1.5),
                ha='center', va='center', weight='bold', color='#D32F2F', fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#D32F2F", lw=1))
    
    ax.set_xlim(-1, 20)
    ax.set_ylim(-1.2, 4)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Phase 4: Test', 'Phase 3: Val', 'Phase 2: Train', 'Phase 1: Train'], weight='bold')
    ax.set_xlabel('Relative Calendar Progression (2013 - 2014 Presentations)', weight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig2_chronological_design.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2 to {out_path}", flush=True)

def plot_figure3_performance():
    print("Generating Figure 3: Performance vs Prediction Week...", flush=True)
    t3_path = os.path.join(TABLES_DIR, "table3_model_performance.csv")
    t4_path = os.path.join(TABLES_DIR, "table4_contact_budgets.csv")
    if not os.path.exists(t3_path):
        print("Table 3 not found yet.")
        return
        
    df_t3 = pd.read_csv(t3_path)
    df_t4 = pd.read_csv(t4_path)
    
    # Extract numeric values
    def parse_ci(val_str):
        parts = val_str.split('[')
        mean_val = float(parts[0].strip().replace('%', ''))
        ci_parts = parts[1].replace(']', '').split(',')
        low = float(ci_parts[0].strip().replace('%', ''))
        high = float(ci_parts[1].strip().replace('%', ''))
        return mean_val, low, high

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    
    weeks = [2, 4, 6, 8]
    models = ['LightGBM (GBDT)', 'XGBoost', 'Logistic Regression', 'Activity Heuristic Rule']
    colors = {'LightGBM (GBDT)': '#1565C0', 'XGBoost': '#2E7D32', 'Logistic Regression': '#E65100', 'Activity Heuristic Rule': '#757575'}
    markers = {'LightGBM (GBDT)': 'o', 'XGBoost': 's', 'Logistic Regression': '^', 'Activity Heuristic Rule': 'd'}
    
    # Panel A: PR-AUC vs Week
    for m in models:
        m_df = df_t3[df_t3['Model'] == m]
        means, lows, highs = [], [], []
        for _, row in m_df.iterrows():
            mu, lo, hi = parse_ci(row['PR-AUC'])
            means.append(mu)
            lows.append(lo)
            highs.append(hi)
            
        err_low = np.array(means) - np.array(lows)
        err_high = np.array(highs) - np.array(means)
        ax1.errorbar(weeks, means, yerr=[err_low, err_high], label=m, color=colors[m],
                     marker=markers[m], markersize=6, capsize=3, lw=1.8)
        
    # Prevalence line
    ax1.axhline(0.16, color='black', linestyle=':', label='Prevalence Baseline (~0.16)')
    ax1.set_title('A: Precision-Recall AUC (Primary Metric)', weight='bold')
    ax1.set_xlabel('Prediction Timing (Cutoff Week)', weight='bold')
    ax1.set_ylabel('PR-AUC (Average Precision)', weight='bold')
    ax1.set_xticks(weeks)
    ax1.set_xticklabels(['Week 2\n(Day 14)', 'Week 4\n(Day 28)', 'Week 6\n(Day 42)', 'Week 8\n(Day 56)'])
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True)
    
    # Panel B: Recall at Top 10% Budget vs Week
    for m in models:
        m_df = df_t4[df_t4['Model'] == m]
        means, lows, highs = [], [], []
        for _, row in m_df.iterrows():
            mu, lo, hi = parse_ci(row['Rec@10%'])
            means.append(mu)
            lows.append(lo)
            highs.append(hi)
            
        err_low = np.array(means) - np.array(lows)
        err_high = np.array(highs) - np.array(means)
        ax2.errorbar(weeks, means, yerr=[err_low, err_high], label=m, color=colors[m],
                     marker=markers[m], markersize=6, capsize=3, lw=1.8)
        
    # Random selection baseline at 10% budget is 10%
    ax2.axhline(10.0, color='black', linestyle=':', label='Random Flagging (10%)')
    ax2.set_title('B: Recall @ Top 10% Advisor Contact Budget', weight='bold')
    ax2.set_xlabel('Prediction Timing (Cutoff Week)', weight='bold')
    ax2.set_ylabel('Recall (% of Future Withdrawals Caught)', weight='bold')
    ax2.set_xticks(weeks)
    ax2.set_xticklabels(['Week 2\n(Day 14)', 'Week 4\n(Day 28)', 'Week 6\n(Day 42)', 'Week 8\n(Day 56)'])
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig3_performance_vs_week.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 3 to {out_path}", flush=True)

def plot_figure4_calibration():
    print("Generating Figure 4: Probability calibration plots...", flush=True)
    pred_path = os.path.join(TABLES_DIR, "predictions.json")
    if not os.path.exists(pred_path):
        print("predictions.json not found yet.")
        return
        
    with open(pred_path, 'r') as f:
        preds = json.load(f)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    
    for ax, w_key, title in zip([ax1, ax2], ['week_4', 'week_8'], ['A: Week 4 (Day 28)', 'B: Week 8 (Day 56)']):
        data = preds[w_key]
        y_true = np.array(data['y_test'])
        
        # LightGBM raw & cal
        prob_raw = np.array(data['prob_lgb_raw'])
        prob_cal = np.array(data['prob_lgb_cal'])
        
        # Logistic Regression cal
        prob_lr = np.array(data['prob_lr_cal'])
        
        # Fraction of positives vs mean predicted value
        fop_raw, mpv_raw = calibration_curve(y_true, prob_raw, n_bins=10)
        fop_cal, mpv_cal = calibration_curve(y_true, prob_cal, n_bins=10)
        fop_lr, mpv_lr = calibration_curve(y_true, prob_lr, n_bins=10)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
        ax.plot(mpv_raw, fop_raw, 's-', color='#E53935', label='LightGBM (Uncalibrated)', lw=1.6)
        ax.plot(mpv_cal, fop_cal, 'o-', color='#1E88E5', label='LightGBM (Isotonic Calibrated)', lw=1.8)
        ax.plot(mpv_lr, fop_lr, '^-', color='#43A047', label='Logistic Reg. (Platt Calibrated)', lw=1.6)
        
        ax.set_title(title, weight='bold')
        ax.set_xlabel('Mean Predicted Withdrawal Probability', weight='bold')
        ax.set_ylabel('Empirical Observed Withdrawal Rate', weight='bold')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper left', frameon=True)
        
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig4_calibration_curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 4 to {out_path}", flush=True)

def plot_figure5_shap():
    print("Generating Figure 5: SHAP explanations and stability across presentations...", flush=True)
    fpath4 = os.path.join(PROCESSED_DIR, "cohort_week_4.csv")
    df4 = pd.read_csv(fpath4)
    
    feats = [
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
    
    train_mask = df4['code_presentation'].isin(['2013B', '2013J'])
    test_mask = df4['code_presentation'] == '2014J'
    seen = set(df4.loc[train_mask, 'id_student']) | set(df4.loc[df4['code_presentation'] == '2014B', 'id_student'])
    test_unseen = test_mask & (~df4['id_student'].isin(seen))
    
    X_train = df4.loc[train_mask, feats].values
    y_train = df4.loc[train_mask, 'y_withdrawal'].values
    X_test = df4.loc[test_unseen, feats].values
    
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    # Subsample 1000 test cases for fast computing
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_test), min(1000, len(X_test)), replace=False)
    X_sample = X_test[sample_idx]
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1] # positive class
    else:
        shap_vals = shap_values
        
    # Mean absolute SHAP
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:12]
    
    top_feats = [feats[i] for i in top_indices]
    top_shap = mean_abs_shap[top_indices]
    
    # Feature name cleanup for readability
    clean_names = {
        'days_since_last_click': 'Days Inactive (Dormancy)',
        'clicks_last_14d': 'Clicks in Last 14 Days',
        'active_days_last_14d': 'Active Days in Last 14 Days',
        'total_clicks': 'Total Cumulative Clicks',
        'active_days': 'Total Active Days',
        'date_registration_imputed': 'Registration Lead Time',
        'studied_credits': 'Enrolled Credit Load',
        'distinct_sites': 'Distinct VLE Sites Visited',
        'clicks_oucontent': 'Clicks on Course Content',
        'clicks_homepage': 'Clicks on Homepage',
        'click_ratio_7d_to_14d': 'Fortnight Activity Velocity',
        'recent_to_total_click_share': 'Recent Activity Share',
        'missed_assessments': 'Missed Scheduled TMAs',
        'submission_rate': 'TMA Submission Rate'
    }
    plot_names = [clean_names.get(f, f) for f in top_feats]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # Panel A: Bar chart of global importance
    y_pos = np.arange(len(plot_names))
    ax1.barh(y_pos, top_shap[::-1], color='#1976D2', edgecolor='#0D47A1', height=0.65)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(plot_names[::-1], weight='bold')
    ax1.set_xlabel('Mean |SHAP Value| (Impact on Model Log-Odds)', weight='bold')
    ax1.set_title('A: Global Predictor Importance (Week 4)', weight='bold')
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Panel B: Presentation stability comparison (Train 2013 vs Test 2014J)
    train_sample_idx = np.random.choice(len(X_train), min(1000, len(X_train)), replace=False)
    shap_train = explainer.shap_values(X_train[train_sample_idx])
    if isinstance(shap_train, list):
        shap_tr = shap_train[1]
    else:
        shap_tr = shap_train
    mean_tr = np.mean(np.abs(shap_tr), axis=0)[top_indices]
    
    width = 0.35
    ax2.barh(y_pos + width/2, mean_tr[::-1], width, label='Training Cohort (2013B/J)', color='#42A5F5', alpha=0.85)
    ax2.barh(y_pos - width/2, top_shap[::-1], width, label='Held-Out Test Cohort (2014J)', color='#2E7D32', alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(plot_names[::-1], weight='bold')
    ax2.set_xlabel('Mean |SHAP Value|', weight='bold')
    ax2.set_title('B: Predictor Stability Across Cohorts', weight='bold')
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', frameon=True)
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig5_shap_stability.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 5 to {out_path}", flush=True)

def main():
    plot_figure1_pipeline()
    plot_figure2_chronological_design()
    plot_figure3_performance()
    plot_figure4_calibration()
    plot_figure5_shap()
    print("\nAll figures generated successfully in E:\\J3_IJIET\\figures\\", flush=True)

if __name__ == "__main__":
    main()
