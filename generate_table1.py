import pandas as pd
import numpy as np
import os

PROCESSED_DIR = r"..\data\processed"
TABLES_DIR = r"..\tables"
os.makedirs(TABLES_DIR, exist_ok=True)

def generate_table1():
    print("Generating Table 1: Cohort sizes, exclusions, and withdrawal prevalence by cutoff...")
    rows = []
    for week in [2, 4, 6, 8]:
        fpath = os.path.join(PROCESSED_DIR, f"cohort_week_{week}.csv")
        if not os.path.exists(fpath):
            print(f"File not found: {fpath}")
            continue
        df = pd.read_csv(fpath)
        
        cutoff_day = {2: 14, 4: 28, 6: 42, 8: 56}[week]
        
        # Partition masks
        train_mask = df['code_presentation'].isin(['2013B', '2013J'])
        val_mask = df['code_presentation'] == '2014B'
        test_mask = df['code_presentation'] == '2014J'
        
        train_students = set(df.loc[train_mask, 'id_student'])
        val_students = set(df.loc[val_mask, 'id_student'])
        seen_students = train_students | val_students
        test_unseen_mask = test_mask & (~df['id_student'].isin(seen_students))
        
        total_eligible = len(df)
        total_withdrawn = df['y_withdrawal'].sum()
        total_prev = (total_withdrawn / total_eligible) * 100
        
        train_n = train_mask.sum()
        train_w = df.loc[train_mask, 'y_withdrawal'].sum()
        train_prev = (train_w / train_n) * 100 if train_n > 0 else 0
        
        val_n = val_mask.sum()
        val_w = df.loc[val_mask, 'y_withdrawal'].sum()
        val_prev = (val_w / val_n) * 100 if val_n > 0 else 0
        
        test_unseen_n = test_unseen_mask.sum()
        test_unseen_w = df.loc[test_unseen_mask, 'y_withdrawal'].sum()
        test_unseen_prev = (test_unseen_w / test_unseen_n) * 100 if test_unseen_n > 0 else 0
        
        test_all_n = test_mask.sum()
        test_all_w = df.loc[test_mask, 'y_withdrawal'].sum()
        test_all_prev = (test_all_w / test_all_n) * 100 if test_all_n > 0 else 0
        
        rows.append({
            'Week': f"Week {week} (Day {cutoff_day})",
            'Total Eligible (N)': total_eligible,
            'Total Withdrawn': total_withdrawn,
            'Total Prevalence (%)': f"{total_prev:.2f}%",
            'Train N (2013B/J)': train_n,
            'Train Prev (%)': f"{train_prev:.2f}%",
            'Val N (2014B)': val_n,
            'Val Prev (%)': f"{val_prev:.2f}%",
            'Test Unseen N (2014J)': test_unseen_n,
            'Test Unseen Prev (%)': f"{test_unseen_prev:.2f}%",
            'Test All N (2014J)': test_all_n,
            'Test All Prev (%)': f"{test_all_prev:.2f}%"
        })
        
    t1_df = pd.DataFrame(rows)
    out_csv = os.path.join(TABLES_DIR, "table1_cohort_summary.csv")
    out_md = os.path.join(TABLES_DIR, "table1_cohort_summary.md")
    t1_df.to_csv(out_csv, index=False)
    t1_df.to_markdown(out_md, index=False)
    print("\n--- Table 1 Summary ---")
    print(t1_df.to_string(index=False))

if __name__ == "__main__":
    generate_table1()
