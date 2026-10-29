import pandas as pd
import numpy as np
import os

DATA_DIR = r"..\data"

def run_audit():
    print("=== OULAD DATA AUDIT ===")
    courses = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"), na_values=['?', ''])
    print("\n--- Courses / Presentations ---")
    print(courses.to_string())

    reg = pd.read_csv(os.path.join(DATA_DIR, "studentRegistration.csv"), na_values=['?', ''])
    reg['date_registration'] = pd.to_numeric(reg['date_registration'], errors='coerce')
    reg['date_unregistration'] = pd.to_numeric(reg['date_unregistration'], errors='coerce')

    print("\n--- Student Registration Summary ---")
    print(f"Total registration records: {len(reg):,}")
    print(f"Unique students: {reg['id_student'].nunique():,}")
    print(f"Missing date_registration: {reg['date_registration'].isna().sum():,}")
    print(f"Non-null date_unregistration (unregistered at some point): {reg['date_unregistration'].notna().sum():,}")
    print(f"Null date_unregistration (remained registered through end): {reg['date_unregistration'].isna().sum():,}")
    
    info = pd.read_csv(os.path.join(DATA_DIR, "studentInfo.csv"), na_values=['?', ''])
    print("\n--- Student Info Summary ---")
    print(f"Total studentInfo records: {len(info):,}")
    print("Final result distribution:")
    print(info['final_result'].value_counts())
    
    # Merge registration with studentInfo
    merged = pd.merge(info, reg, on=['code_module', 'code_presentation', 'id_student'], how='left')
    print("\nCross-tabulation of final_result vs date_unregistration.notna():")
    print(pd.crosstab(merged['final_result'], merged['date_unregistration'].notna(), 
                      rownames=['final_result'], colnames=['has_unreg_date']))

    print("\nAudit of date_unregistration values for Withdrawn students:")
    withdrawn = merged[merged['final_result'] == 'Withdrawn']
    print(f"Total 'Withdrawn': {len(withdrawn)}")
    print(f"Withdrawn with null date_unregistration: {withdrawn['date_unregistration'].isna().sum()}")
    print("date_unregistration summary for Withdrawn:")
    print(withdrawn['date_unregistration'].describe())

    print("\nNegative date_unregistration (unregistered before course start day 0):")
    neg_unreg = merged[merged['date_unregistration'] < 0]
    print(f"Count with date_unregistration < 0: {len(neg_unreg)}")
    print("Their final_result breakdown:")
    print(neg_unreg['final_result'].value_counts())

    print("\nStudents with date_unregistration == 0:")
    zero_unreg = merged[merged['date_unregistration'] == 0]
    print(f"Count: {len(zero_unreg)}")

    # Check non-withdrawn students with date_unregistration
    non_withdrawn_unreg = merged[(merged['final_result'] != 'Withdrawn') & (merged['date_unregistration'].notna())]
    print(f"\nNon-withdrawn students who have a date_unregistration: {len(non_withdrawn_unreg)}")
    if len(non_withdrawn_unreg) > 0:
        print(non_withdrawn_unreg[['final_result', 'date_unregistration']].head())

    # Check Withdrawn students without date_unregistration
    withdrawn_no_unreg = merged[(merged['final_result'] == 'Withdrawn') & (merged['date_unregistration'].isna())]
    print(f"\nWithdrawn students who have NO date_unregistration: {len(withdrawn_no_unreg)}")

    # Cohort analysis for Cutoffs: Days 14 (W2), 28 (W4), 42 (W6), 56 (W8)
    # Rules:
    # 1. Registered before or on cutoff: (date_registration <= cutoff) | (date_registration.isna())
    # 2. Exclude students who unregistered on or before cutoff: date_unregistration <= cutoff
    # 3. Eligible cohort: date_unregistration is null OR date_unregistration > cutoff
    # 4. Target label (Withdrawal after cutoff): date_unregistration > cutoff (or final_result == 'Withdrawn' with date_unregistration > cutoff)
    
    print("\n================ COHORT BREAKDOWN BY CUTOFF (TABLE 1 AUDIT) ================")
    cutoffs = [14, 28, 42, 56]
    for c in cutoffs:
        # Initial enrolled count in presentations
        total_enrolled = len(merged)
        
        # Excluded before start (date_unregistration < 0)
        unreg_before_start = (merged['date_unregistration'] < 0).sum()
        
        # Excluded between day 0 and cutoff day c: 0 <= date_unregistration <= c
        unreg_0_to_c = ((merged['date_unregistration'] >= 0) & (merged['date_unregistration'] <= c)).sum()
        
        # Total excluded prior to cutoff
        total_excluded_early = (merged['date_unregistration'] <= c).sum()
        
        # Missing date_registration audit:
        # In OULAD, some students have missing date_registration, but did they stay or withdraw?
        missing_reg = merged['date_registration'].isna().sum()
        
        # Eligible students at cutoff c:
        # Enrolled students who have NOT withdrawn by cutoff c
        eligible = merged[merged['date_unregistration'].isna() | (merged['date_unregistration'] > c)]
        
        # Future withdrawals (positive label y=1):
        # Among eligible, how many withdraw later before end of course?
        future_withdrawn = eligible[eligible['date_unregistration'].notna()] # All have date_unregistration > c
        future_withdrawn_count = len(future_withdrawn)
        
        # Non-withdrawals (negative label y=0):
        # Completed the course (Pass, Distinction, Fail)
        retained_count = len(eligible) - future_withdrawn_count
        
        prevalence = (future_withdrawn_count / len(eligible)) * 100
        
        print(f"\n--- Cutoff Week {c // 7} (Day {c}) ---")
        print(f"Total enrollment records: {total_enrolled:,}")
        print(f"Excluded: Withdrew before course start (day < 0): {unreg_before_start:,}")
        print(f"Excluded: Withdrew between day 0 and day {c}: {unreg_0_to_c:,}")
        print(f"Total excluded prior to cutoff: {total_excluded_early:,}")
        print(f"Eligible cohort size (N): {len(eligible):,}")
        print(f"Future withdrawals (y = 1): {future_withdrawn_count:,} ({prevalence:.2f}%)")
        print(f"Retained students (y = 0): {retained_count:,} ({100 - prevalence:.2f}%)")

if __name__ == "__main__":
    run_audit()
