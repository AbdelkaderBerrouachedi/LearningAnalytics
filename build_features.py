import pandas as pd
import numpy as np
import os
import time

DATA_DIR = r"..\data"
OUTPUT_DIR = r"..\data\processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CUTOFFS = {
    2: 14,
    4: 28,
    6: 42,
    8: 56
}

def load_base_data():
    print("Loading base CSVs...")
    courses = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"), na_values=['?', ''])
    reg = pd.read_csv(os.path.join(DATA_DIR, "studentRegistration.csv"), na_values=['?', ''])
    info = pd.read_csv(os.path.join(DATA_DIR, "studentInfo.csv"), na_values=['?', ''])
    vle = pd.read_csv(os.path.join(DATA_DIR, "vle.csv"), na_values=['?', ''])
    assessments = pd.read_csv(os.path.join(DATA_DIR, "assessments.csv"), na_values=['?', ''])
    student_assessment = pd.read_csv(os.path.join(DATA_DIR, "studentAssessment.csv"), na_values=['?', ''])
    
    reg['date_registration'] = pd.to_numeric(reg['date_registration'], errors='coerce')
    reg['date_unregistration'] = pd.to_numeric(reg['date_unregistration'], errors='coerce')
    assessments['date'] = pd.to_numeric(assessments['date'], errors='coerce')
    student_assessment['date_submitted'] = pd.to_numeric(student_assessment['date_submitted'], errors='coerce')
    student_assessment['score'] = pd.to_numeric(student_assessment['score'], errors='coerce')

    # Merge student base
    merged = pd.merge(info, reg, on=['code_module', 'code_presentation', 'id_student'], how='left')
    merged = pd.merge(merged, courses[['code_module', 'code_presentation', 'module_presentation_length']], 
                      on=['code_module', 'code_presentation'], how='left')
    
    # Audit exclusions:
    # 1. Unregistered before day 0 (date_unregistration < 0)
    # 2. Withdrawn without date_unregistration (93 records)
    valid_mask = ~((merged['date_unregistration'] < 0) | 
                   ((merged['final_result'] == 'Withdrawn') & merged['date_unregistration'].isna()))
    
    base_students = merged[valid_mask].copy()
    print(f"Total valid students in baseline pool: {len(base_students):,}")
    
    # Preload studentVle
    print("Loading studentVle.csv (this may take ~10s)...")
    t0 = time.time()
    svle = pd.read_csv(os.path.join(DATA_DIR, "studentVle.csv"), na_values=['?', ''],
                       usecols=['code_module', 'code_presentation', 'id_student', 'id_site', 'date', 'sum_click'])
    print(f"studentVle loaded ({len(svle):,} rows) in {time.time() - t0:.2f}s")
    
    # Map id_site to activity_type
    site_type_map = dict(zip(vle['id_site'], vle['activity_type']))
    svle['activity_type'] = svle['id_site'].map(site_type_map).fillna('other')
    
    return base_students, svle, assessments, student_assessment

def extract_features_for_cutoff(week_num, cutoff_day, base_students, svle, assessments, student_assessment):
    print(f"\n=======================================================")
    print(f"Extracting features for Week {week_num} (Day {cutoff_day})...")
    print(f"=======================================================")
    
    # 1. Filter eligible students at cutoff_day
    # Student must not have unregistered on or before cutoff_day
    eligible = base_students[base_students['date_unregistration'].isna() | 
                             (base_students['date_unregistration'] > cutoff_day)].copy()
    
    # Target variable y: 1 if withdrew after cutoff_day, 0 otherwise
    eligible['y_withdrawal'] = (eligible['date_unregistration'] > cutoff_day).astype(int)
    
    print(f"Eligible cohort size: {len(eligible):,}")
    print(f"Withdrawals after Day {cutoff_day}: {eligible['y_withdrawal'].sum():,} ({eligible['y_withdrawal'].mean()*100:.2f}%)")
    
    # 2. VLE Features strictly <= cutoff_day
    t0 = time.time()
    vle_cutoff = svle[svle['date'] <= cutoff_day].copy()
    print(f"VLE interactions <= Day {cutoff_day}: {len(vle_cutoff):,} (filtered in {time.time()-t0:.2f}s)")
    
    # Aggregations by student enrollment
    # Total clicks & active days
    agg_all = vle_cutoff.groupby(['code_module', 'code_presentation', 'id_student']).agg(
        total_clicks=('sum_click', 'sum'),
        active_days=('date', 'nunique'),
        distinct_sites=('id_site', 'nunique'),
        last_click_day=('date', 'max')
    ).reset_index()
    agg_all['days_since_last_click'] = cutoff_day - agg_all['last_click_day']
    agg_all['clicks_per_active_day'] = agg_all['total_clicks'] / (agg_all['active_days'] + 1e-5)
    
    # Pre-course activity (date < 0)
    pre_vle = vle_cutoff[vle_cutoff['date'] < 0]
    agg_pre = pre_vle.groupby(['code_module', 'code_presentation', 'id_student']).agg(
        pre_course_clicks=('sum_click', 'sum'),
        pre_course_active_days=('date', 'nunique')
    ).reset_index()
    
    # Last 7 days [cutoff_day - 7, cutoff_day]
    last_7d = vle_cutoff[(vle_cutoff['date'] >= cutoff_day - 7) & (vle_cutoff['date'] <= cutoff_day)]
    agg_7d = last_7d.groupby(['code_module', 'code_presentation', 'id_student']).agg(
        clicks_last_7d=('sum_click', 'sum'),
        active_days_last_7d=('date', 'nunique')
    ).reset_index()
    
    # Last 14 days [cutoff_day - 14, cutoff_day]
    last_14d = vle_cutoff[(vle_cutoff['date'] >= cutoff_day - 14) & (vle_cutoff['date'] <= cutoff_day)]
    agg_14d = last_14d.groupby(['code_module', 'code_presentation', 'id_student']).agg(
        clicks_last_14d=('sum_click', 'sum'),
        active_days_last_14d=('date', 'nunique')
    ).reset_index()
    
    # Activity types breakdown
    top_types = ['forumng', 'oucontent', 'resource', 'subpage', 'url', 'quiz', 'homepage']
    vle_cutoff['act_cat'] = vle_cutoff['activity_type'].apply(lambda x: x if x in top_types else 'other')
    type_pivot = vle_cutoff.pivot_table(index=['code_module', 'code_presentation', 'id_student'],
                                        columns='act_cat', values='sum_click', aggfunc='sum', fill_value=0).reset_index()
    type_cols = [f"clicks_{col}" for col in type_pivot.columns if col not in ['code_module', 'code_presentation', 'id_student']]
    type_pivot.columns = ['code_module', 'code_presentation', 'id_student'] + type_cols
    
    # 3. Assessment Features strictly <= cutoff_day
    # Find assessments due <= cutoff_day
    due_assess = assessments[assessments['date'].notna() & (assessments['date'] <= cutoff_day)].copy()
    
    # Course level due count
    course_due_counts = due_assess.groupby(['code_module', 'code_presentation'])['id_assessment'].count().rename('assessments_due')
    
    # Student submissions on due assessments with date_submitted <= cutoff_day
    merged_sa = pd.merge(student_assessment[student_assessment['date_submitted'] <= cutoff_day],
                         due_assess[['id_assessment', 'code_module', 'code_presentation', 'date', 'weight']],
                         on='id_assessment', how='inner')
    
    merged_sa['delay'] = merged_sa['date_submitted'] - merged_sa['date']
    merged_sa['is_late'] = (merged_sa['delay'] > 0).astype(int)
    
    agg_assess = merged_sa.groupby(['code_module', 'code_presentation', 'id_student']).agg(
        assessments_submitted=('id_assessment', 'count'),
        mean_submission_delay=('delay', 'mean'),
        late_submissions_count=('is_late', 'sum'),
        mean_assessment_score=('score', 'mean') # secondary feature for ablation
    ).reset_index()

    # 4. Merge all features into eligible dataframe
    df = pd.merge(eligible, agg_all, on=['code_module', 'code_presentation', 'id_student'], how='left')
    df = pd.merge(df, agg_pre, on=['code_module', 'code_presentation', 'id_student'], how='left')
    df = pd.merge(df, agg_7d, on=['code_module', 'code_presentation', 'id_student'], how='left')
    df = pd.merge(df, agg_14d, on=['code_module', 'code_presentation', 'id_student'], how='left')
    df = pd.merge(df, type_pivot, on=['code_module', 'code_presentation', 'id_student'], how='left')
    df = pd.merge(df, course_due_counts, on=['code_module', 'code_presentation'], how='left')
    df = pd.merge(df, agg_assess, on=['code_module', 'code_presentation', 'id_student'], how='left')
    
    # Fill NAs for activity / counts
    df['total_clicks'] = df['total_clicks'].fillna(0)
    df['active_days'] = df['active_days'].fillna(0)
    df['clicks_per_active_day'] = df['clicks_per_active_day'].fillna(0)
    df['distinct_sites'] = df['distinct_sites'].fillna(0)
    df['days_since_last_click'] = df['days_since_last_click'].fillna(cutoff_day + 30)
    df['pre_course_clicks'] = df['pre_course_clicks'].fillna(0)
    df['pre_course_active_days'] = df['pre_course_active_days'].fillna(0)
    df['clicks_last_7d'] = df['clicks_last_7d'].fillna(0)
    df['active_days_last_7d'] = df['active_days_last_7d'].fillna(0)
    df['clicks_last_14d'] = df['clicks_last_14d'].fillna(0)
    df['active_days_last_14d'] = df['active_days_last_14d'].fillna(0)
    
    for c in type_cols:
        df[c] = df[c].fillna(0)
        
    df['assessments_due'] = df['assessments_due'].fillna(0)
    df['assessments_submitted'] = df['assessments_submitted'].fillna(0)
    df['missed_assessments'] = (df['assessments_due'] - df['assessments_submitted']).clip(lower=0)
    df['submission_rate'] = np.where(df['assessments_due'] > 0, 
                                     df['assessments_submitted'] / df['assessments_due'], 
                                     1.0)
    df['mean_submission_delay'] = df['mean_submission_delay'].fillna(0)
    df['late_submissions_count'] = df['late_submissions_count'].fillna(0)
    df['mean_assessment_score'] = df['mean_assessment_score'].fillna(-1.0) # -1 for unassessed
    
    # Derived momentum/trend features
    df['click_ratio_7d_to_14d'] = df['clicks_last_7d'] / (df['clicks_last_14d'] - df['clicks_last_7d'] + 1.0)
    df['active_day_ratio_7d_to_14d'] = df['active_days_last_7d'] / (df['active_days_last_14d'] - df['active_days_last_7d'] + 1.0)
    df['recent_to_total_click_share'] = df['clicks_last_14d'] / (df['total_clicks'] + 1.0)
    
    # Static encodings
    df['gender_binary'] = (df['gender'] == 'F').astype(int)
    df['disability_binary'] = (df['disability'] == 'Y').astype(int)
    
    age_map = {'0-35': 1, '35-55': 2, '55<=': 3}
    df['age_band_ord'] = df['age_band'].map(age_map).fillna(1)
    
    edu_map = {
        'No Formal quals': 0,
        'Lower Than A Level': 1,
        'A Level or Equivalent': 2,
        'HE Qualification': 3,
        'Post Graduate Qualification': 4
    }
    df['education_ord'] = df['highest_education'].map(edu_map).fillna(2)
    
    imd_map = {
        '0-10%': 0, '10-20': 1, '20-30%': 2, '30-40%': 3, '40-50%': 4,
        '50-60%': 5, '60-70%': 6, '70-80%': 7, '80-90%': 8, '90-100%': 9
    }
    df['imd_band_ord'] = df['imd_band'].map(imd_map).fillna(4.5)
    df['date_registration_imputed'] = df['date_registration'].fillna(df['date_registration'].median())

    # Save to disk
    out_file = os.path.join(OUTPUT_DIR, f"cohort_week_{week_num}.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df):,} records to {out_file} ({os.path.getsize(out_file)/(1024*1024):.2f} MB)")
    return df

def main():
    base_students, svle, assessments, student_assessment = load_base_data()
    for w, c in CUTOFFS.items():
        extract_features_for_cutoff(w, c, base_students, svle, assessments, student_assessment)
    print("\nFeature extraction complete for all weeks (2, 4, 6, 8)!")

if __name__ == "__main__":
    main()
