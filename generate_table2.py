import pandas as pd
import os

TABLES_DIR = r"..\tables"
os.makedirs(TABLES_DIR, exist_ok=True)

def generate_table2():
    print("Generating Table 2: Features and their availability rules...")
    features = [
        {
            "Category": "Study Background",
            "Feature": "studied_credits",
            "Operational Definition": "Total credit value of modules student is currently enrolled in",
            "Availability Rule": "Available at initial enrollment (fixed prior to Day 0)",
            "Leakage Prevention Rationale": "Fixed institutional record; completely invariant to in-course withdrawal timing"
        },
        {
            "Category": "Study Background",
            "Feature": "num_of_prev_attempts",
            "Operational Definition": "Number of previous attempts at the same module",
            "Availability Rule": "Available at initial enrollment (fixed prior to Day 0)",
            "Leakage Prevention Rationale": "Historical administrative log; unaffected by current semester actions"
        },
        {
            "Category": "Study Background",
            "Feature": "date_registration_imputed",
            "Operational Definition": "Days relative to presentation start when student registered (negative value)",
            "Availability Rule": "Available at presentation start (Day 0)",
            "Leakage Prevention Rationale": "Measures early vs. late registration; missing values imputed from training median"
        },
        {
            "Category": "Demographics",
            "Feature": "gender_binary, disability_binary, age_band_ord, education_ord, imd_band_ord",
            "Operational Definition": "Self-reported demographic & socio-economic decile attributes",
            "Availability Rule": "Recorded upon institutional matriculation",
            "Leakage Prevention Rationale": "Audited primarily for subgroup parity and algorithmic fairness; static across terms"
        },
        {
            "Category": "Cumulative Engagement",
            "Feature": "total_clicks",
            "Operational Definition": "Sum of all interaction clicks in the VLE up to cutoff day t",
            "Availability Rule": "Events with date <= t only",
            "Leakage Prevention Rationale": "Truncates future clicks; excludes any interaction post-dating the prediction cutoff"
        },
        {
            "Category": "Cumulative Engagement",
            "Feature": "active_days",
            "Operational Definition": "Count of distinct calendar days with >= 1 VLE interaction up to cutoff day t",
            "Availability Rule": "Events with date <= t only",
            "Leakage Prevention Rationale": "Measures longitudinal study habit frequency without observing future sessions"
        },
        {
            "Category": "Cumulative Engagement",
            "Feature": "clicks_per_active_day",
            "Operational Definition": "total_clicks / (active_days + 1e-5)",
            "Availability Rule": "Derived from events <= t",
            "Leakage Prevention Rationale": "Normalized study depth per session up to cutoff"
        },
        {
            "Category": "Cumulative Engagement",
            "Feature": "distinct_sites",
            "Operational Definition": "Count of distinct VLE activity resources accessed up to day t",
            "Availability Rule": "Events with date <= t only",
            "Leakage Prevention Rationale": "Curriculum breadth explored without looking forward"
        },
        {
            "Category": "Cumulative Engagement",
            "Feature": "pre_course_clicks & pre_course_active_days",
            "Operational Definition": "VLE clicks and active days strictly prior to course start (date < 0)",
            "Availability Rule": "Events with date < 0 only",
            "Leakage Prevention Rationale": "Pre-course orientation engagement, observable at Day 0"
        },
        {
            "Category": "Recent Engagement",
            "Feature": "clicks_last_7d & active_days_last_7d",
            "Operational Definition": "Clicks and active days in the rolling window [t-7, t]",
            "Availability Rule": "Events where (t-7) <= date <= t",
            "Leakage Prevention Rationale": "Captures immediate trailing momentum; strictly bound to cutoff"
        },
        {
            "Category": "Recent Engagement",
            "Feature": "clicks_last_14d & active_days_last_14d",
            "Operational Definition": "Clicks and active days in the rolling window [t-14, t]",
            "Availability Rule": "Events where (t-14) <= date <= t",
            "Leakage Prevention Rationale": "Medium-term recent participation intensity"
        },
        {
            "Category": "Recent Engagement",
            "Feature": "days_since_last_click",
            "Operational Definition": "Cutoff day t minus max(date) observed <= t",
            "Availability Rule": "Computed from events <= t",
            "Leakage Prevention Rationale": "Measures dormancy period without observing whether student returns post-cutoff"
        },
        {
            "Category": "Engagement Trajectory",
            "Feature": "click_ratio_7d_to_14d",
            "Operational Definition": "clicks_last_7d / (clicks_last_14d - clicks_last_7d + 1)",
            "Availability Rule": "Derived from [t-14, t] sub-windows",
            "Leakage Prevention Rationale": "Ratiometric acceleration/deceleration metric strictly bounded to the past fortnight"
        },
        {
            "Category": "Engagement Trajectory",
            "Feature": "recent_to_total_click_share",
            "Operational Definition": "clicks_last_14d / (total_clicks + 1)",
            "Availability Rule": "Derived from events <= t",
            "Leakage Prevention Rationale": "Relative decay of activity relative to student's own historical baseline"
        },
        {
            "Category": "Resource Modality",
            "Feature": "clicks_[forumng, oucontent, resource, subpage, url, quiz, homepage, other]",
            "Operational Definition": "VLE clicks partitioned across 8 functional LMS resource modalities <= t",
            "Availability Rule": "Events with date <= t mapped via vle.csv",
            "Leakage Prevention Rationale": "Captures pedagogical activity type diversity without forward lookahead"
        },
        {
            "Category": "Assessment Participation",
            "Feature": "assessments_due",
            "Operational Definition": "Count of module assessments scheduled to be submitted on or before cutoff day t",
            "Availability Rule": "Assessments with date <= t",
            "Leakage Prevention Rationale": "Scheduled syllabus requirements observable at cutoff day"
        },
        {
            "Category": "Assessment Participation",
            "Feature": "assessments_submitted",
            "Operational Definition": "Count of scheduled assessments actually submitted by the student by cutoff day t",
            "Availability Rule": "Submissions where date_submitted <= t",
            "Leakage Prevention Rationale": "Excludes submissions made after cutoff day; counts observable compliance"
        },
        {
            "Category": "Assessment Participation",
            "Feature": "missed_assessments",
            "Operational Definition": "max(0, assessments_due - assessments_submitted)",
            "Availability Rule": "Computed strictly at day t",
            "Leakage Prevention Rationale": "Clear non-compliance indicator observable in real time"
        },
        {
            "Category": "Assessment Participation",
            "Feature": "submission_rate",
            "Operational Definition": "assessments_submitted / assessments_due (1.0 if none due)",
            "Availability Rule": "Computed strictly at day t",
            "Leakage Prevention Rationale": "Proportional milestone completion rate"
        },
        {
            "Category": "Assessment Participation",
            "Feature": "mean_submission_delay & late_submissions_count",
            "Operational Definition": "Mean difference (date_submitted - date_due) and count of late submissions <= t",
            "Availability Rule": "Submissions with date_submitted <= t",
            "Leakage Prevention Rationale": "Behavioral punctuality; does not peek at future assignments"
        },
        {
            "Category": "Assessment Performance (Ablation Only)",
            "Feature": "mean_assessment_score",
            "Operational Definition": "Average percentage score across assessments with deadline <= t",
            "Availability Rule": "Evaluated ONLY in secondary ablation under zero-grading-latency assumption",
            "Leakage Prevention Rationale": "Excluded from primary model because OULAD lacks grade release timestamps"
        }
    ]
    
    t2_df = pd.DataFrame(features)
    out_csv = os.path.join(TABLES_DIR, "table2_feature_availability.csv")
    out_md = os.path.join(TABLES_DIR, "table2_feature_availability.md")
    t2_df.to_csv(out_csv, index=False)
    t2_df.to_markdown(out_md, index=False)
    print(f"Table 2 generated successfully ({len(t2_df)} features/groups documented)!")

if __name__ == "__main__":
    generate_table2()
