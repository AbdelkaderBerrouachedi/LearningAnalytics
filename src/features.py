"""
Point-in-Time Feature Engineering Pipeline with Strict Zero-Leakage Firewall.
Restricts all predictors strictly to information timestamped <= cutoff day t.
"""
from pathlib import Path
import time
import pandas as pd
import numpy as np

def extract_features_for_cutoff(
    week_num: int,
    cutoff_day: int,
    base_students: pd.DataFrame,
    svle: pd.DataFrame,
    assessments: pd.DataFrame,
    student_assessment: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract observable features up to cutoff_day, enforcing zero target leakage.
    Target y: 1 if student unregistered after cutoff_day, 0 if completed course.
    """
    print(f"\n[Feature Pipeline] Extracting Week {week_num} (Cutoff Day {cutoff_day})...")
    t_start = time.time()

    # 1. Point-in-Time Eligible Cohort (Zero-Leakage Active Risk Set)
    # Strictly require student to have registered on or before cutoff_day (or date missing pre-course)
    # and not unregistered on or before cutoff_day.
    is_registered = base_students["date_registration"].isna() | (base_students["date_registration"] <= cutoff_day)
    is_not_unregistered = base_students["date_unregistration"].isna() | (base_students["date_unregistration"] > cutoff_day)
    eligible = base_students[is_registered & is_not_unregistered].copy()

    # Outcome definition: student unregisters after cutoff_day and within presentation duration
    is_withdrawn = (eligible["date_unregistration"] > cutoff_day)
    if "module_presentation_length" in eligible.columns:
        is_withdrawn = is_withdrawn & (eligible["date_unregistration"] <= eligible["module_presentation_length"])
    eligible["y_withdrawal"] = is_withdrawn.astype(int)
    print(f"  - Active eligible students: {len(eligible):,}")
    print(f"  - Subsequent withdrawals: {eligible['y_withdrawal'].sum():,} ({eligible['y_withdrawal'].mean()*100:.2f}%)")

    # 2. VLE Clickstream Features (strictly date <= cutoff_day)
    if not svle.empty:
        vle_cutoff = svle[svle["date"] <= cutoff_day].copy()
        
        # Cumulative
        agg_cum = vle_cutoff.groupby(["code_module", "code_presentation", "id_student"]).agg(
            total_clicks=("sum_click", "sum"),
            active_days=("date", "nunique"),
            distinct_sites=("id_site", "nunique"),
            max_click_date=("date", "max")
        ).reset_index()

        # Pre-course (date < 0)
        vle_pre = vle_cutoff[vle_cutoff["date"] < 0]
        agg_pre = vle_pre.groupby(["code_module", "code_presentation", "id_student"]).agg(
            pre_course_clicks=("sum_click", "sum"),
            pre_course_active_days=("date", "nunique")
        ).reset_index()

        # Last 7 days [cutoff_day - 6, cutoff_day] (7 calendar days inclusive)
        vle_7d = vle_cutoff[vle_cutoff["date"] >= (cutoff_day - 6)]
        agg_7d = vle_7d.groupby(["code_module", "code_presentation", "id_student"]).agg(
            clicks_last_7d=("sum_click", "sum"),
            active_days_last_7d=("date", "nunique")
        ).reset_index()

        # Last 14 days [cutoff_day - 13, cutoff_day] (14 calendar days inclusive)
        vle_14d = vle_cutoff[vle_cutoff["date"] >= (cutoff_day - 13)]
        agg_14d = vle_14d.groupby(["code_module", "code_presentation", "id_student"]).agg(
            clicks_last_14d=("sum_click", "sum"),
            active_days_last_14d=("date", "nunique")
        ).reset_index()

        # Resource Category breakdown
        cat_pivot = vle_cutoff.pivot_table(
            index=["code_module", "code_presentation", "id_student"],
            columns="activity_type",
            values="sum_click",
            aggfunc="sum",
            fill_value=0
        ).reset_index()

        # Standardize category column names
        for c in ["forumng", "oucontent", "resource", "subpage", "url", "quiz", "homepage"]:
            if c not in cat_pivot.columns:
                cat_pivot[c] = 0
            cat_pivot = cat_pivot.rename(columns={c: f"clicks_{c}"})

        other_cols = [c for c in cat_pivot.columns if c not in ["code_module", "code_presentation", "id_student",
                                                               "clicks_forumng", "clicks_oucontent", "clicks_resource",
                                                               "clicks_subpage", "clicks_url", "clicks_quiz", "clicks_homepage"]]
        cat_pivot["clicks_other"] = cat_pivot[other_cols].sum(axis=1) if other_cols else 0
        cat_cols = ["code_module", "code_presentation", "id_student",
                    "clicks_forumng", "clicks_oucontent", "clicks_resource",
                    "clicks_subpage", "clicks_url", "clicks_quiz", "clicks_homepage", "clicks_other"]
        cat_pivot = cat_pivot[cat_cols]

        # Merge VLE tables
        df_feat = eligible.merge(agg_cum, on=["code_module", "code_presentation", "id_student"], how="left")
        df_feat = df_feat.merge(agg_pre, on=["code_module", "code_presentation", "id_student"], how="left")
        df_feat = df_feat.merge(agg_7d, on=["code_module", "code_presentation", "id_student"], how="left")
        df_feat = df_feat.merge(agg_14d, on=["code_module", "code_presentation", "id_student"], how="left")
        df_feat = df_feat.merge(cat_pivot, on=["code_module", "code_presentation", "id_student"], how="left")
    else:
        df_feat = eligible.copy()
        for col in ["total_clicks", "active_days", "distinct_sites", "pre_course_clicks", "pre_course_active_days",
                    "clicks_last_7d", "active_days_last_7d", "clicks_last_14d", "active_days_last_14d",
                    "clicks_forumng", "clicks_oucontent", "clicks_resource", "clicks_subpage", "clicks_url",
                    "clicks_quiz", "clicks_homepage", "clicks_other"]:
            df_feat[col] = 0
        df_feat["max_click_date"] = -1

    # Fill VLE NaNs with zero (dormant/inactive students)
    fill_zero_cols = [
        "total_clicks", "active_days", "distinct_sites",
        "pre_course_clicks", "pre_course_active_days",
        "clicks_last_7d", "active_days_last_7d", "clicks_last_14d", "active_days_last_14d",
        "clicks_forumng", "clicks_oucontent", "clicks_resource", "clicks_subpage",
        "clicks_url", "clicks_quiz", "clicks_homepage", "clicks_other"
    ]
    for c in fill_zero_cols:
        if c in df_feat.columns:
            df_feat[c] = df_feat[c].fillna(0)

    # Derived VLE momentum features
    df_feat["clicks_per_active_day"] = df_feat["total_clicks"] / (df_feat["active_days"] + 1e-5)
    df_feat["days_since_last_click"] = np.where(
        df_feat["max_click_date"].isna() | (df_feat["max_click_date"] < 0),
        cutoff_day + 30,  # Never clicked in course: high inactivity penalty
        cutoff_day - df_feat["max_click_date"]
    )
    df_feat["click_ratio_7d_to_14d"] = df_feat["clicks_last_7d"] / (df_feat["clicks_last_14d"] + 1e-5)
    df_feat["active_day_ratio_7d_to_14d"] = df_feat["active_days_last_7d"] / (df_feat["active_days_last_14d"] + 1e-5)
    df_feat["recent_to_total_click_share"] = df_feat["clicks_last_14d"] / (df_feat["total_clicks"] + 1e-5)

    # 3. Assessment Participation Features (scheduled date <= cutoff_day)
    due_assess = assessments[(assessments["date"].notna()) & (assessments["date"] <= cutoff_day)]
    due_per_pres = due_assess.groupby(["code_module", "code_presentation"])["id_assessment"].count().rename("assessments_due").reset_index()
    df_feat = df_feat.merge(due_per_pres, on=["code_module", "code_presentation"], how="left")
    df_feat["assessments_due"] = df_feat["assessments_due"].fillna(0)

    # Submissions strictly submitted on or before cutoff_day
    sub_cutoff = student_assessment[
        student_assessment["date_submitted"].notna() & (student_assessment["date_submitted"] <= cutoff_day)
    ].merge(due_assess[["id_assessment", "code_module", "code_presentation", "date"]], on="id_assessment", how="inner")

    if "is_banked" in sub_cutoff.columns:
        sub_cutoff = sub_cutoff[sub_cutoff["is_banked"] == 0]

    if not sub_cutoff.empty:
        sub_cutoff["delay"] = sub_cutoff["date_submitted"] - sub_cutoff["date"]
        sub_cutoff["is_late"] = (sub_cutoff["delay"] > 0).astype(int)

        agg_sub = sub_cutoff.groupby(["code_module", "code_presentation", "id_student"]).agg(
            assessments_submitted=("id_assessment", "count"),
            mean_submission_delay=("delay", "mean"),
            late_submissions_count=("is_late", "sum"),
            mean_assessment_score=("score", "mean")
        ).reset_index()

        df_feat = df_feat.merge(agg_sub, on=["code_module", "code_presentation", "id_student"], how="left")
    else:
        df_feat["assessments_submitted"] = 0
        df_feat["mean_submission_delay"] = 0.0
        df_feat["late_submissions_count"] = 0
        df_feat["mean_assessment_score"] = 0.0

    df_feat["assessments_submitted"] = df_feat["assessments_submitted"].fillna(0)
    df_feat["mean_submission_delay"] = df_feat["mean_submission_delay"].fillna(0.0)
    df_feat["late_submissions_count"] = df_feat["late_submissions_count"].fillna(0)
    df_feat["mean_assessment_score"] = df_feat["mean_assessment_score"].fillna(0.0)
    df_feat["missed_assessments"] = np.maximum(0, df_feat["assessments_due"] - df_feat["assessments_submitted"])
    df_feat["submission_rate"] = np.where(
        df_feat["assessments_due"] > 0,
        df_feat["assessments_submitted"] / df_feat["assessments_due"],
        1.0
    )

    # 4. Demographic & Study Background Encodings
    # Fit registration median strictly on historical training partition to prevent test distribution leakage
    train_mask = df_feat["code_presentation"].isin(["2013B", "2013J"])
    if train_mask.sum() > 0:
        reg_median = df_feat.loc[train_mask, "date_registration"].median()
    else:
        reg_median = df_feat["date_registration"].median()
    reg_val = reg_median if pd.notna(reg_median) else -15.0
    df_feat["date_registration_imputed"] = df_feat["date_registration"].fillna(reg_val)
    df_feat["gender_binary"] = (df_feat["gender"] == "M").astype(int)
    df_feat["disability_binary"] = (df_feat["disability"] == "Y").astype(int)

    age_map = {"0-35": 0, "35-55": 1, "55<=": 2}
    df_feat["age_band_ord"] = df_feat["age_band"].map(age_map).fillna(0)

    edu_map = {
        "No Formal quals": 0,
        "Lower Than A Level": 1,
        "A Level or Equivalent": 2,
        "HE Qualification": 3,
        "Post Graduate Qualification": 4
    }
    df_feat["education_ord"] = df_feat["highest_education"].map(edu_map).fillna(2)

    imd_map = {
        "0-10%": 0, "10-20%": 1, "20-30%": 2, "30-40%": 3, "40-50%": 4,
        "50-60%": 5, "60-70%": 6, "70-80%": 7, "80-90%": 8, "90-100%": 9,
        "0-10": 0, "10-20": 1, "20-30": 2, "30-40": 3, "40-50": 4,
        "50-60": 5, "60-70": 6, "70-80": 7, "80-90": 8, "90-100": 9
    }
    df_feat["imd_band_ord"] = df_feat["imd_band"].map(imd_map).fillna(5)

    print(f"  - Completed feature extraction in {time.time() - t_start:.2f}s ({len(df_feat.columns)} columns)")
    return df_feat

