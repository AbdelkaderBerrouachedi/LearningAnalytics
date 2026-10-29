"""
Data loading, verification, and audit module for OULAD.
Enforces cohort eligibility and points-in-time constraints.
"""
import os
import time
import zipfile
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

UCI_OULAD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00349/OULAD.zip"

def download_oulad(target_dir: Path) -> None:
    """Download OULAD dataset from UCI repository and unpack into target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "oulad_uci.zip"

    print(f"Downloading OULAD from {UCI_OULAD_URL} -> {zip_path}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(UCI_OULAD_URL, headers=headers)

    with urllib.request.urlopen(req, timeout=120) as resp, open(zip_path, "wb") as out_file:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 2 * 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                print(f"Progress: {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB ({downloaded/total*100:.1f}%)", end="\r")
            else:
                print(f"Downloaded: {downloaded / 1e6:.1f} MB", end="\r")

    print(f"\nDownload completed ({downloaded / 1e6:.2f} MB). Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    print(f"Dataset successfully unpacked to {target_dir}")

def load_raw_data(data_dir: Path):
    """Load raw OULAD tables and parse numeric identifiers."""
    courses = pd.read_csv(data_dir / "courses.csv", na_values=["?", ""])
    reg = pd.read_csv(data_dir / "studentRegistration.csv", na_values=["?", ""])
    info = pd.read_csv(data_dir / "studentInfo.csv", na_values=["?", ""])
    vle = pd.read_csv(data_dir / "vle.csv", na_values=["?", ""])
    assessments = pd.read_csv(data_dir / "assessments.csv", na_values=["?", ""])
    student_assessment = pd.read_csv(data_dir / "studentAssessment.csv", na_values=["?", ""])

    reg["date_registration"] = pd.to_numeric(reg["date_registration"], errors="coerce")
    reg["date_unregistration"] = pd.to_numeric(reg["date_unregistration"], errors="coerce")
    assessments["date"] = pd.to_numeric(assessments["date"], errors="coerce")
    student_assessment["date_submitted"] = pd.to_numeric(student_assessment["date_submitted"], errors="coerce")
    student_assessment["score"] = pd.to_numeric(student_assessment["score"], errors="coerce")

    # Base student merge
    merged = pd.merge(info, reg, on=["code_module", "code_presentation", "id_student"], how="left")
    merged = pd.merge(merged, courses[["code_module", "code_presentation", "module_presentation_length"]],
                      on=["code_module", "code_presentation"], how="left")

    # Audit exclusions:
    # 1. Unregistered prior to Day 0 (pre-course drops)
    # 2. Withdrawn without recorded date_unregistration (93 corrupt records)
    valid_mask = ~((merged["date_unregistration"] < 0) |
                   ((merged["final_result"] == "Withdrawn") & merged["date_unregistration"].isna()))

    base_students = merged[valid_mask].copy()

    # Load studentVle
    svle_path = data_dir / "studentVle.csv"
    if svle_path.exists():
        t0 = time.time()
        svle = pd.read_csv(svle_path, na_values=["?", ""],
                           usecols=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"])
        site_type_map = dict(zip(vle["id_site"], vle["activity_type"]))
        svle["activity_type"] = svle["id_site"].map(site_type_map).fillna("other")
        print(f"Loaded studentVle ({len(svle):,} rows) in {time.time() - t0:.2f}s")
    else:
        svle = pd.DataFrame()

    return base_students, svle, assessments, student_assessment, courses

def generate_synthetic_demo_data(target_dir: Path, n_students: int = 600, random_seed: int = 42) -> None:
    """
    Generate synthetic schema-compliant OULAD datasets for rapid pipeline testing
    and verifiable demonstration without requiring external download.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(random_seed)

    presentations = ["2013B", "2013J", "2014B", "2014J"]
    modules = ["AAA", "BBB", "CCC", "DDD"]

    # 1. courses.csv
    courses_data = []
    for m in modules:
        for p in presentations:
            courses_data.append({
                "code_module": m,
                "code_presentation": p,
                "module_presentation_length": rng.choice([240, 260, 268])
            })
    courses_df = pd.DataFrame(courses_data)
    courses_df.to_csv(target_dir / "courses.csv", index=False)

    # 2. studentInfo and studentRegistration
    student_records = []
    reg_records = []
    student_ids = np.arange(100000, 100000 + n_students)

    for i, s_id in enumerate(student_ids):
        p = presentations[i % len(presentations)]
        m = modules[i % len(modules)]
        
        # 16% withdrawal rate
        is_withdrawn = rng.rand() < 0.16
        if is_withdrawn:
            unreg_day = rng.choice([rng.randint(5, 30), rng.randint(31, 70), rng.randint(71, 150)])
            final_res = "Withdrawn"
        else:
            unreg_day = np.nan
            final_res = rng.choice(["Pass", "Distinction", "Fail"], p=[0.6, 0.2, 0.2])

        reg_day = rng.randint(-60, 0)
        
        student_records.append({
            "code_module": m,
            "code_presentation": p,
            "id_student": s_id,
            "gender": rng.choice(["M", "F"]),
            "region": rng.choice(["London", "South Region", "North Region", "Scotland"]),
            "highest_education": rng.choice(["A Level or Equivalent", "HE Qualification", "Lower Than A Level"]),
            "imd_band": rng.choice(["0-10%", "20-30%", "40-50%", "70-80%", "90-100%"]),
            "age_band": rng.choice(["0-35", "35-55", "55<="]),
            "num_of_prev_attempts": rng.choice([0, 1, 2], p=[0.85, 0.12, 0.03]),
            "studied_credits": rng.choice([60, 120, 180], p=[0.65, 0.30, 0.05]),
            "disability": rng.choice(["N", "Y"], p=[0.9, 0.1]),
            "final_result": final_res
        })

        reg_records.append({
            "code_module": m,
            "code_presentation": p,
            "id_student": s_id,
            "date_registration": reg_day,
            "date_unregistration": unreg_day
        })

    pd.DataFrame(student_records).to_csv(target_dir / "studentInfo.csv", index=False)
    pd.DataFrame(reg_records).to_csv(target_dir / "studentRegistration.csv", index=False)

    # 3. vle.csv
    vle_records = []
    site_types = ["forumng", "oucontent", "resource", "subpage", "url", "quiz", "homepage"]
    for s_idx in range(1, 101):
        vle_records.append({
            "id_site": s_idx,
            "code_module": rng.choice(modules),
            "code_presentation": rng.choice(presentations),
            "activity_type": rng.choice(site_types),
            "week_from": np.nan,
            "week_to": np.nan
        })
    pd.DataFrame(vle_records).to_csv(target_dir / "vle.csv", index=False)

    # 4. assessments.csv
    assess_records = []
    assess_id = 1
    for m in modules:
        for p in presentations:
            for d in [18, 32, 54, 85]:
                assess_records.append({
                    "code_module": m,
                    "code_presentation": p,
                    "id_assessment": assess_id,
                    "assessment_type": "TMA",
                    "date": d,
                    "weight": 25.0
                })
                assess_id += 1
    assess_df = pd.DataFrame(assess_records)
    assess_df.to_csv(target_dir / "assessments.csv", index=False)

    # 5. studentAssessment.csv
    sa_records = []
    for s in student_records:
        mod_assess = assess_df[(assess_df["code_module"] == s["code_module"]) &
                               (assess_df["code_presentation"] == s["code_presentation"])]
        for _, a in mod_assess.iterrows():
            if rng.rand() > 0.15:  # 85% submission
                sa_records.append({
                    "id_assessment": a["id_assessment"],
                    "id_student": s["id_student"],
                    "date_submitted": a["date"] + rng.randint(-3, 5),
                    "is_banked": 0,
                    "score": max(0, min(100, rng.normal(70, 15)))
                })
    pd.DataFrame(sa_records).to_csv(target_dir / "studentAssessment.csv", index=False)

    # 6. studentVle.csv
    svle_records = []
    for s in student_records:
        n_clicks = rng.randint(20, 150)
        for _ in range(n_clicks):
            day = rng.randint(-15, 100)
            svle_records.append({
                "code_module": s["code_module"],
                "code_presentation": s["code_presentation"],
                "id_student": s["id_student"],
                "id_site": rng.randint(1, 101),
                "date": day,
                "sum_click": rng.randint(1, 15)
            })
    pd.DataFrame(svle_records).to_csv(target_dir / "studentVle.csv", index=False)

    print(f"Generated synthetic demo OULAD dataset ({n_students} students) in {target_dir}")

