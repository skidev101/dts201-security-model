"""
STEP 1: Data Preprocessing
===========================
Loads the Kaggle crime dataset and your survey data,
cleans and merges them into a single analysis-ready dataframe.
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# CONFIGURATION — Change paths here if needed
# ─────────────────────────────────────────────
KAGGLE_DATA_PATH = "../data/raw/crime_data.csv"   # Your Kaggle CSV file
SURVEY_DATA_PATH = "../data/raw/survey_data.csv"  # Your Google Form export
OUTPUT_PATH      = "../data/processed/merged_data.csv"

# ─────────────────────────────────────────────
# CAMPUS-RELEVANT PREMISE CODES
# (These are the Kaggle 'Premis Desc' values related to educational institutions)
# ─────────────────────────────────────────────
CAMPUS_PREMISES = [
    "SCHOOL INTERIOR", "SCHOOL EXTERIOR", "COLLEGE/UNIVERSITY",
    "COLLEGE CAMPUS", "UNIVERSITY CAMPUS", "SCHOOL", "CAMPUS"
]

# If none of the above match your dataset, we fall back to using ALL data
# but label it as 'general urban crime' for proxy modeling.


def load_kaggle_data(path):
    """Load and do first-pass clean on the Kaggle crime dataset."""
    print("📂 Loading Kaggle crime dataset...")

    # Try loading — if file not found, generate synthetic sample for testing
    if not os.path.exists(path):
        print("⚠️  Kaggle file not found. Generating synthetic sample data for demo...")
        return generate_synthetic_kaggle_data()

    # Load in chunks for large files (1M rows)
    chunks = []
    for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"   ✅ Loaded {len(df):,} rows from Kaggle dataset")
    return df


def preprocess_kaggle(df):
    """Clean, filter and engineer features from the Kaggle crime data."""
    print("🔧 Preprocessing Kaggle data...")

    # ── Column name normalization ──
    df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")

    # ── Parse datetime ──
    for col in ["DATE_OCC", "DATE_RPTD"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ── Extract time features ──
    if "TIME_OCC" in df.columns:
        df["TIME_OCC"] = pd.to_numeric(df["TIME_OCC"], errors="coerce")
        df["HOUR"] = (df["TIME_OCC"] // 100).clip(0, 23)
        df["TIME_CATEGORY"] = pd.cut(
            df["HOUR"],
            bins=[-1, 5, 11, 16, 20, 23],
            labels=["Late Night", "Morning", "Afternoon", "Evening", "Night"]
        )

    # ── Day of week ──
    if "DATE_OCC" in df.columns:
        df["DAY_OF_WEEK"] = df["DATE_OCC"].dt.day_name()
        df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin(["Saturday", "Sunday"]).astype(int)

    # ── Filter for campus-related premises ──
    if "PREMIS_DESC" in df.columns:
        campus_mask = df["PREMIS_DESC"].str.upper().str.contains(
            "|".join(["SCHOOL", "COLLEGE", "UNIVERSITY", "CAMPUS"]),
            na=False
        )
        campus_df = df[campus_mask].copy()
        if len(campus_df) < 1000:
            print(f"   ℹ️  Only {len(campus_df)} campus records found. Using all data as proxy.")
            campus_df = df.copy()
            campus_df["IS_CAMPUS_SPECIFIC"] = 0
        else:
            print(f"   ✅ Filtered to {len(campus_df):,} campus-related records")
            campus_df["IS_CAMPUS_SPECIFIC"] = 1
    else:
        campus_df = df.copy()
        campus_df["IS_CAMPUS_SPECIFIC"] = 0

    # ── Map crime types to our 5 campus categories ──
    def map_crime_category(desc):
        if not isinstance(desc, str):
            return "OTHER"
        desc = desc.upper()
        if any(w in desc for w in ["THEFT", "BURGLARY", "STOLEN", "ROBBERY", "PICKPOCKET"]):
            return "THEFT/ROBBERY"
        elif any(w in desc for w in ["ASSAULT", "BATTERY", "FIGHT", "ATTACK", "AGGRAVATED"]):
            return "ASSAULT/VIOLENCE"
        elif any(w in desc for w in ["SEX", "RAPE", "HARASS", "MOLEST", "INDECENT"]):
            return "SEXUAL HARASSMENT/ASSAULT"
        elif any(w in desc for w in ["VANDAL", "DAMAGE", "TRESPASS", "GRAFFITI"]):
            return "VANDALISM/TRESPASSING"
        elif any(w in desc for w in ["DRUG", "NARCO", "SUBSTANCE"]):
            return "DRUG-RELATED"
        else:
            return "OTHER"

    crime_col = next((c for c in ["CRM_CD_DESC", "CRIME_DESC", "OFFENSE"] if c in campus_df.columns), None)
    if crime_col:
        campus_df["CRIME_CATEGORY"] = campus_df[crime_col].apply(map_crime_category)
    else:
        campus_df["CRIME_CATEGORY"] = "UNKNOWN"

    # ── Severity score (1=low, 3=high) ──
    severity_map = {
        "THEFT/ROBBERY": 2,
        "ASSAULT/VIOLENCE": 3,
        "SEXUAL HARASSMENT/ASSAULT": 3,
        "VANDALISM/TRESPASSING": 1,
        "DRUG-RELATED": 2,
        "OTHER": 1,
        "UNKNOWN": 1
    }
    campus_df["SEVERITY_SCORE"] = campus_df["CRIME_CATEGORY"].map(severity_map)

    # ── Risk flag (target variable for model) ──
    campus_df["HIGH_RISK"] = (campus_df["SEVERITY_SCORE"] >= 2).astype(int)

    print(f"   ✅ Kaggle data preprocessed: {len(campus_df):,} records, {campus_df['CRIME_CATEGORY'].nunique()} crime categories")
    return campus_df


def load_survey_data(path):
    """Load and clean the Google Forms survey export."""
    print("📋 Loading survey data...")

    if not os.path.exists(path):
        print("   ⚠️  Survey file not found. Generating synthetic survey for demo...")
        return generate_synthetic_survey()

    df = pd.read_csv(path)
    print(f"   ✅ Loaded {len(df)} survey responses")
    return df


def preprocess_survey(df):
    """Standardize survey column names and encode responses."""
    print("🔧 Preprocessing survey data...")

    # Rename columns to short standard names
    # ⚠️  IMPORTANT: Update this map to match your actual Google Form column headers exactly
    rename_map = {
        # "Your Actual Column Name": "standard_name"
        "Age": "AGE",
        "Gender": "GENDER",
        "Current level": "LEVEL",
        "Residence": "RESIDENCE",
        "Have you experienced a security incident on campus in the past 12 months?": "HAD_INCIDENT",
        "If yes, what type of incident?": "INCIDENT_TYPE",
        "Where did the incident(s) occur?": "INCIDENT_LOCATION",
        "Time of day of incident(s)": "INCIDENT_TIME",
        "Are campus security patrols or vigilantes visible in your area?": "PATROL_VISIBLE",
        "How effective do you think campus security is?": "SECURITY_EFFECTIVENESS",
        "What measures will make you feel safer?": "SUGGESTIONS",
    }

    # Only rename columns that exist
    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_renames)

    # Encode binary responses
    if "HAD_INCIDENT" in df.columns:
        df["HAD_INCIDENT_BIN"] = df["HAD_INCIDENT"].str.lower().map({"yes": 1, "no": 0})

    # Encode patrol visibility
    if "PATROL_VISIBLE" in df.columns:
        df["PATROL_VISIBLE_BIN"] = df["PATROL_VISIBLE"].str.lower().map({"yes": 1, "no": 0, "sometimes": 0.5})

    # Security effectiveness is numeric (1-5)
    if "SECURITY_EFFECTIVENESS" in df.columns:
        df["SECURITY_EFFECTIVENESS"] = pd.to_numeric(df["SECURITY_EFFECTIVENESS"], errors="coerce")

    print(f"   ✅ Survey preprocessed: {len(df)} responses, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATORS (for demo/testing)
# ─────────────────────────────────────────────

def generate_synthetic_kaggle_data(n=50000):
    """Generate realistic synthetic crime data when Kaggle file is missing."""
    np.random.seed(42)
    crime_types = [
        "THEFT FROM MOTOR VEHICLE", "BURGLARY", "ASSAULT WITH DEADLY WEAPON",
        "BATTERY - SIMPLE ASSAULT", "VANDALISM - FELONY", "SEX OFFENDER",
        "ROBBERY", "VEHICLE - STOLEN", "DRUG/NARCOTIC", "TRESPASSING"
    ]
    premises = [
        "SCHOOL INTERIOR", "SCHOOL EXTERIOR", "COLLEGE/UNIVERSITY",
        "PARKING LOT", "STREET", "SIDEWALK", "PARK"
    ]
    hours = np.random.choice(range(24), n, p=[
        0.01,0.01,0.01,0.01,0.01,0.02,0.03,0.05,0.06,0.06,
        0.06,0.06,0.05,0.05,0.05,0.05,0.05,0.06,0.07,0.07,
        0.06,0.05,0.03,0.02
    ])
    dates = pd.date_range("2020-01-01", "2023-12-31", periods=n)

    df = pd.DataFrame({
        "DR_NO": range(n),
        "DATE_OCC": np.random.choice(dates, n),
        "TIME_OCC": hours * 100,
        "AREA_NAME": np.random.choice(["North", "South", "East", "West", "Central"], n),
        "CRM_CD_DESC": np.random.choice(crime_types, n, p=[0.18,0.12,0.1,0.1,0.08,0.07,0.1,0.1,0.1,0.05]),
        "PREMIS_DESC": np.random.choice(premises, n, p=[0.2,0.2,0.15,0.15,0.1,0.1,0.1]),
        "VICT_AGE": np.random.randint(17, 65, n),
        "VICT_SEX": np.random.choice(["M", "F"], n),
        "LAT": np.random.uniform(34.0, 34.3, n),
        "LON": np.random.uniform(-118.5, -118.2, n),
    })
    df["HOUR"] = hours
    df["DAY_OF_WEEK"] = pd.DatetimeIndex(df["DATE_OCC"]).day_name()
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin(["Saturday","Sunday"]).astype(int)
    df["TIME_CATEGORY"] = pd.cut(df["HOUR"], bins=[-1,5,11,16,20,23],
        labels=["Late Night","Morning","Afternoon","Evening","Night"])
    return df


def generate_synthetic_survey(n=30):
    """Generate synthetic survey data matching your Google Form structure."""
    np.random.seed(7)
    return pd.DataFrame({
        "AGE": np.random.choice(["18-20","21-23","24-26","27+"], n, p=[0.4,0.35,0.15,0.1]),
        "GENDER": np.random.choice(["Male","Female","Other"], n, p=[0.45,0.5,0.05]),
        "LEVEL": np.random.choice(["100","200","300","400","500"], n),
        "RESIDENCE": np.random.choice(["On-campus","Off-campus"], n, p=[0.55,0.45]),
        "HAD_INCIDENT": np.random.choice(["Yes","No"], n, p=[0.6,0.4]),
        "INCIDENT_TYPE": np.random.choice(
            ["Theft","Physical assault","Sexual harassment","Vandalism","Drug-related","None"], n),
        "INCIDENT_LOCATION": np.random.choice(
            ["Hostel","Parking lot","Library","Lecture hall","Sports field","Campus gate"], n),
        "INCIDENT_TIME": np.random.choice(["Morning","Afternoon","Evening","Night","Late Night"], n),
        "PATROL_VISIBLE": np.random.choice(["Yes","No","Sometimes"], n, p=[0.2,0.5,0.3]),
        "SECURITY_EFFECTIVENESS": np.random.randint(1, 6, n),
        "SUGGESTIONS": np.random.choice([
            "More lighting","Increase patrols","Install CCTV",
            "Emergency call points","Better access control","Student escort service"
        ], n),
    })


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run():
    os.makedirs("../data/processed", exist_ok=True)

    kaggle_raw = load_kaggle_data(KAGGLE_DATA_PATH)
    kaggle_clean = preprocess_kaggle(kaggle_raw)

    survey_raw = load_survey_data(SURVEY_DATA_PATH)
    survey_clean = preprocess_survey(survey_raw)

    # Save both
    kaggle_clean.to_csv("../data/processed/kaggle_clean.csv", index=False)
    survey_clean.to_csv("../data/processed/survey_clean.csv", index=False)

    print("\n✅ STEP 1 COMPLETE!")
    print(f"   Kaggle: {len(kaggle_clean):,} records → data/processed/kaggle_clean.csv")
    print(f"   Survey: {len(survey_clean)} records   → data/processed/survey_clean.csv")
    return kaggle_clean, survey_clean


if __name__ == "__main__":
    run()
