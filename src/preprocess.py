import pandas as pd
import os



KAGGLE_DATA_PATH = "data/raw/crime_data.csv"
SURVEY_DATA_PATH = "data/raw/survey_data.csv"  # google form data
OUTPUT_PATH      = "data/processed/merged_data.csv"

CAMPUS_PREMISES = [
    "SCHOOL INTERIOR", "SCHOOL EXTERIOR", "COLLEGE/UNIVERSITY",
    "COLLEGE CAMPUS", "UNIVERSITY CAMPUS", "SCHOOL", "CAMPUS"
]


def load_dataset(path):
  print(f"Loading crime dataset from {path}")

  if not os.path.exists(path):
    print(f"Path {path} does not exist")
    return

  # load datasets in chunk for large files
  chunks = []
  for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
    chunks.append(chunk)
  df = pd.concat(chunks, ignore_index=True)
  print(f"Loaded {len(df)} from dataset")
  return df

def preprocess(df):
  print("Preprocessing data...")

  # normalize column names
  df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

  # parse data and time columns
  for col in ['DATA_OCC', 'DATE_RPTD']:
    if col in df.columns:
      df[col] = pd.to_datetime(df[col], errors='coerce')

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
            print(f"   Only {len(campus_df)} campus records found. Using all data as proxy.")
            campus_df = df.copy()
            campus_df["IS_CAMPUS_SPECIFIC"] = 0
        else:
            print(f"  Filtered to {len(campus_df):,} campus-related records")
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

    crime_col = next((c for c in ["crm_cd_desc", "crime_desc", "offense"] if c in campus_df.columns), None)
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

    print("\nCrime category distribution:")
    print(campus_df["CRIME_CATEGORY"].value_counts())

    print("\nHIGH_RISK distribution:")
    print(campus_df["HIGH_RISK"].value_counts())

    print(f"  Data preprocessed: {len(campus_df):,} records, {campus_df['CRIME_CATEGORY'].nunique()} crime categories")
    return campus_df




# MAIN
def run():
    os.makedirs("data/processed", exist_ok=True)

    kaggle_raw = load_dataset(KAGGLE_DATA_PATH)
    kaggle_clean = preprocess(kaggle_raw)


    # Save both
    kaggle_clean.to_csv("data/processed/kaggle_clean.csv", index=False)

    print("\n STEP 1 COMPLETE!")
    print(f"   Kaggle: {len(kaggle_clean):,} records → data/processed/kaggle_clean.csv")
    return kaggle_clean


if __name__ == "__main__":
    run()