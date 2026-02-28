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
  for col in ['date_occ', 'date_report', 'date_rptd']:
    if col in df.columns:
      df[col] = pd.to_datetime(df[col], errors='coerce')

  # ── Extract time features ──
    if "time_occ" in df.columns:
        df["time_occ"] = pd.to_numeric(df["time_occ"], errors="coerce")
        df["hour"] = (df["time_occ"] // 100).clip(0, 23)
        df["TIME_CATEGORY"] = pd.cut(
            df["hour"],
            bins=[-1, 5, 11, 16, 20, 23],
            labels=["Late Night", "Morning", "Afternoon", "Evening", "Night"]
        )

  # ── Day of week ──
    if "date_occ" in df.columns:
        df["day_of_week"] = df["date_occ"].dt.day_name()
        df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

  # ── Filter for campus-related premises ──
    if "premis_desc" in df.columns:
        campus_mask = df["premis_desc"].str.upper().str.contains(
            "|".join(["SCHOOL", "COLLEGE", "UNIVERSITY", "CAMPUS"]),
            na=False
        )
        campus_df = df[campus_mask].copy()
        if len(campus_df) < 10:
            print(f"   Only {len(campus_df)} campus records found. Using all data as proxy.")
            campus_df = df.copy()
            campus_df["is_campus_specific"] = 0
        else:
            print(f"  Filtered to {len(campus_df):,} campus-related records")
            campus_df["is_campus_specific"] = 1
    else:
        campus_df = df.copy()
        campus_df["is_campus_specific"] = 0


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
        campus_df["crime_category"] = campus_df[crime_col].apply(map_crime_category)
    else:
        campus_df["crime_category"] = "UNKNOWN"

    crime_risk_rate = campus_df.groupby("crime_category").size()
    high_freq_threshold = crime_risk_rate.quantile(0.75)

    high_risk_categories = crime_risk_rate[
        crime_risk_rate >= high_freq_threshold
    ].index

    campus_df["high_risk"] = campus_df["crime_category"].isin(
        high_risk_categories
    ).astype(int)

    print("\nCrime category distribution:")
    print(campus_df["crime_category"].value_counts())

    print("\nHIGH_RISK distribution:")
    print(campus_df["high_risk"].value_counts())

    print(f"  Data preprocessed: {len(campus_df):,} records, {campus_df['crime_category'].nunique()} crime categories")
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