"""
STEP 2: Exploratory Data Analysis (EDA) + Visualizations
=========================================================
Generates all charts and statistics used in the final report.
Run this AFTER step1_preprocess.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

# ─── Style ───
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
COLORS = ["#1a3a5c", "#2e86ab", "#a23b72", "#f18f01", "#c73e1d", "#3b1f2b"]
PLOT_DIR = "../outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def load_data():
    kaggle = pd.read_csv("../data/processed/kaggle_clean.csv", low_memory=False)
    survey = pd.read_csv("../data/processed/survey_clean.csv")
    return kaggle, survey


# ══════════════════════════════
#   KAGGLE CRIME DATA CHARTS
# ══════════════════════════════

def plot_crime_category_distribution(df):
    counts = df["CRIME_CATEGORY"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(counts.index, counts.values, color=COLORS[:len(counts)], edgecolor="white")
    ax.bar_label(bars, fmt="%,.0f", padding=4, fontsize=9)
    ax.set_title("Crime Category Distribution on Campus", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Incidents")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/01_crime_categories.png")
    plt.close()
    print("   ✅ Plot 1: Crime category distribution")


def plot_time_heatmap(df):
    if "HOUR" not in df.columns or "DAY_OF_WEEK" not in df.columns:
        return
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    df["DAY_OF_WEEK"] = pd.Categorical(df["DAY_OF_WEEK"], categories=day_order, ordered=True)
    pivot = df.groupby(["DAY_OF_WEEK", "HOUR"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3, cbar_kws={"label": "Incident Count"})
    ax.set_title("Crime Incidents by Day of Week and Hour", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Hour of Day (24hr)")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/02_time_heatmap.png")
    plt.close()
    print("   ✅ Plot 2: Time heatmap")


def plot_severity_pie(df):
    severity_labels = {1: "Low Risk", 2: "Medium Risk", 3: "High Risk"}
    counts = df["SEVERITY_SCORE"].map(severity_labels).value_counts()
    colors = ["#2e86ab", "#f18f01", "#c73e1d"]
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    ax.set_title("Incident Risk Level Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/03_severity_pie.png")
    plt.close()
    print("   ✅ Plot 3: Severity distribution")


def plot_time_category_bar(df):
    if "TIME_CATEGORY" not in df.columns:
        return
    order = ["Morning", "Afternoon", "Evening", "Night", "Late Night"]
    counts = df["TIME_CATEGORY"].value_counts().reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color=COLORS, edgecolor="white", width=0.6)
    ax.set_title("Incidents by Time of Day", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Incidents")
    ax.set_xlabel("Time of Day")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts)*0.01, f"{v:,}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/04_time_of_day.png")
    plt.close()
    print("   ✅ Plot 4: Time-of-day breakdown")


# ══════════════════════════════
#   SURVEY DATA CHARTS
# ══════════════════════════════

def plot_survey_incident_experience(survey):
    if "HAD_INCIDENT" not in survey.columns:
        return
    counts = survey["HAD_INCIDENT"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(counts.index, counts.values,
           color=["#c73e1d" if x=="Yes" else "#2e86ab" for x in counts.index],
           width=0.5, edgecolor="white")
    ax.set_title("Students Who Experienced\nSecurity Incidents (12 months)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Students")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.2, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/05_survey_incident_experience.png")
    plt.close()
    print("   ✅ Plot 5: Survey incident experience")


def plot_survey_security_effectiveness(survey):
    if "SECURITY_EFFECTIVENESS" not in survey.columns:
        return
    counts = survey["SECURITY_EFFECTIVENESS"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(counts.index.astype(str), counts.values,
           color=["#c73e1d","#f18f01","#f1c40f","#2ecc71","#1a3a5c"][:len(counts)],
           edgecolor="white", width=0.6)
    ax.set_title("Perceived Security Effectiveness\n(1=Not Effective, 5=Very Effective)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Respondents")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/06_survey_effectiveness.png")
    plt.close()
    print("   ✅ Plot 6: Security effectiveness rating")


def plot_survey_incident_locations(survey):
    if "INCIDENT_LOCATION" not in survey.columns:
        return
    counts = survey["INCIDENT_LOCATION"].value_counts().head(8)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(counts.index, counts.values, color=COLORS[:len(counts)], edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_title("Where Incidents Occurred (Survey)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Reports")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/07_survey_locations.png")
    plt.close()
    print("   ✅ Plot 7: Incident locations from survey")


def plot_suggestions_wordcloud_alternative(survey):
    """Bar chart of safety measure suggestions (no wordcloud library needed)."""
    if "SUGGESTIONS" not in survey.columns:
        return
    counts = survey["SUGGESTIONS"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(counts.index, counts.values, color=COLORS[:len(counts)], edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_title("Student Suggestions for Improved Safety", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Students")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/08_safety_suggestions.png")
    plt.close()
    print("   ✅ Plot 8: Safety suggestions")


def plot_patrol_visibility(survey):
    if "PATROL_VISIBLE" not in survey.columns:
        return
    counts = survey["PATROL_VISIBLE"].value_counts()
    colors = ["#2ecc71" if x=="Yes" else "#c73e1d" if x=="No" else "#f18f01" for x in counts.index]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)
    ax.set_title("Patrol / Vigilante Visibility\n(Survey Responses)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Students")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.1, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/09_patrol_visibility.png")
    plt.close()
    print("   ✅ Plot 9: Patrol visibility")


# ══════════════════════════════
#   SUMMARY STATS
# ══════════════════════════════

def compute_summary_stats(kaggle, survey):
    stats = {
        "total_kaggle_records": len(kaggle),
        "total_survey_responses": len(survey),
        "most_common_crime": kaggle["CRIME_CATEGORY"].mode()[0] if "CRIME_CATEGORY" in kaggle.columns else "N/A",
        "pct_high_risk": round(kaggle["HIGH_RISK"].mean() * 100, 1) if "HIGH_RISK" in kaggle.columns else "N/A",
        "peak_hour": int(kaggle["HOUR"].mode()[0]) if "HOUR" in kaggle.columns else "N/A",
        "survey_incident_rate": round(
            (survey["HAD_INCIDENT"].str.lower()=="yes").mean() * 100, 1
        ) if "HAD_INCIDENT" in survey.columns else "N/A",
        "avg_security_rating": round(
            survey["SECURITY_EFFECTIVENESS"].mean(), 2
        ) if "SECURITY_EFFECTIVENESS" in survey.columns else "N/A",
    }
    print("\n📊 KEY STATISTICS:")
    for k, v in stats.items():
        print(f"   {k}: {v}")
    return stats


def run():
    print("📊 Running EDA and generating plots...\n")
    kaggle, survey = load_data()

    # Kaggle plots
    plot_crime_category_distribution(kaggle)
    plot_time_heatmap(kaggle)
    plot_severity_pie(kaggle)
    plot_time_category_bar(kaggle)

    # Survey plots
    plot_survey_incident_experience(survey)
    plot_survey_security_effectiveness(survey)
    plot_survey_incident_locations(survey)
    plot_suggestions_wordcloud_alternative(survey)
    plot_patrol_visibility(survey)

    stats = compute_summary_stats(kaggle, survey)

    print(f"\n✅ STEP 2 COMPLETE! — {len(os.listdir(PLOT_DIR))} plots saved to outputs/plots/")
    return stats


if __name__ == "__main__":
    run()
